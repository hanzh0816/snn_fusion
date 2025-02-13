import logging
import os
import profile
import time
import warnings
import weakref
from argparse import Namespace
from logging import Logger
from typing import Any, Dict, List, Optional, OrderedDict, Union

import torch.distributed as dist
from lightning import Trainer
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler

import core.utils.comm as comm
from core.config import get_cfg
from core.config.config import CfgNode
from core.data import build_detection_val_loader, build_detection_train_loader
from core.engine import default_argument_parser, default_setup
from core.engine.build import build_lr_scheduler, build_optimizer
from core.engine.defaults import default_logging
from core.evaluation.build import build_evaluator
from core.evaluation.evaluator import DatasetEvaluator
from core.model import build_model
from core.utils.logger import print_csv_format
from spikingjelly.clock_driven import functional

# 禁用所有 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


class TrainingModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg=self.cfg.MODEL, name=self.cfg.MODEL.NAME)
        self.evaluator: DatasetEvaluator = build_evaluator(self.cfg)

        self.hyper_params = {
            "log_name": self.cfg.OUTPUT_DIR,
            "lr": self.cfg.SOLVER.BASE_LR,
            "weight_decay": self.cfg.SOLVER.WEIGHT_DECAY,
            "batch_size": self.cfg.DATALOADER.TOTAL_BATCH_SIZE,
            "scale": self.cfg.MODEL.SCALES,
            "num_epochs": self.cfg.TRAINING.NUM_EPOCHS,
        }

        self.save_hyperparameters(self.hyper_params, ignore="cfg")

    def training_step(self, batch, batch_idx):
        log_dict = self.model.loss(batch)
        log_dict["lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.cfg.DATALOADER.BATCH_SIZE,
        )
        return log_dict

    def on_train_batch_end(self, *args, **kwargs):
        functional.reset_net(self.model)

    def on_validation_batch_end(self, *args, **kwargs):
        functional.reset_net(self.model)

    def on_test_batch_end(self, *args, **kwargs):
        functional.reset_net(self.model)

    def validation_step(self, batch, batch_idx: int) -> None:
        outputs = self.model.predict(batch)

        self.log(
            "val_loss",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=self.cfg.DATALOADER.BATCH_SIZE,
        )
        self.evaluator.process(batch, outputs)

    def on_train_epoch_start(self):
        self.trainer.datamodule.train_dataloader().dataset.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self):
        self.model.head.set_epoch(self.current_epoch)
        self.evaluator.reset()

    def on_validation_epoch_end(self):
        result = self.evaluator.evaluate()
        if comm.is_main_process():
            print_csv_format(result)
        self.log_dict(result, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        optimizer = build_optimizer(self.cfg, self.model)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        return build_detection_train_loader(self.cfg)

    def val_dataloader(self):
        return build_detection_val_loader(self.cfg)


def build_trainer(cfg: CfgNode, args: Namespace):
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath=os.path.join(cfg.OUTPUT_DIR, "ckpt"),
        filename="{cfg.MODEL.NAME}-{epoch:02d}-{val_loss:.2f}",
    )
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.WANDB.PROJECT,
            name=f"{cfg.MODEL.NAME}-{cfg.EXP_BASE_NAME}",
            save_dir=cfg.OUTPUT_DIR,
        )
    else:
        wandb_logger = None

    trainer_params = {
        "accelerator": "gpu",
        "strategy": "ddp",
        "max_epochs": cfg.TRAINING.NUM_EPOCHS,
        "num_nodes": args.num_machines,
        "devices": args.num_gpus,
        "check_val_every_n_epoch": cfg.TRAINING.EVAL_PERIOD,
        "default_root_dir": cfg.OUTPUT_DIR,  # create path in default_setup
        "logger": wandb_logger,
        "callbacks": [checkpoint_callback, CustomProgressBar(leave=True), DeviceStatsMonitor()],
        "num_sanity_val_steps": 0,
        "profiler": AdvancedProfiler(dirpath=cfg.OUTPUT_DIR, filename="profile"),
        "sync_batchnorm": cfg.TRAINING.SYNC_BN,
    }

    # todo: 添加gradient_clip参数

    if args.debug:
        trainer_params["fast_dev_run"] = 10
        trainer_params["num_sanity_val_steps"] = 10

    if cfg.SOLVER.AMP.ENABLED:
        trainer_params["precision"] = "16-mixed"

    trainer = Trainer(**trainer_params)
    return trainer


def train(trainer: Trainer, logger: Logger, cfg: CfgNode, args: Namespace):

    logger.info(f"start to train with {args.num_machines} nodes and {args.num_gpus} GPUs")

    if args.resume:
        # resume training from checkpoint
        ckpt_path = os.path.join(cfg.OUTPUT_DIR, "last.ckpt")
        logger.info(f"Resuming training from checkpoint: {ckpt_path}.")
    else:
        ckpt_path = None

    module = TrainingModule(cfg)
    data_module = DataModule(cfg)
    if args.eval_only:
        logger.info("Running inference")
        trainer.validate(module, data_module)
    else:
        logger.info("Running training")
        trainer.fit(model=module, datamodule=data_module, ckpt_path=ckpt_path)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # update output_dir, batch_size, lr, etc.
    cfg = default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    trainer = build_trainer(cfg, args)
    default_logging(cfg, args)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("tools.train")

    train(trainer, logger, cfg, args)


def invoke_main():
    parser = default_argument_parser()
    args, _ = parser.parse_known_args()
    main(args)
    dist.destroy_process_group()


if __name__ == "__main__":
    invoke_main()


# train.py configs/spike_yolo/spike_yolo_nano.yaml --num-gpus=4 --use-wandb --exp-name=exp_n_0
