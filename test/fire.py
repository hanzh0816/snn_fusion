import cv2
import torch
from sympy import I

import core.data.dataset
from core.config.config import get_cfg
from core.data.build import (
    _get_detection_dataset_dicts,
    build_detection_train_loader,
    build_detection_val_loader,
)
from core.data.catalog import DatasetCatalog
from core.data.dataset_mapper import DatasetMapper, MapDataset
from core.data.dataset.dsec.dsec import dsec_collate_fn
from core.evaluation.build import build_evaluator
from core.evaluation.dsec_eval import DSECEvaluator
from core.model import build_model
from core.model.build import build_model
from train import TrainingModule

cfg = get_cfg()
cfg.merge_from_file("configs/spike_yolo/spike_yolo_nano.yaml")

model = TrainingModule.load_from_checkpoint(
    "/home/hzh/code/rgb_event_fusion/snn_fusion/work_dirs/exp_n_lr1e-3_mosaic_ep100_bs40_vthrs1.0_preprocess_2025-02-08_20-34/ckpt/cfg.MODEL.NAME=0-epoch=09-val_loss=18.00.ckpt",
    cfg=cfg,
)


Dataloader = build_detection_val_loader(cfg)

for data in Dataloader:
    data['events'] = data['events'].to(model.device)
    loss = model.model.loss(data)
    break
 