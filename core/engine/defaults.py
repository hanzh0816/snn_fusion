import argparse
import logging
import os
import sys
from argparse import Namespace
from datetime import datetime

import torch
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from core.config import CfgNode
from core.utils import comm
from core.utils.file_io import PathManager
from core.utils.logger import setup_logger


# 默认参数解析器
def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(  #  创建一个ArgumentParser对象，用于解析命令行参数
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,  #  使用RawDescriptionHelpFormatter格式化帮助信息，保留原始格式
    )
    parser.add_argument("config_file", default="", metavar="FILE", help="path to config file")
    # parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--exp-name", type=str, default="default", help="experiment name")
    parser.add_argument("--use-wandb", action="store_true", help="Whether to use wandb")
    parser.add_argument("--debug", action="store_true", help="run in debug mode")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--epochs", type=int, default=-1, help="training epochs")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    # parser.add_argument(  #  添加一个参数，用于显示分布式训练的详细信息，链接指向PyTorch官方文档
    #     "opts",
    #     help="""
    #         Modify config options at the end of the command. For Yacs configs, use
    #         space-separated "PATH.KEY VALUE" pairs.
    #         For python-based LazyConfig, use "path.key=value".
    #     """.strip(),
    #     default=None,
    #     nargs=argparse.REMAINDER,
    # )  #  接收剩余的命令行参数
    return parser


# 基础设置,包括日志设置,随机种子,精度设置等
def default_setup(cfg: CfgNode, args: Namespace) -> CfgNode:
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """

    # tag: 初始化ddp,只有多卡训练才需要
    if args.num_gpus > 1:
        comm.init_process_group()

    # output路径从exp_name中获取,添加时间戳后缀
    frozen = cfg.is_frozen()
    cfg.defrost()

    exp_base_name = exp_name = _try_get_key(cfg, "EXP_NAME", "CFG_NAME", default="exp")
    epochs = cfg.TRAINING.NUM_EPOCHS
    # merge from args
    if args.exp_name != "default":
        exp_base_name = exp_name = args.exp_name

    if args.epochs != -1:
        epochs = args.epochs
    cfg.TRAINING.NUM_EPOCHS = epochs

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_name = "{}_{}".format(exp_name, current_time)
    output_dir = os.path.join("work_dirs", exp_name)
    cfg.EXP_NAME = exp_name
    cfg.EXP_BASE_NAME = exp_base_name
    cfg.OUTPUT_DIR = output_dir

    if not PathManager.exists(output_dir):
        PathManager.mkdirs(output_dir)

    # create pl log path
    if not PathManager.exists(os.path.join(cfg.OUTPUT_DIR, "lightning")):
        PathManager.mkdirs(os.path.join(cfg.OUTPUT_DIR, "lightning"))

    # update ddp params
    cfg.NUM_GPUS = args.num_gpus
    cfg.NUM_MACHINES = args.num_machines

    # auto scale batch_size & lr
    cfg = auto_scale_workers(cfg, num_workers=args.num_gpus)

    # set seed
    seed = _try_get_key(cfg, "SEED", "train.seed", default=None)
    seed_everything(seed=seed)

    if frozen:
        cfg.freeze()

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )

    fp32_precision = _try_get_key(cfg, "FLOAT32_PRECISION", "train.float32_precision", default="")
    if fp32_precision != "":
        _set_float32_precision(fp32_precision)

    return cfg


# 设置logger,输出基本信息
def default_logging(cfg: CfgNode, args: Namespace) -> None:

    output_dir = cfg.OUTPUT_DIR
    rank = comm.get_rank()
    # tag: 配置`core`父节点的logger的handler和formatter
    logger = setup_logger(output_dir, name="core")
    # tag: 配置`tools`父节点的logger的handler和formatter(同上)
    _ = setup_logger(output_dir, name="tools")

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))

    logger.info("Command line arguments: " + str(args))

    logger.info(f"Output dir : {cfg.OUTPUT_DIR}")

    logger.info(
        f"Auto-scaling the config to batch_size={cfg.DATALOADER.TOTAL_BATCH_SIZE}, learning_rate={cfg.SOLVER.BASE_LR}, "
        f"max_iter={cfg.SOLVER.MAX_ITER}, warmup={cfg.SOLVER.WARMUP_ITERS}."
    )

    # 完整配置文件保存到output_dir
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            logger.debug("Running with full config:\n{}".format(_highlight(cfg.dump(), ".yaml")))
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            raise NotImplementedError("Only CfgNode is supported for now!")
        logger.info("Full config saved to {}".format(path))


# 根据num_workers自动调整bs/lr/iters配置
def auto_scale_workers(cfg, num_workers: int):
    # todo: 修改docs
    """
    When the config is defined for certain number of workers (according to
    ``cfg.DATALOADER.REFERENCE_WORLD_SIZE``) that's different from the number of
    workers currently in use, returns a new cfg where the total batch size
    is scaled so that the per-GPU batch size stays the same as the
    original ``BATCH_SIZE // REFERENCE_WORLD_SIZE``.

    Other config options are also scaled accordingly:
    * training steps and warmup steps are scaled inverse proportionally.
    * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.

    For example, with the original config like the following:

    .. code-block:: yaml

        TOTAL_BATCH_SIZE: 16
        BASE_LR: 0.1
        REFERENCE_WORLD_SIZE: 8
        MAX_ITER: 5000
        STEPS: (4000,)
        CHECKPOINT_PERIOD: 1000

    When this config is used on 16 GPUs instead of the reference number 8,
    calling this method will return a new config with:

    .. code-block:: yaml

        TOTAL_BATCH_SIZE: 32
        BASE_LR: 0.2
        REFERENCE_WORLD_SIZE: 16
        MAX_ITER: 2500
        STEPS: (2000,)
        CHECKPOINT_PERIOD: 500

    Note that both the original config and this new config can be trained on 16 GPUs.
    It's up to user whether to enable this feature (by setting ``REFERENCE_WORLD_SIZE``).

    Returns:
        CfgNode: a new config. Same as original if ``cfg.DATALOADER.REFERENCE_WORLD_SIZE==0``.
    """

    reference_world_size = cfg.DATALOADER.REFERENCE_WORLD_SIZE
    reference_batch_size = cfg.DATALOADER.REFERENCE_BATCH_SIZE

    reference_total_batch_size = reference_batch_size * reference_world_size
    total_batch_size = cfg.DATALOADER.BATCH_SIZE * num_workers
    if reference_total_batch_size == num_workers:
        return cfg
    cfg = cfg.clone()
    frozen = cfg.is_frozen()
    cfg.defrost()

    scale = total_batch_size / reference_total_batch_size
    cfg.DATALOADER.TOTAL_BATCH_SIZE = total_batch_size
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
    cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
    cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
    cfg.DATALOADER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant

    if frozen:
        cfg.freeze()
    return cfg


def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.formatters import Terminal256Formatter
    from pygments.lexers import Python3Lexer, YamlLexer

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default


# adapted from:
# https://github.com/pytorch/tnt/blob/ebda066f8f55af6a906807d35bc829686618074d/torchtnt/utils/device.py#L328-L346
def _set_float32_precision(precision: str = "high") -> None:
    """Sets the precision of float32 matrix multiplications and convolution operations.

    For more information, see the PyTorch docs:
    - https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    - https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.allow_tf32

    Args:
        precision: The setting to determine which datatypes to use for matrix
        multiplication and convolution operations.
    """
    if not (torch.cuda.is_available()):  # Not relevant for non-CUDA devices
        return
    # set precision for matrix multiplications
    torch.set_float32_matmul_precision(precision)
    # set precision for convolution operations
    if precision == "highest":
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
