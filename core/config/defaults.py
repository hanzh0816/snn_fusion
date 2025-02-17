# Copyright (c) Facebook, Inc. and its affiliates.
from sympy import false
from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./output"
# Experiment name as in the experiment log
_C.EXP_NAME = "exp"
_C.EXP_BASE_NAME = "exp"
_C.NUM_GPUS = 1
_C.NUM_NODES = 1

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.NAME = "SpikeYOLO"
_C.MODEL.DEVICE = "cuda"

# Path (a file path, or URL like detectron2://.., https://..) to a checkpoint file
# to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

# model depth,width,max_channels
_C.MODEL.SCALES = [1.00, 1.00, 512]

# num_classes of detection
_C.MODEL.NUM_CLASSES = 2

_C.MODEL.PRETRAINED = False

# -----------------------------------------------------------------------------
# Model Spike options
# -----------------------------------------------------------------------------
_C.MODEL.SPIKE_T = 4
_C.MODEL.SPIKE_CONFIG = CN()
_C.MODEL.SPIKE_CONFIG.spike_mode = "lif"
_C.MODEL.SPIKE_CONFIG.tau = 2.0
_C.MODEL.SPIKE_CONFIG.v_threshold = 1.0
_C.MODEL.SPIKE_CONFIG.v_reset = 0.0
_C.MODEL.SPIKE_CONFIG.detach_reset = True
_C.MODEL.SPIKE_CONFIG.backend = "torch"

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.EVENT = CN()
_C.MODEL.BACKBONE.EVENT.NAME = ""
_C.MODEL.BACKBONE.EVENT.IN_CHANNELS = 2
_C.MODEL.BACKBONE.EVENT.CONFIG = []
_C.MODEL.BACKBONE.EVENT.OUT_INDICES = [-1]

_C.MODEL.BACKBONE.RGB = CN()
_C.MODEL.BACKBONE.RGB.NAME = ""
_C.MODEL.BACKBONE.RGB.IN_CHANNELS = 3
_C.MODEL.BACKBONE.RGB.CONFIG = []
_C.MODEL.BACKBONE.RGB.OUT_INDICES = [-1]


# ---------------------------------------------------------------------------- #
# Neck options
# ---------------------------------------------------------------------------- #
_C.MODEL.NECK = CN()

_C.MODEL.NECK.EVENT = CN()
_C.MODEL.NECK.EVENT.NAME = ""
_C.MODEL.NECK.EVENT.IN_CHANNELS = 1024
_C.MODEL.NECK.EVENT.CONFIG = []
_C.MODEL.NECK.EVENT.FROM_INDICES = [-1]
_C.MODEL.NECK.EVENT.OUT_INDICES = [-1]

_C.MODEL.NECK.RGB = CN()
_C.MODEL.NECK.RGB.NAME = ""
_C.MODEL.NECK.RGB.IN_CHANNELS = 1024
_C.MODEL.NECK.RGB.CONFIG = []
_C.MODEL.NECK.RGB.FROM_INDICES = [-1]
_C.MODEL.NECK.RGB.OUT_INDICES = [-1]

# ---------------------------------------------------------------------------- #
# Fusion options
# ---------------------------------------------------------------------------- #
_C.MODEL.FUSION = CN()

_C.MODEL.FUSION.NAME = ""


# ---------------------------------------------------------------------------- #
# Head options
# ---------------------------------------------------------------------------- #
_C.MODEL.HEAD = CN()

_C.MODEL.HEAD.NAME = ""
_C.MODEL.HEAD.CONFIG = []
_C.MODEL.HEAD.STRIDE = [8, 16, 32]
_C.MODEL.HEAD.WEIGHT = [7.5, 0.5, 1.5]  # box,cls,iou loss weight

_C.MODEL.HEAD.NMS = CN()

_C.MODEL.HEAD.NMS.DYNAMIC_CONF = False
_C.MODEL.HEAD.NMS.EXP_K = 0.2  # conf_thres * (1-exp(k*cur_epoch)) for dynamic conf threshold
_C.MODEL.HEAD.NMS.CONF_THRES = 0.25
_C.MODEL.HEAD.NMS.IOU_THRES = 0.7
_C.MODEL.HEAD.NMS.MAX_DET = 300
_C.MODEL.HEAD.NMS.MAX_TIME_IMG = 0.5
_C.MODEL.HEAD.NMS.AGNOSTIC = False
_C.MODEL.HEAD.NMS.MULTI_LABEL = False

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TYPE = ""
# List of the dataset names for training. Must be registered in DatasetCatalog
# Samples from these datasets will be merged and used as one dataset.
_C.DATASETS.TRAIN = ""

_C.DATASETS.VAL = ""

# List of the dataset names for testing. Must be registered in DatasetCatalog
_C.DATASETS.TEST = ""

# modality of the dataset used by the model
_C.DATASETS.MODALITY = "fusion"

# transforms to apply on a sample
_C.DATASETS.TRANSFORMS = []

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True
# Options: TrainingSampler, RepeatFactorTrainingSampler
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
# Repeat threshold for RepeatFactorTrainingSampler
_C.DATALOADER.REPEAT_THRESHOLD = 0.0
# if True, take square root when computing repeating factor
_C.DATALOADER.REPEAT_SQRT = True
# Tf True, when working on datasets that have instance annotations, the
# training dataloader will filter out images without associated annotations
_C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

# Number of images per batch across all machines. This is also the number
# of training images per step (i.e. per iteration). If we use 16 GPUs
# and TOTAL_BATCH_SIZE = 32, each GPU will see 2 images per batch.
# adjusted automatically if REFERENCE_WORLD_SIZE is set.
_C.DATALOADER.TOTAL_BATCH_SIZE = 0

# Number of samples per worker see in a batch.
# To modify batch_size, set it here
_C.DATALOADER.BATCH_SIZE = 1


# The reference number of workers (GPUs) this config is meant to train with.
# It takes no effect when set to 0.
# With a non-zero value, it will be used by DefaultTrainer to compute a desired
# per-worker batch size, and then scale the other related configs (total batch size,
# learning rate, etc) to match the per-worker batch size.
# See documentation of `DefaultTrainer.auto_scale_workers` for details:

_C.DATALOADER.REFERENCE_BATCH_SIZE = 8
_C.DATALOADER.REFERENCE_WORLD_SIZE = 1

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# Options: WarmupMultiStepLR, WarmupCosineLR.
# See detectron2/solver/build.py for definition.
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

_C.SOLVER.MAX_ITER = 40000
_C.SOLVER.MAX_EPOCH = 200

_C.SOLVER.BASE_LR = 0.001
# The end lr, only used by WarmupCosineLR
_C.SOLVER.BASE_LR_END = 0.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)
# Number of decays in WarmupStepWithFixedGammaLR schedule
_C.SOLVER.NUM_DECAYS = 3

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"
# Whether to rescale the interval for the learning schedule after warmup
_C.SOLVER.RESCALE_INTERVAL = False

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = None  # None means following WEIGHT_DECAY

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# Enable automatic mixed precision for training
# Note that this does not change model's inference behavior.
# To use AMP in inference, run inference under autocast()
_C.SOLVER.AMP = CN({"ENABLED": False})

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAINING = CN()
_C.TRAINING.NUM_EPOCHS = 100
_C.TRAINING.EVAL_PERIOD = 5
_C.TRAINING.SYNC_BN = False


# ---------------------------------------------------------------------------- #
# Wandb logger options
# ---------------------------------------------------------------------------- #
_C.WANDB = CN()
_C.WANDB.PROJECT = ""


# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_C.SEED = -1
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False
# Option to set PyTorch matmul and CuDNN's float32 precision. When set to non-empty string,
# the corresponding precision ("highest", "high" or "medium") will be used. The highest
# precision will effectively disable tf32.
_C.FLOAT32_PRECISION = "medium"
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable.
_C.VIS_PERIOD = 0
