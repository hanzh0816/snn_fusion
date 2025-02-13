from core.utils.registry import Registry

TRANSFORMS_REGISTRY = Registry("TRANSFORMS")

from .build import build_detection_val_loader, build_detection_train_loader
from .transforms import *
