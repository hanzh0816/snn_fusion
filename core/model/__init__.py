from core.utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")

from .detector import *
from .backbone import *
from .neck import *
from .fusion import *
from .head import *
from .build import build_model
