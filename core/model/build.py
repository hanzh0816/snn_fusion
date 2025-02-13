import torch

from core.model import MODEL_REGISTRY


def build_model(cfg, name, *args, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.NAME``.
    Note that it does not load any weights from ``cfg``.
    """

    model = MODEL_REGISTRY.get(name)(cfg, *args, **kwargs)
    return model
