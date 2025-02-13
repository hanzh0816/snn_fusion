import math

import torch
from torch import nn

from core.config.config import configurable
from core.model.build import MODEL_REGISTRY


def _make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


@MODEL_REGISTRY.register()
class FusionBackbone(nn.Module):

    @configurable
    def __init__(
        self,
        in_channels,
        modules_cfg,
        scales,
        out_indices=[-1],
        spike_t=4,
        spike_cfg=dict(
            detach_reset=True,
            tau=2.0,
            v_reset=0.0,
            v_threshold=1.0,
            backend="torch",
        ),
        *args,
        **kwargs
    ):
        super(FusionBackbone, self).__init__()
        # tag: 当前方法的输入通道数,正负极性各一个通道
        self.in_channels = in_channels
