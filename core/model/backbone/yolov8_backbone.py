import math

import torch
from torch import nn

from core.config.config import configurable
from core.model.build import MODEL_REGISTRY

from core.model.module.ann_module import *


def _make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


@MODEL_REGISTRY.register()
class YOLOv8Backbone(nn.Module):
    @configurable
    def __init__(self, modules_cfg, scales, out_indices=[-1], *args, **kwargs):
        super(YOLOv8Backbone, self).__init__()
        # image input channels=3
        self.in_channels = 3
        self.modules_cfg = modules_cfg
        self.depth, self.width, self.max_channels = scales

        self.out_indices = out_indices

        self.layers = nn.ModuleList()
        self._init_layers()

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        modules_cfg = cfg.CONFIG
        out_indices = cfg.OUT_INDICES
        ret = {
            "modules_cfg": modules_cfg,
            "out_indices": out_indices,
            **kwargs,
        }

        return ret

    def _init_layers(self):
        in_channels = self.in_channels
        for module in self.modules_cfg:
            module_name = module["name"]
            module_args = module["args"]

            repeat_nums = module.pop("repeat_nums", 1)
            repeat_nums = (
                max(round(repeat_nums * self.depth), 1) if repeat_nums > 1 else repeat_nums
            )
            module_args["repeat_nums"] = repeat_nums

            module_args["in_channels"] = in_channels

            if module_name in ["Conv", "SPPF"]:
                module_args["out_channels"] = _make_divisible(
                    min(module_args["out_channels"], self.max_channels) * self.width, 8
                )

            else:
                module_args["out_channels"] = in_channels

            in_channels = module_args["out_channels"]
            _m = (
                getattr(torch.nn, module_name[3:])
                if "nn." in module_name
                else globals()[module_name]
            )

            self.layers.append(_m(**module_args))

    def forward(self, inputs):
        outputs = []
        x = inputs
        actual_indices = [i if i >= 0 else len(self.layers) + i for i in self.out_indices]

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in actual_indices:
                outputs.append(x)

        return outputs  # 返回多个层的输出
