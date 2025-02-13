import math

import torch
from torch import nn

from core.config.config import configurable
from core.model.build import MODEL_REGISTRY

from core.model.module.snn_module import *


def _make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


@MODEL_REGISTRY.register()
class SpikeYOLOBackbone(nn.Module):

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
        super(SpikeYOLOBackbone, self).__init__()
        # tag: 当前方法的输入通道数,正负极性各一个通道
        self.in_channels = in_channels
        self.modules_cfg = modules_cfg
        self.depth, self.width, self.max_channels = scales
        self.out_indices = out_indices
        self.spike_t = spike_t
        self.spike_cfg = spike_cfg

        self.layers = nn.ModuleList()
        self.preprocess = SpikePreprocess(self.spike_t)
        self._init_layers()

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        in_channels = cfg.IN_CHANNELS
        modules_cfg = cfg.CONFIG
        out_indices = cfg.OUT_INDICES
        ret = {
            "in_channels": in_channels,
            "modules_cfg": modules_cfg,
            "out_indices": out_indices,
            **kwargs,
        }

        return ret

    def _init_layers(self):
        in_channels = self.in_channels
        first_layer = True
        for module in self.modules_cfg:
            module_name = module["name"]
            module_args = module["args"]
            repeat_nums = module.pop("repeat_nums", 1)
            repeat_nums = (
                max(round(repeat_nums * self.depth), 1) if repeat_nums > 1 else repeat_nums
            )
            module_args["in_channels"] = in_channels

            # tag: 所有的模块统一用同一个resize channel方法
            if module_name in ["DownSampling", "SpikeSPPF", "SpikeConv"]:
                module_args["out_channels"] = _make_divisible(
                    min(module_args["out_channels"], self.max_channels) * self.width, 8
                )
            else:
                module_args["out_channels"] = in_channels

            in_channels = module_args["out_channels"]
            module_args["spike_cfg"] = self.spike_cfg

            module_args["first_layer"] = first_layer
            _m = (
                getattr(torch.nn, module_name[3:])
                if "nn." in module_name
                else globals()[module_name]
            )

            self.layers.append(
                nn.Sequential(*(_m(**module_args) for _ in range(repeat_nums)))
                if repeat_nums > 1
                else _m(**module_args)
            )
            if first_layer:
                first_layer = False

    def forward(self, inputs):
        outputs = []
        x = inputs
        x = self.preprocess(x)
        # 计算实际的索引，将负数索引转换为正数
        actual_indices = [i if i >= 0 else len(self.layers) + i for i in self.out_indices]

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in actual_indices:
                outputs.append(x)

        return outputs  # 返回多个层的输出
