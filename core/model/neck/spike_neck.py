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
class SpikeYOLONeck(nn.Module):
    @configurable
    def __init__(
        self,
        in_channels,
        from_indices,
        out_indices,
        modules_cfg,
        scales,
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
        super(SpikeYOLONeck, self).__init__()
        self.in_channels = in_channels
        self.from_indices = from_indices
        self.out_indices = out_indices
        self.depth, self.width, self.max_channels = scales
        self.spike_cfg = spike_cfg
        self.modules_cfg = modules_cfg

        self.layers = nn.ModuleList()
        self.out_channels = []
        self._init_layers()

    def _init_layers(self):
        # SPPF output channels
        in_channels = _make_divisible(min(self.in_channels, self.max_channels) * self.width, 8)
        for i, module in enumerate(self.modules_cfg):
            module_name = module["name"]
            module_args = module["args"]
            repeat_nums = module.pop("repeat_nums", 1)
            repeat_nums = (
                max(round(repeat_nums * self.depth), 1) if repeat_nums > 1 else repeat_nums
            )

            module_args["in_channels"] = in_channels
            if module_name in ["SpikeSPPF", "SpikeConv"]:
                module_args["out_channels"] = _make_divisible(
                    min(module_args["out_channels"], self.max_channels) * self.width, 8
                )
            elif module_name == "Concat":
                # Concat output channels = last layer output channels + from_channels(resized)
                module_args["out_channels"] = in_channels + _make_divisible(
                    min(module_args["from_channels"], self.max_channels) * self.width, 8
                )
            else:
                module_args["out_channels"] = in_channels

            in_channels = module_args["out_channels"]

            if i in self.out_indices:
                self.out_channels.append(module_args["out_channels"])
            module_args["spike_cfg"] = self.spike_cfg
            module = (
                getattr(torch.nn, module_name[3:])
                if "nn." in module_name
                else globals()[module_name]
            )

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

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        in_channels = cfg.IN_CHANNELS
        modules_cfg = cfg.CONFIG
        from_indices = cfg.FROM_INDICES
        out_indices = cfg.OUT_INDICES
        ret = {
            "in_channels": in_channels,
            "modules_cfg": modules_cfg,
            "from_indices": from_indices,
            "out_indices": out_indices,
            **kwargs,
        }
        return ret

    def forward(self, inputs: list):
        outs = []
        for i, layer in enumerate(self.layers):
            from_idx = self.from_indices[i]
            if type(from_idx) is list:
                x = layer([inputs[k] for k in from_idx])
            else:
                x = layer(inputs[from_idx])
            if i in self.out_indices:
                outs.append(x)

            inputs.append(x)

        return outs
