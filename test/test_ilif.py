import math
from tkinter import Y
from typing import Literal
from torch import Tensor
import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer
from spikingjelly.clock_driven import surrogate
from spikingjelly.clock_driven.neuron import (
    MultiStepIFNode,
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)
import torch.nn.functional as F


class LIFNeuron(nn.Module):
    """
    wrapper for unified LIF nueron node interface
    """

    def __init__(
        self,
        spike_mode: Literal["lif", "plif", "if"] = "lif",
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        detach_reset: bool = False,
        backend: Literal["torch", "cupy"] = "torch",
        **kwargs,
    ):
        super().__init__()
        if spike_mode == "lif":
            self.lif_neuron = MultiStepLIFNode(
                tau=tau,
                v_threshold=v_threshold,
                detach_reset=detach_reset,
                v_reset=v_reset,
                backend=backend,
            )
        elif spike_mode == "plif":
            self.lif_neuron = MultiStepParametricLIFNode(
                init_tau=tau,
                v_threshold=v_threshold,
                detach_reset=detach_reset,
                v_reset=v_reset,
                backend=backend,
            )
        elif spike_mode == "if":
            self.lif_neuron = MultiStepIFNode(
                v_threshold=v_threshold,
                v_reset=v_reset,
                detach_reset=detach_reset,
                backend=backend,
            )
        elif spike_mode == "ilif":
            self.lif_neuron = MultiStepLIFNode(
                tau=tau,
                v_threshold=v_threshold,
                detach_reset=detach_reset,
                v_reset=v_reset,
                backend=backend,
                ilif=True,
            )
        else:
            raise NotImplementedError("Only support LIF/P-LIF spiking neuron")

    def forward(self, x: Tensor) -> Tensor:
        return self.lif_neuron(x)


ilif = LIFNeuron(
    spike_mode="ilif", tau=2.0, v_threshold=0.0, v_reset=None, detach_reset=True, backend="torch"
)
lif = LIFNeuron(
    spike_mode="lif", tau=2.0, v_threshold=1.0, v_reset=None, detach_reset=True, backend="torch"
)

x = torch.ones(4, 1)
x = x*2
# y = ilif(x)
y = lif(x)
print(y)
