import math
from typing import Literal
from torch import Tensor
import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer
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


class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


def autopad(kernel_size, padding=None, dilation=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if dilation > 1:
        kernel_size = (
            dilation * (kernel_size - 1) + 1
            if isinstance(kernel_size, int)
            else [dilation * (x - 1) + 1 for x in kernel_size]
        )  # actual kernel-size
    if padding is None:
        padding = (
            kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
        )  # auto-pad
    return padding


class SepRepConv(nn.Module):
    """
    放在Sepconv最后一个1*1卷积，采用3*3分组+1*1降维的方式实现，能提0.5个点。之后可以试试改成1*1降维和3*3分组
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False, group=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        # hidden_channel = in_channels
        #         conv1x1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channels)
        conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size, 1, 0, groups=in_channels, bias=False
            ),  # 这里也是分组卷积
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=group, bias=False),
        )

        self.body = nn.Sequential(bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        in_channels,
        expansion_ratio=2,
        bias=False,
        kernel_size=3,  # 7,3
        padding=1,
        spike_cfg=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        med_channels = int(expansion_ratio * in_channels)
        self.pwconv1 = layer.SeqToANNContainer(
            nn.Conv2d(in_channels, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels),
        )
        self.dwconv2 = layer.SeqToANNContainer(
            nn.Conv2d(
                med_channels,
                med_channels,
                kernel_size=kernel_size,  # 7*7
                padding=padding,
                groups=med_channels,
                bias=bias,
            ),
            nn.BatchNorm2d(med_channels),
        )
        self.pwconv3 = layer.SeqToANNContainer(
            SepRepConv(med_channels, in_channels), nn.BatchNorm2d(in_channels)
        )

        self.lif1 = LIFNeuron(**spike_cfg)
        self.lif2 = LIFNeuron(**spike_cfg)
        self.lif3 = LIFNeuron(**spike_cfg)

    def forward(self, x):
        x = self.lif1(x)
        x = self.pwconv1(x)
        x = self.lif2(x)
        x = self.dwconv2(x)
        x = self.lif3(x)
        x = self.pwconv3(x)
        return x


class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False, group=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        # hidden_channel = in_channels
        conv1x1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channels)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, 1, 0, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SpikeConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        with_bn=True,
        spike_cfg=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.lif = LIFNeuron(**spike_cfg)
        if with_bn:
            self.conv = layer.SeqToANNContainer(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    autopad(kernel_size, padding, dilation),
                    groups=groups,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = layer.SeqToANNContainer(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    autopad(kernel_size, padding, dilation),
                    groups=groups,
                    dilation=dilation,
                    bias=True,
                ),
            )

    def forward(self, x):
        x = self.lif(x)
        x = self.conv(x)
        return x


class DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=256,
        kernel_size=3,
        stride=2,
        padding=1,
        spike_cfg=None,
        first_layer=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        # todo: first layer
        if not first_layer:
            self.encode_lif = LIFNeuron(**spike_cfg)
        self.encode_conv = layer.SeqToANNContainer(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(out_channels),
            )
        )

        # self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # 不是第一层，需要经过LIF神经元
        if hasattr(self, "encode_lif"):
            x = self.encode_lif(x)
        x = self.encode_conv(x)
        return x


class UpSampling(nn.Module):
    def __init__(self, size, scale_factor, mode, *args, **kwargs):
        super().__init__()
        self.up_sample = nn.Upsample(size=size, scale_factor=tuple(scale_factor), mode=mode)

    def forward(self, x):
        return self.up_sample(x)


class Concat(nn.Module):
    def __init__(self, dimension=1, *args, **kwargs):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class ConvBlock1(nn.Module):
    """
    第二阶段使用普通Conv卷积的block
    """

    def __init__(
        self,
        in_channels,
        mlp_ratio=4.0,
        sep_kernel_size=7,
        group=False,
        spike_cfg=None,
        *args,
        **kwargs,
    ):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.seq_conv = SepConv(
            in_channels=in_channels, kernel_size=sep_kernel_size, spike_cfg=spike_cfg
        )  # 内部扩张2倍
        self.mlp_ratio = mlp_ratio
        if group == True:
            # 使用分组卷积
            self.conv1 = SpikeConv(
                in_channels=in_channels,
                out_channels=int(in_channels * mlp_ratio),
                kernel_size=3,
                groups=4,
                spike_cfg=spike_cfg,
            )

        else:
            self.conv1 = SpikeConv(
                in_channels=in_channels,
                out_channels=int(in_channels * mlp_ratio),
                kernel_size=3,
                spike_cfg=spike_cfg,
            )
        self.conv2 = SpikeConv(
            in_channels=int(in_channels * mlp_ratio),
            out_channels=in_channels,
            kernel_size=3,
            spike_cfg=spike_cfg,
        )

    def forward(self, x):
        x = self.seq_conv(x) + x  # sepconv  pw+dw+pw

        x_feat = x

        x = self.conv1(x)  # conv内部实现lif
        x = self.conv2(x)
        x = x_feat + x

        return x


class ConvBlock2(nn.Module):
    """ """

    def __init__(
        self,
        in_channels,
        mlp_ratio=4.0,
        sep_kernel_size=7,
        spike_cfg=None,
        *args,
        **kwargs,
    ):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        # todo: full conv
        self.seq_conv = SepConv(
            in_channels=in_channels, kernel_size=sep_kernel_size, spike_cfg=spike_cfg
        )  # 内部扩张2倍
        self.mlp_ratio = mlp_ratio

        self.lif1 = LIFNeuron(**spike_cfg)
        self.lif2 = LIFNeuron(**spike_cfg)
        # todo: 去掉重参数
        self.conv1 = layer.SeqToANNContainer(
            RepConv(in_channels, int(in_channels * mlp_ratio)),
            nn.BatchNorm2d(int(in_channels * mlp_ratio)),
        )

        self.conv2 = layer.SeqToANNContainer(
            RepConv(int(in_channels * mlp_ratio), in_channels), nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        x = self.seq_conv(x) + x  # sepconv  pw+dw+pw

        x_feat = x

        x = self.lif1(x)
        x = self.conv1(x)
        x = self.lif2(x)
        x = self.conv2(x)
        x = x_feat + x

        return x


class SpikeSPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(
        self, in_channels, out_channels, kernel_size=5, spike_cfg=None, *args, **kwargs
    ):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        med_channels = in_channels // 2  # hidden channels
        self.conv1 = SpikeConv(
            in_channels=in_channels,
            out_channels=med_channels,
            kernel_size=1,
            stride=1,
            spike_cfg=spike_cfg,
        )
        self.conv2 = SpikeConv(
            in_channels=med_channels * 4,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            spike_cfg=spike_cfg,
        )
        self.max_pool = layer.SeqToANNContainer(
            nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.max_pool(x)
        y2 = self.max_pool(x)
        y3 = self.max_pool(x)
        return self.conv2(torch.cat((x, y1, y2, y3), 2))


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    # todo: 待修改

    def __init__(self, in_channels=16, *args, **kwargs):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(in_channels, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, in_channels, 1, 1))
        self.c1 = in_channels

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class SpikePreprocess(nn.Module):
    def __init__(self, spike_t):
        super().__init__()
        self.spike_t = spike_t

    def forward(self, x):
        T, B, C, H, W = x.shape  # T=1000
        assert T % self.spike_t == 0, "T must be divisible by 4"

        data_aggregated = x.reshape(self.spike_t, T // self.spike_t, B, C, H, W).sum(axis=1)
        return data_aggregated
