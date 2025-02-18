# 用的原版sj

# -*- coding: utf-8 -*-
# from visualizer import get_local
import torch
import torchinfo
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import layer
from timm.layers import to_2tuple, trunc_normal_, DropPath
from timm.models import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
import warnings

# from visualizer import get_local
import numpy as np

from core.model.loss.loss_utils import dist2bbox, make_anchors


# time_window = 1lif
thresh = 1  # 0.5 # neuronal threshold
decay = 0.25  # 0.25 # decay constants
lens = 0.5  # 0.5 # hyper-parameters of approximate function

import math


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


@torch.jit.script
def jit_mul(x, y):
    return x.mul(y)


@torch.jit.script
def jit_sum(x):
    return x.sum(dim=[-1, -2], keepdim=True)


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class SpikeDFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)  # [0,1,2,...,15]
        self.conv.weight.data[:] = nn.Parameter(
            x.view(1, c1, 1, 1)
        )  # 这里不是脉冲驱动的，但是是整数乘法
        self.c1 = c1  # 本质上就是个加权和。输入是每个格子的概率(小数)，权重是每个格子的位置(整数)
        self.lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        # x = self.lif(x.view(1,b, 4, self.c1, a).transpose(3, 2))  #[T,B,C,A]
        # x = x.mean(0) #[B,C,A]
        # print("=================================")
        # x = self.conv(x).view(b, 4, a) # #
        # return x
        # self.conv(x.flatten(0, 1))).reshape(T, b, -1, H_new, W_new)
        # return self.conv(x).view(b, 4, a)
        #         print("weight:",self.conv.weight.data)
        # return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1)).view(b, 4, a)  #原版
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)  # 原版


class SpikeDetect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                SpikeConv(x, c2, 3),
                SpikeConv(c2, c2, 3),
                SpikeConvWithoutBN(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                SpikeConv(x, c3, 3), SpikeConv(c3, c3, 3), SpikeConvWithoutBN(c3, self.nc, 1)
            )
            for x in ch
        )
        self.dfl = SpikeDFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = (
            x[0].mean(0).shape
        )  # BCHW  推理：[1，2，64，32，84]  这里必须mean0，否则推理时用到shape会导致报错
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 2)
            x[i] = x[i].mean(0)  # [2，144，32，684]  #这个地方有时候全是1.之后debug看看
        return x


    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].conv.bias.data[:] = 1.0  # box
            b[-1].conv.bias.data[: m.nc] = math.log(
                5 / m.nc / (640 / s) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = MS_StandardConv(c1, c_, k[0], 1)
        self.cv2 = MS_StandardConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class MS_C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        print("MS_C2f_N:", n)
        print("MS_C2f_c:", self.c)
        self.cv1 = MS_StandardConv(c1, 2 * self.c, 1, 1)
        self.cv2 = MS_StandardConv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 2))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 2))

    # def forward_split(self, x):
    #     """Forward pass using split() instead of chunk()."""
    #     y = list(self.cv1(x).split((self.c, self.c), 1))
    #     y.extend(m(y[-1]) for m in self.m)
    #     return self.cv2(torch.cat(y, 1))


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
        super(BNAndPadLayer, self).__init__()
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


class RepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, bias=False, group=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            #              MultiStepLIFNode(detach_reset=True,v_reset=None, backend='torch'),
            nn.Conv2d(
                in_channel, in_channel, kernel_size, 1, 0, groups=in_channel, bias=False
            ),  # 这里也是分组卷积
            #              MultiStepLIFNode(detach_reset=True,v_reset=None, backend='torch'),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class CLRepConv(nn.Module):
    def __init__(self, in_channel, matio, kernel_size=3, bias=False, group=1):
        super().__init__()
        hidden_channel = int(in_channel * matio)
        padding = int((kernel_size - 1) / 2)
        conv1x1 = nn.Conv2d(in_channel, hidden_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=hidden_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(
                hidden_channel, hidden_channel, kernel_size, 1, 0, groups=hidden_channel, bias=False
            ),  # 这里也是分组卷积
            nn.Conv2d(hidden_channel, in_channel, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(in_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepRepConv(
    nn.Module
):  # 放在Sepconv最后一个1*1卷积，采用3*3分组+1*1降维的方式实现，能提0.5个点。之后可以试试改成1*1降维和3*3分组
    def __init__(self, in_channel, out_channel, kernel_size=3, bias=False, group=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        # hidden_channel = in_channel
        #         conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channel, in_channel, kernel_size, 1, 0, groups=in_channel, bias=False
            ),  # 这里也是分组卷积
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=group, bias=False),
        )

        self.body = nn.Sequential(bn, conv3x3)

    def forward(self, x):
        return self.body(x)


# 只实现了3*3的卷积，没有实现升维或降维
class HalfRepConv(nn.Module):
    def __init__(self, in_channel, kernel_size=3, bias=False, group=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channel, in_channel, kernel_size, 1, 0, groups=in_channel, bias=False
            ),  # 这里也是分组卷积
            nn.BatchNorm2d(in_channel),
        )
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        x = self.body(x)
        return x


# 只实现了3*3的卷积，没有实现升维,但3*3卷积和前面的1*1卷积都不是分组的
class HalfFullRepConv(nn.Module):
    def __init__(self, in_channel, kernel_size=3, bias=False, group=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channel, in_channel, kernel_size, 1, 0, groups=1, bias=False
            ),  # 这里也是分组卷积
            nn.BatchNorm2d(in_channel),
        )
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        x = self.body(x)
        return x


# 只实现了3*3的不分组卷积，同时也实现了升维
class FullRepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, bias=False, group=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channel, in_channel, kernel_size, 1, 0, groups=group, bias=False
            ),  # 这里也是分组卷积
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class BinaryRepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, bias=False, group=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channel, in_channel, kernel_size, 1, 0, groups=in_channel, bias=False
            ),  # 这里也是分组卷积
            nn.BatchNorm2d(in_channel),
        )
        self.body = nn.Sequential(conv1x1, bn, conv3x3)
        self.lastconv = (nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=group, bias=False),)

    def forward(self, x):
        return self.body(x)


# 0.499版的sepconv 注释
# class SepConv(nn.Module):
#     r"""
#     Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
#     """

#     def __init__(
#         self,
#         dim,
#         expansion_ratio=2,
#         act2_layer=nn.Identity,
#         bias=False,
#         kernel_size=3,  # 7,3
#         padding=1,
#     ):
#         super().__init__()
#         padding = int((kernel_size - 1) / 2)
#         med_channels = int(expansion_ratio * dim)
#         self.lif1 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
#         self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
#         self.bn1 = nn.BatchNorm2d(med_channels)
#         self.lif2 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
#         self.dwconv = nn.Conv2d(
#             med_channels,
#             med_channels,
#             kernel_size=kernel_size,  # 7*7
#             padding=padding,
#             groups=med_channels,
#             bias=bias,
#         )  # depthwise conv

#         self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias, groups=4)
#         self.bn2 = nn.BatchNorm2d(dim)

#     def forward(self, x):
#         T, B, C, H, W = x.shape
#         x = self.lif1(
#             x
#         )  # x1_lif:0.2328  x2_lif:0.0493  这里x2的均值偏小，因此其经过bn和lif后也偏小，发放率比较低；而x1均值偏大，因此发放率也高
#         x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(
#             T, B, -1, H, W
#         )  # flatten：从第0维开始，展开到第一维
#         x = self.lif2(x)
#         x = self.dwconv(x.flatten(0, 1))

#         x = self.bn2(self.pwconv2(x)).reshape(T, B, -1, H, W)
#         return x


# 0.490版的sepconv(补档)
class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=3,  # 7,3
        padding=1,
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.dwconv2 = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,  # 7*7
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        #         self.pwconv3 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias,groups=1)
        self.pwconv3 = SepRepConv(
            med_channels, dim
        )  # 这里将sepconv最后一个卷积替换为重参数化卷积  大概提0.5个点，可以保留

        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)

        self.lif1 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.lif2 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.lif3 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

    def forward(self, x):
        T, B, C, H, W = x.shape
        #         print("x.shape:",x.shape)
        x = self.lif1(
            x
        )  # x1_lif:0.2328  x2_lif:0.0493  这里x2的均值偏小，因此其经过bn和lif后也偏小，发放率比较低；而x1均值偏大，因此发放率也高
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(
            T, B, -1, H, W
        )  # flatten：从第0维开始，展开到第一维
        x = self.lif2(x)
        x = self.bn2(self.dwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn3(self.pwconv3(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x


# 这个版本将深度可分离卷积直接改为了3*3卷积，并且加入lif
class MS_FullConvBlock(nn.Module):
    def __init__(
        self, input_dim, mlp_ratio=4.0, sep_kernel_size=7, SE_size=5
    ):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        SE_padding = int((SE_size - 1) / 2)
        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)  # 内部扩张2倍
        self.mlp_ratio = mlp_ratio

        self.standardconv = MS_StandardConv(input_dim, input_dim * mlp_ratio)
        self.lif1 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.halfconv = HalfFullRepConv(input_dim, group=1)
        # self.halflif =  MultiStepLIFNode(detach_reset=True,v_reset=None, backend='torch')
        #         self.bn1 = nn.BatchNorm2d(int(input_dim * mlp_ratio))  # 这里可以进行改进

        # self.conv2 = nn.Conv2d(input_dim*mlp_ratio, input_dim, kernel_size=3, padding=1, groups=1, bias=False)
        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进
        self.lif2 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

        # self.conv0 = nn.Conv2d(
        #     input_dim, input_dim, kernel_size=SE_size, #7*7
        #     padding=SE_padding, groups=int(input_dim), bias=False)  # depthwise conv
        # self.Identity = nn.Identity()

        # self.bn0 = nn.BatchNorm2d(input_dim)
        # self.lif0 =  MultiStepLIFNode(detach_reset=True,v_reset=None, backend='torch')
        # self.res2netconv = Res2NetConv(input_dim,input_dim )  # 输入不是脉冲格式，输出也不是脉冲格式，但是脉冲驱动运算的
        # self.res2netconv = Res2NetConv3(input_dim,int(input_dim * mlp_ratio))  #输入不是脉冲格式，输出也不是脉冲格式，但是脉冲驱动运算的

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  # sepconv  pw+dw+pw MS_FullConvBlock

        x_feat = x
        # x = self.res2netconv(x)
        x = self.halfconv(self.lif1(x).flatten(0, 1)).reshape(
            T, B, C, H, W
        )  # 完成repconv的前半部分
        x = self.standardconv(x)  # 升维

        # x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio * C), H, W)
        # repconv，对应conv_mixer，包含1*1,3*3,1*1三个卷积，等价于一个3*3卷积
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


# 这个版本实现的是直接在repconv内部实现升维降维操作的网络 注释
class MS_ConvBlock(nn.Module):
    def __init__(
        self, input_dim, mlp_ratio=4.0, sep_kernel_size=7, SE_size=5
    ):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)  # 内部扩张2倍
        self.mlp_ratio = mlp_ratio

        # self.conv1 = MS_BnnConv(input_dim, int(input_dim * mlp_ratio))

        self.conv1 = CLRepConv(input_dim, mlp_ratio)
        self.conv2 = CLRepConv(input_dim, mlp_ratio / 2)

        # self.conv2 = nn.Conv2d(input_dim*mlp_ratio, input_dim, kernel_size=3, padding=1, groups=1, bias=False)
        # self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn1 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进
        self.lif1 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.bn2 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进
        self.lif2 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  # sepconv  pw+dw+pw

        x_feat = x

        # x = self.conv1(x)
        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, -1, H, W)
        # x = self.conv1(x
        # repconv，对应conv_mixer，包含1*1,3*3,1*1三个卷积，等价于一个3*3卷积
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


# 升维，深度卷积(1*1,3*3,5*5) ,降维
class BottleneckConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, bias=False, mlp_ratio=1):
        super().__init__()

        self.hidden_channel = int(in_channel * mlp_ratio)
        self.conv1 = MS_StandardConv(in_channel, self.hidden_channel)  # 用于内部升维
        self.dwconv1 = MS_StandardConv(
            self.hidden_channel, self.hidden_channel, 5, g=self.hidden_channel
        )  #
        self.dwconv2 = MS_StandardConv(
            self.hidden_channel, self.hidden_channel, 5, g=self.hidden_channel
        )  #
        self.conv2 = MS_StandardConv(self.hidden_channel, out_channel)  # 用于内部升维

    #         self.conv1 = MS_StandardConv(in_channel, out_channel)  #用于内部升维

    def forward(self, x):
        x = self.conv1(x)

        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x = x + x1 + x2
        x = self.conv2(x)
        return x


class MS_ConvBlock_res2net(nn.Module):
    def __init__(
        self, input_dim, output_dim, mlp_ratio=4.0, inter_ratio=4, final_kernel=1
    ):  # in_channels(out_channels), mlp_ratio是外部扩张比例，inter_ratio是内部扩张比例
        super().__init__()

        self.hidden_dim = int(input_dim * mlp_ratio / 4)
        self.conv0 = MS_StandardConv(
            input_dim, int(input_dim * mlp_ratio)
        )  # 实现一阶段升维(1*1卷积)
        self.conv1 = BottleneckConv(self.hidden_dim, self.hidden_dim, mlp_ratio=inter_ratio)
        self.conv2 = BottleneckConv(self.hidden_dim, self.hidden_dim, mlp_ratio=inter_ratio)
        self.conv3 = BottleneckConv(self.hidden_dim, self.hidden_dim, mlp_ratio=inter_ratio)
        self.conv4 = MS_StandardConv(int(input_dim * mlp_ratio), output_dim, final_kernel)

    def forward(self, x):
        x_shortcut = x
        x = self.conv0(x)
        x0, x1, x2, x3 = x.chunk(
            4, dim=2
        )  # 是否改变通道数时，在前3维度加个max_pooling？（比如通道维度的max_pooling）
        x0_out = x0
        x1_out = self.conv1(x0_out + x1)
        x2_out = self.conv2(x1_out + x2)
        x3_out = self.conv3(x2_out + x3)
        x = torch.cat([x0_out, x1_out, x2_out, x3_out], dim=2)
        x = self.conv4(x) + x_shortcut

        return x


class MS_ConvBlock(nn.Module):
    def __init__(
        self, input_dim, mlp_ratio=4.0, sep_kernel_size=7, full=False
    ):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.full = full
        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)  # 内部扩张2倍
        self.mlp_ratio = mlp_ratio

        self.lif1 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.lif2 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

        #         self.standardconv = MS_StandardConv(input_dim, input_dim * mlp_ratio)
        #         self.halfconv = HalfRepConv(input_dim,group=1)
        if self.full == True:  # 如果repconv不分组，直接用不分组的fullrepconv来得到self.conv1
            self.conv1 = FullRepConv(input_dim, int(input_dim * mlp_ratio))
        else:  # 否则用分组的repconv得到。
            #             self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio),group=4) #针对137模型的改进，将所有block的第一个repconv改为4个分组卷积
            self.conv1 = RepConv(
                input_dim, int(input_dim * mlp_ratio)
            )  # 137以外的模型，在第一个block不做分组

        self.bn1 = nn.BatchNorm2d(int(input_dim * mlp_ratio))  # 这里可以进行改进

        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进

        # self.conv0 = nn.Conv2d(
        #     input_dim, input_dim, kernel_size=SE_size, #7*7
        #     padding=SE_padding, groups=int(input_dim), bias=False)  # depthwise conv
        # self.Identity = nn.Identity()

        # self.bn0 = nn.BatchNorm2d(input_dim)
        # self.lif0 =  MultiStepLIFNode(detach_reset=True,v_reset=None, backend='torch')
        # self.res2netconv = Res2NetConv(input_dim,input_dim )  # 输入不是脉冲格式，输出也不是脉冲格式，但是脉冲驱动运算的
        # self.res2netconv = Res2NetConv3(input_dim,int(input_dim * mlp_ratio))  #输入不是脉冲格式，输出也不是脉冲格式，但是脉冲驱动运算的

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  # sepconv  pw+dw+pw

        x_feat = x

        #         x = self.halfconv(self.lif1(x).flatten(0, 1)).reshape(T, B, C, H, W) #完成repconv的前半部分
        #         x = self.standardconv(x) #升维
        #         print("==============self.lif1(x)",self.lif1(x))

        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(
            T, B, int(self.mlp_ratio * C), H, W
        )
        # repconv，对应conv_mixer，包含1*1,3*3,1*1三个卷积，等价于一个3*3卷积
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class MS_AllConvBlock(nn.Module):  # 在这个模式中，将repconv两部分全部替换为普通卷积
    def __init__(
        self, input_dim, mlp_ratio=4.0, sep_kernel_size=7, group=False
    ):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)  # 内部扩张2倍
        self.mlp_ratio = mlp_ratio
        if group == True:
            self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio), 3, g=4)  # 136版本
        else:
            self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio), 3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim, 3)

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  # sepconv  pw+dw+pw

        x_feat = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = x_feat + x

        return x


# BNN部分（包括MS_ConvBlock也是bnn版本的）

# class MS_ConvBlock(nn.Module):
#     def __init__(self, input_dim, mlp_ratio=4.,sep_kernel_size = 7 ,SE_size=5):  # in_channels(out_channels), 内部扩张比例
#         super().__init__()


#         self.Conv = SepConv(dim=input_dim,kernel_size= sep_kernel_size)  #内部扩张2倍
#         self.mlp_ratio = mlp_ratio

#         self.conv1 = MS_BnnConv(input_dim, int(input_dim * mlp_ratio))

#         # self.conv2 = nn.Conv2d(input_dim*mlp_ratio, input_dim, kernel_size=3, padding=1, groups=1, bias=False)
#         self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
#         self.bn2 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进
#         self.lif2 =  MultiStepLIFNode(detach_reset=True,v_reset=None, backend='torch')


#     def forward(self, x):
#         T, B, C, H, W = x.shape
#         x = self.Conv(x) + x  #sepconv  pw+dw+pw

#         x_feat = x

#         x = self.conv1(x)
#         # x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio * C), H, W)
#         # x = self.conv1(x
#             #repconv，对应conv_mixer，包含1*1,3*3,1*1三个卷积，等价于一个3*3卷积
#         x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
#         x = x_feat + x

#         return x


class MS_BnnConv(nn.Module):
    def __init__(
        self, input_size, output_size, k=3, s=1, p=None, g=1, d=1
    ):  # in_channels(out_channels), 内部扩张比例
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = min(192, self.input_size)
        self.output_size = int(output_size)
        self.bnn_size = self.output_size - self.hidden_size  # 通过BNN生成的channel维度

        self.conv1 = nn.Conv2d(
            self.input_size,
            self.hidden_size,
            k,
            s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(self.hidden_size)  # 生成初步网络特征
        self.conv2 = BinarizeConv2d(
            self.hidden_size,
            self.bnn_size,
            k,
            s,
            autopad(k, p, d),
            groups=4,
            dilation=d,
            bias=False,
        )

        self.lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.s = s
        bnn_parameters = self.bnn_size * self.hidden_size * 9 / 4
        print("bnn_parameters:", bnn_parameters)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.bn(self.conv1(self.lif(x).flatten(0, 1)))
        x1 = x.reshape(T, B, -1, int(H / self.s), int(W / self.s))  # 直接卷积生成的
        x2 = self.conv2(x).reshape(T, B, -1, int(H / self.s), int(W / self.s))  # bnn生成的特征
        x = torch.cat((x1, x2), 2)

        return x


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        input_b = input
        weight_b = binarized(self.weight)

        out = nn.functional.conv2d(
            input_b, weight_b, None, self.stride, self.padding, self.dilation, self.groups
        )

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


from torch.autograd.function import Function, InplaceFunction


class Binarize(InplaceFunction):

    def forward(ctx, input, quant_mode="det", allow_scale=False, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        scale = output.abs().max() if allow_scale else 1

        if quant_mode == "det":
            return output.div(scale).sign().mul(scale)
        else:

            return (
                output.div(scale)
                .add_(1)
                .div_(2)
                .add_(torch.rand(output.size()).cuda().add(-0.5))
                .clamp_(0, 1)
                .round()
                .mul_(2)
                .add_(-1)
                .mul(scale)
            )

    def backward(ctx, grad_output):
        # STE
        grad_input = grad_output
        return grad_input, None, None, None


class Quantize(InplaceFunction):
    def forward(ctx, input, quant_mode="det", numBits=4, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        scale = (2**numBits - 1) / (output.max() - output.min())
        output = output.mul(scale).clamp(-(2 ** (numBits - 1)) + 1, 2 ** (numBits - 1))
        if quant_mode == "det":
            output = output.round().div(scale)
        else:
            output = output.round().add(torch.rand(output.size()).add(-0.5)).div(scale)
        return output

    def backward(grad_output):
        # STE
        grad_input = grad_output
        return grad_input, None, None


def binarized(input, quant_mode="det"):
    return Binarize.apply(input, quant_mode)


def quantize(input, quant_mode, numBits):
    return Quantize.apply(input, quant_mode, numBits)


# bnn内容结束


class MS_StandardConv(nn.Module):
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, d=1
    ):  # in_channels(out_channels), 内部扩张比例
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

    def forward(self, x):
        T, B, C, H, W = x.shape  # 4,1,128,32,40
        x = self.bn(self.conv(self.lif(x).flatten(0, 1))).reshape(
            T, B, self.c2, int(H / self.s), int(W / self.s)
        )
        return x


class MS_DownSampling(nn.Module):
    def __init__(
        self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        # self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_lif"):  # 如果不是第一层
            # x_pool = self.pool(x)
            x = self.encode_lif(x)

        x = self.encode_conv(x.flatten(0, 1))
        _, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        return x


class MS_GetT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=4):
        super().__init__()
        self.T = T
        self.in_channels = in_channels

    def forward(self, x):
        if len(x.shape) == 4:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        # elif len(x.shape) == 5:
        #     x = x.transpose(0, 1)  # 1,1,3,256,320 #CHW
        img = x[0][0]
        from PIL import Image

        img = img.permute(2, 1, 0)  # WhC
        img = img.detach().cpu().numpy()
        img = Image.fromarray((img * 255).astype(np.uint8))  # RGB-order PIL image
        img.save("input.jpg")  # save to disk

        return x


class MS_CancelT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=2):
        super().__init__()
        self.T = T

    def forward(self, x):
        x = x.mean(0)
        return x


class SpikeConv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.bn = nn.BatchNorm2d(c2)
        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        return x


class SpikeConvWithoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)
        self.lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.conv(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        return x


class SpikeSPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = SpikeConv(c1, c_, 1, 1)
        self.cv2 = SpikeConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            T, B, C, H, W = x.shape
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x.flatten(0, 1)).reshape(T, B, -1, H, W)
            y2 = self.m(y1.flatten(0, 1)).reshape(T, B, -1, H, W)
            y3 = self.m(y2.flatten(0, 1)).reshape(T, B, -1, H, W)
            return self.cv2(torch.cat((x, y1, y2, y3), 2))


class Res2NetConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
    ):
        super().__init__()
        self.conv0 = RepConv(int(in_channel / 4), int(in_channel / 4), group=1)
        self.conv1 = RepConv(int(in_channel / 4), int(in_channel / 4), group=1)
        self.conv2 = RepConv(int(in_channel / 4), int(in_channel / 4), group=1)
        self.conv3 = RepConv(int(in_channel / 4), int(in_channel / 4), group=1)

        self.bn0 = nn.BatchNorm2d(int(in_channel / 4))
        self.bn1 = nn.BatchNorm2d(int(in_channel / 4))
        self.bn2 = nn.BatchNorm2d(int(in_channel / 4))
        self.bn3 = nn.BatchNorm2d(int(in_channel / 4))

        self.lif0 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.lif1 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.lif2 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.lif3 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

    def forward(self, x):
        x0, x1, x2, x3 = x.chunk(
            4, dim=2
        )  # 是否改变通道数时，在前3维度加个max_pooling？（比如通道维度的max_pooling）
        T, B, C, H, W = x0.shape
        x0 = self.bn0(self.conv0(self.lif0(x0).flatten(0, 1))).reshape(T, B, C, H, W)
        x1 = self.bn1(self.conv1(self.lif1(x0 + x1).flatten(0, 1))).reshape(T, B, C, H, W)
        x2 = self.bn2(self.conv2(self.lif2(x1 + x2).flatten(0, 1))).reshape(T, B, C, H, W)
        x3 = self.bn3(self.conv3(self.lif3(x2 + x3).flatten(0, 1))).reshape(T, B, C, H, W)
        x = torch.cat([x0, x1, x2, x3], dim=2)
        return x


class Res2NetConv2(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
    ):
        super().__init__()

        self.ratio = int(out_channel / in_channel)

        self.conv0 = RepConv(int(in_channel / 4), int(out_channel / 4), group=1)
        self.conv1 = RepConv(int(in_channel / 4), int(out_channel / 4), group=1)
        self.conv2 = RepConv(int(in_channel / 4), int(out_channel / 4), group=1)
        self.conv3 = RepConv(int(in_channel / 4), int(out_channel / 4), group=1)

        self.bn0 = nn.BatchNorm2d(int(out_channel / 4))
        self.bn1 = nn.BatchNorm2d(int(out_channel / 4))
        self.bn2 = nn.BatchNorm2d(int(out_channel / 4))
        self.bn3 = nn.BatchNorm2d(int(out_channel / 4))

        self.lif0 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.lif1 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.lif2 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.lif3 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

        self.pooling = nn.MaxPool1d(kernel_size=self.ratio)

    def forward(self, x):
        x0, x1, x2, x3 = x.chunk(
            4, dim=2
        )  # 是否改变通道数时，在前3维度加个max_pooling？（比如通道维度的max_pooling）
        T, B, C, H, W = x0.shape
        x0_out = self.bn0(self.conv0(self.lif0(x0).flatten(0, 1))).reshape(
            T, B, self.ratio * C, H, W
        )  # 这里得到的输出通道数为2c，下一步要将其池化为c
        x0_in = x0_out.permute(0, 1, 3, 4, 2).reshape(T * B * H, W, self.ratio * C)  # TBCHW  →TBHWC
        x0_in = self.pooling(x0_in).reshape(T, B, H, W, C).permute(0, 1, 4, 2, 3)

        x1_out = self.bn1(self.conv1(self.lif1(x1 + x0_in).flatten(0, 1))).reshape(
            T, B, self.ratio * C, H, W
        )
        x1_in = x1_out.permute(0, 1, 3, 4, 2).reshape(T * B * H, W, self.ratio * C)  # TBCHW  →TBHWC
        x1_in = self.pooling(x1_in).reshape(T, B, H, W, C).permute(0, 1, 4, 2, 3)

        x2_out = self.bn2(self.conv2(self.lif2(x2 + x1_in).flatten(0, 1))).reshape(
            T, B, self.ratio * C, H, W
        )
        x2_in = x2_out.permute(0, 1, 3, 4, 2).reshape(T * B * H, W, self.ratio * C)  # TBCHW  →TBHWC
        x2_in = self.pooling(x2_in).reshape(T, B, H, W, C).permute(0, 1, 4, 2, 3)

        x3_out = self.bn3(self.conv3(self.lif3(x3 + x2_in).flatten(0, 1))).reshape(
            T, B, self.ratio * C, H, W
        )
        # x3_in = x3.permute(0,1,3,4,2)  #TBCHW  →TBHWC
        # x3_in = self.pooling(x3_in).permute(0,1,4,2,3)

        x = torch.cat([x0_out, x1_out, x2_out, x3_out], dim=2)
        return x


class Res2NetConv3(nn.Module):  # 将conv2的maxpooling替换为conv
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
    ):
        super().__init__()

        self.ratio = int(out_channel / in_channel)

        self.conv0 = RepConv(int(in_channel / 4), int(out_channel / 4), group=1)
        self.conv1 = RepConv(int(in_channel / 4), int(out_channel / 4), group=1)
        self.conv2 = RepConv(int(in_channel / 4), int(out_channel / 4), group=1)
        self.conv3 = RepConv(int(in_channel / 4), int(out_channel / 4), group=1)

        self.conv0_back = RepConv(
            int(out_channel / 4), int(in_channel / 4), group=1
        )  # 替代maxpooling取回信息
        self.conv1_back = RepConv(int(out_channel / 4), int(in_channel / 4), group=1)
        self.conv2_back = RepConv(int(out_channel / 4), int(in_channel / 4), group=1)
        self.conv3_back = RepConv(int(out_channel / 4), int(in_channel / 4), group=1)

        self.bn0 = nn.BatchNorm2d(int(out_channel / 4))
        self.bn1 = nn.BatchNorm2d(int(out_channel / 4))
        self.bn2 = nn.BatchNorm2d(int(out_channel / 4))
        self.bn3 = nn.BatchNorm2d(int(out_channel / 4))

        self.bn0_back = nn.BatchNorm2d(int(in_channel / 4))
        self.bn1_back = nn.BatchNorm2d(int(in_channel / 4))
        self.bn2_back = nn.BatchNorm2d(int(in_channel / 4))
        # self.bn3_back = nn.BatchNorm2d(int(in_channel / 4))

        self.lif0 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.lif1 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.lif2 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.lif3 = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

        self.lif0_back = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.lif1_back = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.lif2_back = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        # self.lif3_back =  MultiStepLIFNode(detach_reset=True,v_reset=None, backend='torch')

        self.pooling = nn.MaxPool1d(kernel_size=self.ratio)

    def forward(self, x):
        x0, x1, x2, x3 = x.chunk(
            4, dim=2
        )  # 是否改变通道数时，在前3维度加个max_pooling？（比如通道维度的max_pooling）
        T, B, C, H, W = x0.shape
        x0_out = self.bn0(self.conv0(self.lif0(x0).flatten(0, 1))).reshape(
            T, B, self.ratio * C, H, W
        )  # 这里得到的输出通道数为2c，下一步要将其池化为c
        x0_in = self.bn0_back(self.conv0_back(self.lif0_back(x0_out).flatten(0, 1))).reshape(
            T, B, C, H, W
        )
        x1 = x1 + x0_in

        x1_out = self.bn1(self.conv1(self.lif1(x1 + x0_in).flatten(0, 1))).reshape(
            T, B, self.ratio * C, H, W
        )
        x1_in = self.bn1_back(self.conv1_back(self.lif1_back(x1_out).flatten(0, 1))).reshape(
            T, B, C, H, W
        )
        x2 = x2 + x1_in

        x2_out = self.bn2(self.conv2(self.lif2(x2 + x1_in).flatten(0, 1))).reshape(
            T, B, self.ratio * C, H, W
        )
        x2_in = self.bn2_back(self.conv2_back(self.lif2_back(x2_out).flatten(0, 1))).reshape(
            T, B, C, H, W
        )
        x3 = x3 + x2_in

        x3_out = self.bn3(self.conv3(self.lif3(x3 + x2_in).flatten(0, 1))).reshape(
            T, B, self.ratio * C, H, W
        )

        x = torch.cat([x0_out, x1_out, x2_out, x3_out], dim=2)
        return x


class Ann_SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=3,  # 7,3
        padding=1,
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        med_channels = int(expansion_ratio * dim)
        self.lif1 = nn.ReLU()
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.lif2 = nn.ReLU()
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,  # 7*7
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv

        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif1(
            x
        )  # x1_lif:0.2328  x2_lif:0.0493  这里x2的均值偏小，因此其经过bn和lif后也偏小，发放率比较低；而x1均值偏大，因此发放率也高
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(
            T, B, -1, H, W
        )  # flatten：从第0维开始，展开到第一维
        x = self.lif2(x)
        x = self.dwconv(x.flatten(0, 1))
        x = self.bn2(self.pwconv2(x)).reshape(T, B, -1, H, W)
        return x


class Ann_ConvBlock(nn.Module):
    def __init__(
        self, input_dim, mlp_ratio=4.0, sep_kernel_size=7
    ):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.Conv = Ann_SepConv(dim=input_dim, kernel_size=sep_kernel_size)  # 内部扩张2倍
        self.mlp_ratio = mlp_ratio

        self.lif1 = nn.ReLU()
        # self.conv1 = nn.Conv2d(input_dim, input_dim*mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False)
        self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio))
        self.bn1 = nn.BatchNorm2d(int(input_dim * mlp_ratio))  # 这里可以进行改进

        # self.conv2 = nn.Conv2d(input_dim*mlp_ratio, input_dim, kernel_size=3, padding=1, groups=1, bias=False)
        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进
        self.lif2 = nn.ReLU()

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  # sepconv  pw+dw+pw

        x_feat = x
        # x = self.conv0(x)  #实现脉冲的稀疏化
        # x = self.bn0(self.conv0(self.lif0(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(
            T, B, int(self.mlp_ratio * C), H, W
        )
        # repconv，对应conv_mixer，包含1*1,3*3,1*1三个卷积，等价于一个3*3卷积
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class Ann_DownSampling(nn.Module):
    def __init__(
        self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = nn.ReLU()
        # self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_lif"):  # 如果不是第一层
            # x_pool = self.pool(x)
            x = self.encode_lif(x)

        x = self.encode_conv(x.flatten(0, 1))
        _, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        return x


class Ann_StandardConv(nn.Module):
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, d=1
    ):  # in_channels(out_channels), 内部扩张比例
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.lif = nn.ReLU()

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.bn(self.conv(self.lif(x).flatten(0, 1))).reshape(
            T, B, self.c2, int(H / self.s), int(W / self.s)
        )
        return x


class Ann_Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif = nn.ReLU()
        self.bn = nn.BatchNorm2d(c2)
        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        return x


class Ann_ConvWithoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)
        self.lif = nn.ReLU()

        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.conv(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        return x


class Ann_SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Ann_Conv(c1, c_, 1, 1)
        self.cv2 = Ann_Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            T, B, C, H, W = x.shape
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x.flatten(0, 1)).reshape(T, B, -1, H, W)
            y2 = self.m(y1.flatten(0, 1)).reshape(T, B, -1, H, W)
            y3 = self.m(y2.flatten(0, 1)).reshape(T, B, -1, H, W)
            return self.cv2(torch.cat((x, y1, y2, y3), 2))


class MS_MLP_SMT(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

        self.dw_lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features
        )

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x_feat = self.dw_lif(x).reshape(T, B, self.c_hidden, H, W)
        x_feat = self.dwconv(x_feat.flatten(0, 1)).reshape(T, B, self.c_hidden, N)

        x = self.fc2_lif(x + x_feat)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class MetaSDSA(nn.Module):  # 这个没用多头注意力
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        # self.pool = Erode()
        self.qkv_conv = nn.Sequential(
            RepConv(dim, dim * 2, bias=False),
            nn.BatchNorm2d(dim * 2),
        )
        self.register_parameter("scale", nn.Parameter(torch.tensor([0.0])))
        if spike_mode == "lif":
            self.qk_lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        elif spike_mode == "plif":
            self.qk_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        if spike_mode == "lif":
            self.v_lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        elif spike_mode == "plif":
            self.v_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        if spike_mode == "lif":
            self.proj_lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        elif spike_mode == "plif":
            self.proj_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        if spike_mode == "lif":
            self.talking_heads_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.talking_heads_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch"
            )

        self.proj_conv = nn.Sequential(
            RepConv(dim, dim),
            nn.BatchNorm2d(dim),
        )

        if spike_mode == "lif":
            self.shortcut_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.shortcut_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        self.mode = mode
        self.layer = layer

    def forward(self, x, hook=None):  # 这里本质上只是用了transformer的算子，而没有切片等操作
        T, B, C, H, W = x.shape
        identity = x

        x = self.shortcut_lif(x)  # 把x变为脉冲 [T,B,C,W,H]
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        qkv_conv_out = self.qkv_conv(x.flatten(0, 1)).reshape(
            T, B, 2 * C, H, W
        )  # [T,B,C,W,H]→[T,B,2C,W,H] .通过repconv使通道数加倍,一个是qk，一个是v
        qk, v = qkv_conv_out.chunk(2, dim=2)
        del qkv_conv_out
        qk = self.qk_lif(qk)  # 变为脉冲形式
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_qk_lif"] = qk.detach()

        qk = jit_sum(qk)  # [T,B,C,W,H]→[T,B,C,1,1]
        qk = self.talking_heads_lif(qk)  # 再变为脉冲形式。只要有数就是1，无数就是0
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_qk"] = qk.detach()

        v = self.v_lif(v)  #
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v"] = v.detach()

        x = jit_mul(qk, v)  # 做哈达玛积，可以理解为通过qk对v进行mask。qk包含了全局信息
        del qk, v
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        x = self.proj_lif(x)
        x = x * self.scale

        # x = identity - x
        return x


class MS_ConvBlock_resnet50(nn.Module):
    def __init__(
        self, input_dim, mlp_ratio=4.0, sep_kernel_size=7
    ):  # in_channels(out_channels), 内部扩张比例
        super().__init__()
        # self.lif0 =  MultiStepLIFNode(detach_reset=True,v_reset=None, backend='torch')
        # self.conv0 = RepConv(input_dim, input_dim)
        # self.bn0 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进
        self.conv1 = MS_StandardConv(input_dim, input_dim, 3)
        self.conv2 = MS_StandardConv(input_dim, input_dim, 3)

        # self.conv0 = MS_StandardConv(input_dim, input_dim)
        # self.conv0 = nn.Conv2d(
        #     input_dim, input_dim, kernel_size=7, #7*7
        #     padding=3, groups=int(input_dim), bias=False)  # depthwise conv
        # self.bn0 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进
        # self.lif0 =  MultiStepLIFNode(detach_reset=True,v_reset=None, backend='torch')

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape

        x_feat = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x_feat + x

        return x


class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()

        self.attn = MetaSDSA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.conv = MS_StandardConv(dim, dim)

        # self.attn = MS_Attention_RepConv(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        #     sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_SMT(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x) + self.conv(x)  # 尝试添加并行的卷积
        x = x + self.mlp(x)

        return x


class MS_Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):  # 这里输入x是一个list
        for i in range(len(x)):
            if x[i].dim() == 5:
                x[i] = x[i].mean(0)
        return torch.cat(x, self.d)


class MS_Attention_RepConv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.head_lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

        self.qkv_conv = nn.Sequential(RepConv(dim, dim * 3, bias=False), nn.BatchNorm2d(dim * 3))

        self.q_lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

        self.k_lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

        self.v_lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")

        self.attn_lif = MultiStepLIFNode(
            tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch"
        )

        self.proj_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = self.head_lif(x)

        qkv_conv_out = self.qkv_conv(x.flatten(0, 1)).reshape(T, B, 3 * C, H, W)
        q, k, v = qkv_conv_out.chunk(3, dim=2)
        del qkv_conv_out
        q = self.q_lif(q).flatten(3)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        # 此时q的shape为[1,1,8,256,32]。8是头数

        k = self.k_lif(k).flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v).flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v  # KT[32,256] * V:[256,32]  #256=n=h*w，32=C/H=d  即[d,n]*[n,d]
        # 最后会的得到一个d^2权重，来评估k，v不同维信息的相关性，也可以理解为不同特征图之间的相关性
        x = (q @ x) * self.scale  #

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, H, W)

        return x


# ==============================下面是ems-yolo网络架构


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        temp = temp / (2 * lens)  # ？
        return grad_input * temp.float()


act_fun = ActFun.apply


class batch_norm_2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()  # num_features=16
        self.bn = BatchNorm3d1(
            num_features
        )  # input (N,C,D,H,W) 进行C-dimension batch norm on (N,D,H,W) slice. spatio-temporal Batch Normalization

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return (
            y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)
        )  # 原始输入是(T,N,C,H,W) BN处理时转变为(N,C,T,H,W)


class batch_norm_2d1(nn.Module):
    # 与batch_norm_2d 差异仅在初始化的weight更小
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm3d2(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d2(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            # print('是否执行这里')
            nn.init.constant_(self.weight, 0.2 * thresh)
            nn.init.zeros_(self.bias)


class BatchNorm3d1(torch.nn.BatchNorm3d):  # 5通道的，同时具有学习参数
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)  # 常数初始化
            nn.init.zeros_(self.bias)


# class mem_update(nn.Module):
#     def __init__(self, act=False):
#         super(mem_update, self).__init__()
#         # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)
#         self.actFun = nn.SiLU()
#         self.act = act

#     def forward(self, x):
#         mem = torch.zeros_like(x[0]).to(x.device)
#         spike = torch.zeros_like(x[0]).to(x.device)
#         output = torch.zeros_like(x)
#         mem_old = 0
#         for i in range(time_window):
#             if i >= 1:
#                 mem = mem_old * decay * (1 - spike.detach()) + x[i]
#             else:
#                 mem = x[i]
#             if self.act:
#                 spike = self.actFun(mem)
#             else:
#                 spike = act_fun(mem)

#             mem_old = mem.clone()
#             output[i] = spike
#         # print(output[0][0][0][0])
#         return output


class Spiking_vit_MetaFormer(nn.Module):  # 主干网络
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dim=[64, 128, 256],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        # embed_dim = [64, 128, 256, 512]

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        self.downsample1 = MS_DownSampling(
            in_channels=in_channels,  # 3
            embed_dims=embed_dim // 8,  # 768/8
            kernel_size=7,
            stride=4,
            padding=2,
            first_layer=True,
        )

        self.ConvBlock1_1 = nn.ModuleList([MS_ConvBlock(dim=embed_dim // 8, mlp_ratio=mlp_ratios)])

        self.ConvBlock1_2 = nn.ModuleList([MS_ConvBlock(dim=embed_dim // 8, mlp_ratio=mlp_ratios)])

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim // 8,
            embed_dims=embed_dim // 4,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock2_1 = nn.ModuleList([MS_ConvBlock(dim=embed_dim // 4, mlp_ratio=mlp_ratios)])

        self.ConvBlock2_2 = nn.ModuleList([MS_ConvBlock(dim=embed_dim // 4, mlp_ratio=mlp_ratios)])

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim // 4,
            embed_dims=embed_dim // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block(  # 512,8,4,
                    dim=embed_dim // 2,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                )
                for j in range(8)
            ]
        )

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim // 2,
            embed_dims=embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.block4 = nn.ModuleList(
            [
                MS_Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                )
                for j in range(1)
            ]
        )

        # classification head 这里不需要脉冲，因为输入的是在T时长平均发射值
        self.lif = MultiStepLIFNode(detach_reset=True, v_reset=None, backend="torch")
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return (
                F.interpolate(
                    pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                    size=(H, W),
                    mode="bilinear",
                )
                .reshape(1, -1, H * W)
                .permute(0, 2, 1)
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        x = self.downsample1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)

        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)

        return x  # T,B,C,N

    def forward(self, x):
        T = 1
        x = (x.unsqueeze(0)).repeat(T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = x.flatten(3).mean(3)
        x = self.head(self.lif(x)).mean(0)
        return x


def spikformer_8_384_CAFormer(**kwargs):
    model = Spiking_vit_MetaFormer(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=768,
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model


from timm.models import create_model

if __name__ == "__main__":
    H = 224
    W = 224
    C = 3

    x = torch.randn(2, 3, 224, 224).cuda()
    model = spikformer_8_384_CAFormer().cuda()
    # torchinfo.summary(model, (2, 3, 224, 224))
    y = model(x)
    print("finish")
