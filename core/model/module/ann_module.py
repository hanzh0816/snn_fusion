import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel_size, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel_size-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


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


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, in_channels, out_channels, shortcut=True, groups=1, kernel_size=(3, 3), ratio=0.5
    ):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(out_channels * ratio)  # hidden channels
        self.cv1 = Conv(in_channels, c_, kernel_size[0], 1)
        self.cv2 = Conv(c_, out_channels, kernel_size[1], 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel_size, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        act=True,
        *args,
        **kwargs
    ):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        in_channels,
        out_channels,
        repeat_nums=1,
        shortcut=False,
        groups=1,
        ratio=0.5,
        *args,
        **kwargs
    ):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.hidden_channels = int(out_channels * ratio)  # hidden channels
        self.cv1 = Conv(in_channels, 2 * self.hidden_channels, 1, 1)
        self.cv2 = Conv(
            (2 + repeat_nums) * self.hidden_channels, out_channels, 1
        )  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(
                self.hidden_channels,
                self.hidden_channels,
                shortcut,
                groups,
                kernel_size=((3, 3), (3, 3)),
                ratio=1.0,
            )
            for _ in range(repeat_nums)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(_m(y[-1]) for _m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.hidden_channels, self.hidden_channels), 1))
        y.extend(_m(y[-1]) for _m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, in_channels, out_channels, kernel_size=5, *args, **kwargs):
        """
        Initializes the SPPF layer with given input/output channels and kernel_size size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        hidden_channels = in_channels // 2  # hidden channels
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.max_pool(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


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
