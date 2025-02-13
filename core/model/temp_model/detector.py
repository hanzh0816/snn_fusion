import contextlib
from copy import deepcopy
import glob
import logging
import math
import re

import torch
from pathlib import Path

import yaml

from core.config.config import configurable
from core.model import MODEL_REGISTRY
from core.model.head.head_utils import non_max_suppression
from core.model.loss.loss import v8DetectionLoss
from core.structures.instances import Instances
from .yolo_spikformer import *
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in (
        ".yaml",
        ".yml",
    ), f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string
        # Add YAML filename to dict and return
        data = (
            yaml.safe_load(s) or {}
        )  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    yaml_file = glob.glob(path, recursive=True)[0]
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = "n"
    d["yaml_file"] = str(path)
    return d


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (
        d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape")
    )
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (
            SpikeConv,
            SpikeSPPF,
            Ann_SPPF,
            MS_C2f,
        ):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m is MS_C2f:
                args.insert(2, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is SpikeDetect:
            args.append([ch[x] for x in f])

        elif m is MS_GetT:
            c1 = ch[f]
            c2 = args[0]
            args = [c1, c2, *args[1:]]  # 这里定义有问题，应该再加一个输出通道的参数
        elif m is MS_CancelT:
            c1 = ch[f]
            c2 = args[0]
            args = [c1, c2, *args[1:]]

        elif m is MS_DownSampling:
            c1 = ch[f]  # 输入通道数
            c2 = int(args[0] * width)  # 输出通道数
            args = [c1, c2, *args[1:]]
        elif m is Ann_DownSampling:
            c1 = ch[f]  # 输入通道数
            c2 = int(args[0] * width)  # 输出通道数
            args = [c1, c2, *args[1:]]

        elif m is MS_ConvBlock:
            c1 = ch[f]  # 输入通道数
            c2 = c1  # 输出通道数与输入通道数相同
            args = [
                c1,
            ]

        elif m is MS_AllConvBlock:
            c1 = ch[f]  # 输入通道数
            c2 = c1  # 输出通道数与输入通道数相同
            args = [c1, *args[0:]]

        elif m is MS_FullConvBlock:
            c1 = ch[f]  # 输入通道数
            c2 = c1  # 输出通道数与输入通道数相同
            args = [c1, *args[0:]]

        elif m is MS_ConvBlock_resnet50:
            c1 = ch[f]  # 输入通道数
            c2 = c1  # 输出通道数与输入通道数相同
            args = [c1, *args[0:]]

        elif m is Ann_ConvBlock:
            c1 = ch[f]  # 输入通道数
            c2 = c1  # 输出通道数与输入通道数相同
            args = [c1, *args[0:]]

        elif m is MS_Block:
            c1 = ch[f]  # 输入通道数
            c2 = c1  # 输入输出通道数相同
            args = [c2, *args[0:]]
        elif m is Ann_StandardConv:
            c1 = ch[f]  # 输入通道数
            c2 = int(args[0] * width)
            args = [c1, c2, *args[1:]]
        elif m is MS_StandardConv:
            c1 = ch[f]  # 输入通道数
            c2 = min(int(args[0] * width), int(max_channels * width))
            args = [c1, c2, *args[1:]]

        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


@MODEL_REGISTRY.register()
class SpikeYOLO_v0(nn.Module):

    @configurable
    def __init__(self, ch=3, nc=None, verbose=True, *args, **kwargs):
        super().__init__()
        cfg = "/home/hzh/code/rgb_event_fusion/snn_fusion/core/model/temp_model/snn_yolov8.yaml"
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(
            deepcopy(self.yaml), ch=ch, verbose=verbose
        )  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        self.shape = None  # predict时所用的shape,当shape变化时要重新计算anchors
        self.anchors = torch.empty(0)  # anchors
        self.strides = torch.empty(0)  # strides

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (SpikeDetect)):
            m.stride = torch.tensor([8, 16, 32])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)

        self.criterion = v8DetectionLoss(
            reg_max=16,
            num_classes=2,
            num_out_channels=66,
            stride=[8, 16, 32],
            box_weight=7.5,
            cls_weight=0.5,
            dfl_weight=1.5,
        )
        self.dfl = SpikeDFL(16)

    @classmethod
    def from_config(cls, cfg):
        return {}

    def forward(self, x):
        # 单event通路不需要image输入

        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            #             print("m:",m)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def loss(self, inputs):
        events = inputs["events"]
        target: list["Instances"] = inputs["instances"]

        features = self.forward(events)
        loss_dict = self.criterion(features, target)

        return loss_dict

    def predict(self, inputs):
        events = inputs["events"]
        target: list["Instances"] = inputs["instances"]

        shape = events.shape[2:]

        x = self.forward(events)

        loss_dict = self.criterion(x, target)

        if self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )

        batch_size = x[0].shape[0]
        x_cat = torch.cat(
            [xi.view(batch_size, 66, -1) for xi in x], 2
        )  # [bs, nc+reg_max*4, sum_of_anchors]
        box, cls = x_cat.split((16 * 4, 2), 1)

        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        preds = torch.cat((dbox, cls.sigmoid()), 1)

        instances = non_max_suppression(
            prediction=preds,
            conf_thres=0.001,
            iou_thres=0.7,
            labels=[],
            multi_label=False,
            agnostic=False,
            max_det=300,
            max_time_img=0.5,
        )
        # print([len(ins) for ins in instances])

        return {"instances": instances, "loss": loss_dict["loss"]}
