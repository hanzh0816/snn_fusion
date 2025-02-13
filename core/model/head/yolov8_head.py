import math
from typing import List

import torch
from torch import Tensor, nn

from core.config.config import configurable
from core.model.build import MODEL_REGISTRY
from core.model.loss import dist2bbox, make_anchors, v8DetectionLoss
from core.structures.instances import Instances

from .head_utils import *

from core.model.module.ann_module import Conv, DFL


@MODEL_REGISTRY.register()
class YOLOv8Head(nn.Module):
    @configurable
    def __init__(self, ch, num_classes, stride, weight,nms_cfg=None,  *args, **kwargs):
        super().__init__()
        self.in_channels = ch
        self.num_layers = len(ch)
        self.num_classes = num_classes
        self.stride = torch.Tensor(stride)  # tag: 当模型结构修改时，需要从配置文件中修改stride
        self.weight = weight  # 不同loss加权权重
        self.current_epoch = 0
        self.nms_cfg = nms_cfg

        self.reg_max = 16  # DFL channels
        self.num_out_channels = self.num_classes + self.reg_max * 4  # number of outputs per anchor

        reg_med_channels = max((16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_med_channels = max(self.in_channels[0], min(self.num_classes, 100))

        self.shape = None  # predict时所用的shape,当shape变化时要重新计算anchors
        self.anchors = torch.empty(0)  # anchors
        self.strides = torch.empty(0)  # strides

        # dynamic conf thres
        self.current_epoch = 0
        self.dynamic_conf_thres = self.nms_cfg.DYNAMIC_CONF
        self.final_conf_thres = self.nms_cfg.CONF_THRES
        self.k = self.nms_cfg.EXP_K  # 指数调整conf_thres系数

        # nms conf
        self.iou_thres = self.nms_cfg.IOU_THRES
        self.max_det = self.nms_cfg.MAX_DET
        self.max_time_img = self.nms_cfg.MAX_TIME_IMG
        self.agnostic = self.nms_cfg.AGNOSTIC
        self.multi_label = self.nms_cfg.MULTI_LABEL

        # reg conv layers
        self.reg_conv = nn.ModuleList(
            nn.Sequential(
                Conv(
                    in_channels=in_ch,
                    out_channels=reg_med_channels,
                    kernel_size=3,
                ),
                Conv(
                    in_channels=reg_med_channels,
                    out_channels=reg_med_channels,
                    kernel_size=3,
                ),
                nn.Conv2d(
                    in_channels=reg_med_channels,
                    out_channels=self.reg_max * 4,
                    kernel_size=1,
                ),
            )
            for in_ch in self.in_channels
        )

        # cls conv layers
        self.cls_conv = nn.ModuleList(
            nn.Sequential(
                Conv(
                    in_channels=in_ch,
                    out_channels=cls_med_channels,
                    kernel_size=3,
                ),
                Conv(
                    in_channels=cls_med_channels,
                    out_channels=cls_med_channels,
                    kernel_size=3,
                ),
                nn.Conv2d(
                    in_channels=cls_med_channels,
                    out_channels=self.num_classes,
                    kernel_size=1,
                ),
            )
            for in_ch in self.in_channels
        )

        self.det_loss = v8DetectionLoss(
            reg_max=self.reg_max,
            num_classes=self.num_classes,
            num_out_channels=self.num_out_channels,
            stride=self.stride,
            box_weight=self.weight[0],
            cls_weight=self.weight[1],
            dfl_weight=self.weight[2],
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    @property
    def conf_thres(self):
        if self.dynamic_conf_thres:
            return self._get_dynamic_conf_thres()
        else:
            return self.final_conf_thres

    def _get_dynamic_conf_thres(self):
        return self.final_conf_thres * (1 - math.exp(-self.k * self.current_epoch))

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        stride = cfg.STRIDE
        weight = cfg.WEIGHT
        nms_cfg = cfg.NMS
        return {"stride": stride, "weight": weight,"nms_cfg": nms_cfg, **kwargs}

    def forward(self, x: List[Tensor]):
        # reg & cls for each level
        for i in range(self.num_layers):
            x[i] = torch.cat((self.reg_conv[i](x[i]), self.cls_conv[i](x[i])), dim=1)

        return x

    def loss(self, x: List[Tensor], target: list["Instances"]):
        x = self.forward(x)
        return self.det_loss(x, target)

    def predict(self, x: List[Tensor], target: list["Instances"]):
        shape = x[0].shape
        x = self.forward(x)
        loss_dict = self.det_loss(x, target)
        if self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        batch_size = shape[0]
        x_cat = torch.cat(
            [xi.view(batch_size, self.num_out_channels, -1) for xi in x], 2
        )  # [bs, nc+reg_max*4, sum_of_anchors]
        box, cls = x_cat.split((self.reg_max * 4, self.num_classes), 1)

        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        preds = torch.cat((dbox, cls.sigmoid()), 1)

        instances = non_max_suppression(
            prediction=preds,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            labels=[],
            multi_label=self.multi_label,
            agnostic=self.agnostic,
            max_det=self.max_det,
            max_time_img=self.max_time_img,
        )
        return {"instances": instances, "loss": loss_dict["loss"]}
