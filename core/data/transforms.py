import random
import sys
from copy import deepcopy
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from core.data import TRANSFORMS_REGISTRY
from core.data.preprocess_utils import preprocess_annotations
from core.structures import Boxes, Instances


@TRANSFORMS_REGISTRY.register()
class Mosaic:
    def __init__(
        self,
        dataset: Dataset,
        mosaic_size: list[int, int],
        modality: str,
        close_mosaic_epoch: int,
        *args,
        **kwargs,
    ):
        self.dataset = dataset  # dataset_api for read another samples
        self.mosaic_size = mosaic_size
        self.modality = modality
        self.close_mosaic_epoch = close_mosaic_epoch
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def apply_transform(self, dataset_dict: dict):
        if self.current_epoch >= self.close_mosaic_epoch:
            return dataset_dict
        dataset_dict = deepcopy(dataset_dict)
        indices = [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        dataset_dicts = [dataset_dict] + [self.dataset[i] for i in indices]

        # 将标注进行预处理
        for i in range(len(dataset_dicts)):
            dataset_dicts[i] = preprocess_annotations(dataset_dicts[i])

        # 创建mosaic 图像/events
        dataset_dict = self._create_mosaic(dataset_dict, dataset_dicts)

        return dataset_dict

    def _create_mosaic(self, dataset_dict: dict, dataset_dicts: list[dict]):
        # 生成2x2 mosaic中心点坐标
        size_h, size_w = self.mosaic_size
        xc = random.randint(int(size_w * 0.4), int(size_w * 0.6))
        yc = random.randint(int(size_h * 0.4), int(size_h * 0.6))

        if self.modality != "image":
            event_shape = dataset_dicts[0]["events"].shape
            mosaic_events = torch.zeros(
                [*list(event_shape[:2]), size_h, size_w], dtype=torch.float32
            )  # tag: 这里能否改成int8
        if self.modality != "events":
            mosaic_img = np.full((size_h, size_w, 3), 114, dtype=np.uint8)

        mosaic_bboxes = []
        mosaic_classes = []

        # 每个pos按top-left x,y, bottom-right x,y顺序排列
        positions = [
            (0, 0, xc, yc),  # 左上
            (xc, 0, size_w, yc),  # 右上
            (0, yc, xc, size_h),  # 左下
            (xc, yc, size_w, size_h),  # 右下
        ]
        for idx, pos in enumerate(positions):
            scale = random.uniform(0.5, 1.5)
            h, w = (
                dataset_dicts[idx]["image"].shape[:2]  # h,w,c
                if self.modality != "events"
                else dataset_dicts[idx]["events"].shape[2:]
            )
            new_h, new_w = int(h * scale), int(w * scale)

            # mosaic上顶点坐标
            x1a, y1a, x2a, y2a = self._mosaic_pos(idx=idx, pos=pos, w=new_w, h=new_h)
            # 原图上裁切后顶点坐标
            x1b, y1b, x2b, y2b = new_w - (x2a - x1a), new_h - (y2a - y1a), new_w, new_h

            # 处理该样本对应的所有标注信息
            bboxes, classes = self._mosaic_anns(
                idx=idx,
                instances=dataset_dicts[idx]["instances"],
                mosaic_pos=[x1a, y1a, x2a, y2a],
                ori_pos=[x1b, y1b, x2b, y2b],
                scale=scale,
            )
            mosaic_bboxes.extend(bboxes)
            mosaic_classes.extend(classes)

            # 将图像或events放在mos对应位置
            if self.modality != "image":
                events = deepcopy(dataset_dicts[idx]["events"])
                T, C, H, W = events.shape
                events = events.reshape(-1, H, W)
                events = F.interpolate(
                    events.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="nearest",
                ).squeeze(1)
                events = events.reshape(T, C, new_h, new_w)
                # 裁剪events
                events = events[:, :, y1b:y2b, x1b:x2b]
                # 放置到mosaic上
                mosaic_events[:, :, y1a:y2a, x1a:x2a] = events

            if self.modality != "events":
                image = deepcopy(dataset_dicts[idx]["image"])
                image = cv2.resize(image, (new_w, new_h))
                image = image[y1b:y2b, x1b:x2b]
                mosaic_img[y1a:y2a, x1a:x2a] = image

        # 创建新的实例信息(sample_idx为原图像的idx)
        target = Instances(
            ori_image_size=dataset_dict["ori_shape"], sample_idx=dataset_dict["sample_idx"]
        )
        #
        target._shape = [size_h, size_w]
        target.gt_bboxes = Boxes(mosaic_bboxes)
        target.gt_classes = torch.tensor(mosaic_classes, dtype=torch.int64)
        dataset_dict["instances"] = target
        dataset_dict["shape"] = [size_h, size_w]
        if self.modality != "image":
            dataset_dict["events"] = mosaic_events
        if self.modality != "events":
            dataset_dict["image"] = mosaic_img
        return dataset_dict

    def _mosaic_pos(self, idx, pos, w, h):
        """
        根据mosaic图像中心点坐标和裁剪后图像大小,根据样本在四宫格中的位置计算mosaic图像上该样本的坐标
        """
        # top left
        if idx == 0:
            x1a, y1a, x2a, y2a = max(pos[2] - w, 0), max(pos[3] - h, 0), pos[2], pos[3]
        # top right
        elif idx == 1:
            x1a, y1a, x2a, y2a = pos[0], max(pos[3] - h, 0), min(pos[0] + w, pos[2]), pos[3]
        # bottom left
        elif idx == 2:
            x1a, y1a, x2a, y2a = max(pos[2] - w, 0), pos[1], pos[2], min(pos[1] + h, pos[3])
        # bottom right
        elif idx == 3:
            x1a, y1a, x2a, y2a = pos[0], pos[1], min(pos[0] + w, pos[2]), min(pos[1] + h, pos[3])
        return x1a, y1a, x2a, y2a

    def _mosaic_anns(
        self, idx: int, instances: Instances, mosaic_pos: list, ori_pos: list, scale: int
    ):
        x1a, y1a, x2a, y2a = mosaic_pos
        x1b, y1b, _, _ = ori_pos
        bboxes = []
        classes = []
        for ann_idx in range(len(instances.gt_bboxes)):
            x1, y1, x2, y2 = instances.gt_bboxes[ann_idx][0]  # xyxy_abs
            cls = instances.gt_classes[ann_idx]
            # 应用缩放
            x1 = int(x1 * scale)
            y1 = int(y1 * scale)
            x2 = int(x2 * scale)
            y2 = int(y2 * scale)

            # 应用平移
            shift_x = x1a - x1b
            shift_y = y1a - y1b

            x1 += shift_x
            y1 += shift_y
            x2 += shift_x
            y2 += shift_y

            # 裁剪边界
            x1 = max(x1, x1a)
            x2 = min(x2, x2a)
            y1 = max(y1, y1a)
            y2 = min(y2, y2a)

            # 过滤无效标签
            area = (x2 - x1) * (y2 - y1)
            if area <= 20:  # tag: 过滤过小目标框的阈值
                continue

            bboxes.append([x1, y1, x2, y2])
            classes.append(cls)

        return bboxes, classes


@TRANSFORMS_REGISTRY.register()
class ResizeKeepRatio:
    """
    保持长宽比的同时resize image/events.
    尝试将短边缩放到给定的`short_edge_length`的同时长边不超过`max_size`.
    如果达到`max_size`则缩小比例直到长边不超过`max_size`.
    """

    def __init__(
        self,
        short_edge_length,
        max_size=sys.maxsize,
        stride=32,
        sample_style="choice",
        img_interp=cv2.INTER_LINEAR,
        events_interp="nearest",
        *args,
        **kwargs,
    ):
        """
        Args:
            short_edge_length (list[int]): 缩放后的短边长度.
            max_size (int): 长边的最大长度.
            stride (int): 模型下采样的最大倍数.
            sample_style (str): "range" or "choice".
                如果是"range", 则`short_edge_length`表示[min, max],从该范围内随机选择一个值,
                如果是"choice", 则从`short_edge_length`中随机选择一个值.
            img_interp (int): PIL插值方法.
        """
        super().__init__()
        assert sample_style in ["range", "choice"], "sample_style must be 'range' or 'choice'!"

        self.is_range = sample_style == "range"
        self.stride = stride
        self.img_interp = img_interp
        self.events_interp = events_interp
        self.max_size = max_size

        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )

        self.short_edge_length = short_edge_length

    def apply_transform(
        self,
        dataset_dict: dict,
    ):
        image = dataset_dict.get("image", None)
        events = dataset_dict.get("events", None)
        annotations = dataset_dict.get("instances", None)

        h, w = image.shape[:2] if image is not None else events.shape[2:]

        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)

        new_h, new_w = self.get_output_shape(
            oldh=h, oldw=w, short_edge_length=size, max_size=self.max_size, stride=self.stride
        )
        new_image = None
        new_events = None
        new_annotations = None

        if image is not None:
            new_image = self.resize_image(image, new_h, new_w)

        if events is not None:
            new_events = self.resize_events(events, new_h, new_w)

        if annotations is not None:
            new_annotations = deepcopy(annotations)
            scale_x = new_w * 1.0 / w
            scale_y = new_h * 1.0 / h
            new_annotations.gt_bboxes.scale(scale_x, scale_y)

        dataset_dict["image"] = new_image
        dataset_dict["events"] = new_events
        dataset_dict["instances"] = new_annotations
        dataset_dict["shape"] = [new_h, new_w]
        return dataset_dict

    def resize_image(self, image: np.ndarray, new_h, new_w):
        img = cv2.resize(image, (new_w, new_h), interpolation=self.img_interp)
        return np.asarray(img)

    def resize_events(self, events: torch.Tensor, new_h, new_w):
        T, C, H, W = events.shape
        events = events.reshape(-1, H, W)  # [T, C, H, W] -> [T*C, H, W]

        events = F.interpolate(
            events.unsqueeze(1),
            size=(new_h, new_w),
            mode=self.events_interp,
        ).squeeze(1)

        events = events.reshape(T, C, new_h, new_w)
        return events

    @staticmethod
    def get_output_shape(
        oldh: int, oldw: int, short_edge_length: int, max_size: int, stride: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target short edge length.
        """
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale

        # 确保尺寸能被stride整除
        newh = int(np.ceil(newh / stride) * stride)
        neww = int(np.ceil(neww / stride) * stride)

        return (newh, neww)


@TRANSFORMS_REGISTRY.register()
class RandomFlip:
    def __init__(self, prob=0.5, horizontal=True, vertical=False, *args, **kwargs):
        super().__init__()
        self.prob = prob
        self.horizontal = horizontal
        self.vertical = vertical
        assert not (
            self.horizontal and self.vertical
        ), "Cannot do both horiz and vert. Please use two Flip instead."

        assert self.horizontal or self.vertical, "At least one of horiz or vert must be True!"

    def apply_transform(self, dataset_dict: dict):
        image = dataset_dict.get("image", None)
        events = dataset_dict.get("events", None)
        annotations = dataset_dict.get("instances", None)
        h, w = image.shape[:2] if image is not None else events.shape[2:]

        do = self._rand_range() < self.prob
        if not do:
            return dataset_dict

        new_image = None
        new_events = None
        new_annotations = None

        if image is not None:
            if self.horizontal:
                new_image = image[:, ::-1, :]
            elif self.vertical:
                new_image = image[::-1, :, :]
        if events is not None:
            if self.horizontal:
                new_events = events.flip(dims=[-1])  # Flip the last dimension (W)
            elif self.vertical:
                new_events = events.flip(dims=[-2])

        if annotations is not None:
            new_annotations = deepcopy(annotations)

            new_annotations.gt_bboxes.flip(
                height=h, width=w, mode="horizontal" if self.horizontal else "vertical"
            )

        dataset_dict["image"] = new_image
        dataset_dict["events"] = new_events
        dataset_dict["instances"] = new_annotations
        # flip不改变shape，所以不需要更新shape

        return dataset_dict

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)
