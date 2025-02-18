from asyncio import events
import logging
from typing import Callable, Union

import cv2
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from core.config import configurable
from core.data import TRANSFORMS_REGISTRY
from core.structures.boxes import Boxes, BoxMode
from core.structures.instances import Instances
from core.utils.visual import plot_events_map


class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.
    """

    def __init__(self, dataset, map_func):
        """
        Args:
            dataset: a dataset where map function is applied. Can be either
                map-style or iterable dataset. When given an iterable dataset,
                the returned object will also be an iterable dataset.
            map_func: a callable which maps the element in dataset. map_func can
                return None to skip the data (e.g. in case of errors).
                How None is handled depends on the style of `dataset`.
                If `dataset` is map-style, it randomly tries other elements.
                If `dataset` is iterable, it skips the data and tries the next.
        """
        self._dataset = dataset
        self._map_func = map_func  # wrap so that a lambda will work

    def __new__(cls, dataset, map_func):
        return super().__new__(cls)

    def __getnewargs__(self):
        return self._dataset, self._map_func

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        cur_idx = int(idx)
        try:
            data = self._map_func(self._dataset[cur_idx])
        except Exception as e:
            raise e
        return data

    def set_epoch(self, epoch):
        if hasattr(self._map_func, "set_epoch"):
            self._map_func.set_epoch(epoch)
        else:
            raise ValueError("The map_func does not have a set_epoch function.")


class DSECDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    # todo: 完善DatasetMapper逻辑
    @configurable
    def __init__(
        self,
        is_train: bool,
        modality: str,
        num_time_bins: int = 100,
        dataset: Dataset = None,
        mosaic_cfg: dict = None,
        transforms: list = [],
        **kwargs,
    ):

        self.is_train = is_train
        self.modality = modality
        self.num_time_bins = num_time_bins
        self.dataset = dataset
        self.transforms = []
        # mosaic settings
        self.mosaic_cfg = mosaic_cfg

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"

        assert self.dataset is not None, logger.warning("[DatasetMapper] dataset must be provided")

        # if self.is_train:
        # only apply transforms in training
        # name list -> class list
        self.transforms = self.build_transforms(transforms)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, *args, **kwargs):
        modality = cfg.DATASETS.MODALITY
        transforms = cfg.DATASETS.TRANSFORMS
        ret = {"modality": modality, "is_train": is_train, "transforms": transforms, **kwargs}
        return ret

    def build_transforms(self, transforms: list[dict]):
        for tfm in transforms:
            tfm_name = tfm["name"]
            self.transforms.append(
                TRANSFORMS_REGISTRY.get(tfm_name)(
                    **tfm, dataset=self.dataset, modality=self.modality
                )
            )
        return self.transforms

    def __call__(self, dataset_dict):
        dataset_dict = self._preprocess_annotations(dataset_dict)  # list -> Instances
        if self.modality == "image":
            dataset_dict.pop("events")
        elif self.modality == "events":
            dataset_dict.pop("image")

        for tfm in self.transforms:
            dataset_dict = tfm.apply_transform(dataset_dict)

        image = dataset_dict.get("image", None)  # 只有image需要转换为tensor
        if image is not None:
            image = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.float32
            )
        # _display_events(dataset_dict["events"])
        dataset_dict["image"] = image
        return dataset_dict

    def set_epoch(self, epoch):
        for tfm in self.transforms:
            if hasattr(tfm, "set_epoch"):
                tfm.set_epoch(epoch)

    def _preprocess_events(self, events, ori_shape, num_time_bins):
        """
        Args:
            - events (dict): 一个字典，包含以下键：
                - "t" (np.ndarray): 事件的时间戳。
                - "x" (np.ndarray): 事件的 x 坐标
                - "y" (np.ndarray): 事件的 y 坐标
                - "p" (np.ndarray): 事件的极性，取值为 1 或 -1(正极性或负极性)
            - ori_shape (tuple): 图像的原始形状(height, width)。
            - num_time_bins (int): 要分割的时间步数，用于将时间戳归一化到这个范围。
        Returns:
            - events_map (torch.Tensor): shape=(T, 2, H, W), 其中 T为分割时间步数num_time_bins

        """
        if events is None:
            return None
        t = np.array(events["t"])
        x = np.array(events["x"])
        y = np.array(events["y"])
        p = np.array(events["p"])

        # 归一化时间戳到 [0, num_time_bins)
        t_min, t_max = t.min(), t.max()
        t_bins = np.floor((t - t_min) / (t_max - t_min) * num_time_bins).astype(int)
        t_bins[t_bins == num_time_bins] = num_time_bins - 1
        # 初始化张量：T 时间步, C=2 极性, H 高度, W 宽度

        height, width = ori_shape
        events_map = np.zeros((num_time_bins, 2, height, width), dtype=np.float32)

        for t_i, x_i, y_i, p_i in zip(t_bins, x, y, p):
            if p_i == 1:  # 正极性
                events_map[t_i, 0, y_i, x_i] += 1
            elif p_i == -1:  # 负极性
                events_map[t_i, 1, y_i, x_i] += 1

        events_map = torch.Tensor(events_map)
        return events_map

    def _preprocess_annotations(self, dataset_dict):
        # todo: detection0/1 转换为annotations序列
        dataset_dict.pop("annotations_next")
        ori_annotations = dataset_dict.pop("annotations")
        # 一个样本中的所有bbox
        bboxes = [self._preprocess_bbox(obj) for obj in ori_annotations]

        target = Instances(dataset_dict["ori_shape"])
        target.gt_bboxes = Boxes(bboxes)

        classes = [int(obj["class_id"]) for obj in ori_annotations]
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes

        dataset_dict["instances"] = target
        return dataset_dict

    def _preprocess_bbox(self, bbox):
        bbox = [bbox["x"], bbox["y"], bbox["w"], bbox["h"]]

        # xywh -> xyxy_abs
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]

        return bbox


class YOLODatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    # todo: 完善DatasetMapper逻辑
    @configurable
    def __init__(
        self,
        is_train: bool,
        modality: str,
        num_time_bins: int = 100,
        dataset: Dataset = None,
        mosaic_cfg: dict = None,
        transforms: list = [],
        **kwargs,
    ):

        self.is_train = is_train
        self.modality = modality
        self.num_time_bins = num_time_bins
        self.dataset = dataset
        self.transforms = []
        # mosaic settings
        self.mosaic_cfg = mosaic_cfg

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"

        assert self.dataset is not None, logger.warning("[DatasetMapper] dataset must be provided")

        # if self.is_train:
        # only apply transforms in training
        # name list -> class list
        self.transforms = self.build_transforms(transforms)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, *args, **kwargs):
        modality = cfg.DATASETS.MODALITY
        transforms = cfg.DATASETS.TRANSFORMS
        ret = {"modality": modality, "is_train": is_train, "transforms": transforms, **kwargs}
        return ret

    def build_transforms(self, transforms: list[dict]):
        for tfm in transforms:
            tfm_name = tfm["name"]
            self.transforms.append(
                TRANSFORMS_REGISTRY.get(tfm_name)(
                    **tfm, dataset=self.dataset, modality=self.modality
                )
            )
        return self.transforms

    def __call__(self, dataset_dict):
        if self.modality == "image":
            dataset_dict.pop("events")
        elif self.modality == "events":
            dataset_dict.pop("image")

        for tfm in self.transforms:
            dataset_dict = tfm.apply_transform(dataset_dict)

        image = dataset_dict.get("image", None)  # 只有image需要转换为tensor
        if image is not None:
            image = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.float32
            )
        # _display_events(dataset_dict["events"])
        dataset_dict["image"] = image
        return dataset_dict

    def set_epoch(self, epoch):
        for tfm in self.transforms:
            if hasattr(tfm, "set_epoch"):
                tfm.set_epoch(epoch)

def _draw_bbox(img, instances):
    bboxes = instances.gt_bboxes.tensor.cpu().numpy()
    for box in bboxes:
        x1, y1, x2, y2 = box
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return img


def _display_events(events):
    T = events.shape[0]
    for t in range(T):
        # 提取正极性（events[:, 0]）和负极性（events[:, 1]）
        positive_event = events[t, 0, :, :].cpu().numpy()  # 正极性
        negative_event = events[t, 1, :, :].cpu().numpy()  # 负极性

        # 将正负极性事件合并为一个图像（可以通过差值或其他方式）
        event_image = positive_event - negative_event  # 负极性和正极性差值

        # 将图像转换为0-255的范围并保存
        event_image = np.clip(event_image, -1, 1)  # 保证值在[-1, 1]之间
        event_image = ((event_image + 1) * 127.5).astype(np.uint8)  # 转换到[0, 255]范围
        cv2.imwrite(f"event_image_{t}.png", event_image)
