import logging

import cv2
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset

from core.config import configurable
from core.data import TRANSFORMS_REGISTRY
from core.data.preprocess_utils import preprocess_annotations
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
        dataset_dict = preprocess_annotations(dataset_dict)  # list -> Instances
        dataset_dict["shape"] = dataset_dict["ori_shape"]  # init sample shape

        # 根据模态pop不用的数据
        if self.modality == "image":
            dataset_dict.pop("events")
        elif self.modality == "events":
            dataset_dict.pop("image")

        for tfm in self.transforms:
            dataset_dict = tfm.apply_transform(dataset_dict)

        # 更新transform后的shape
        dataset_dict["instances"]._shape = dataset_dict["shape"]
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
        dataset_dict["shape"] = dataset_dict["ori_shape"]  # init sample shape
        if self.modality == "image":
            dataset_dict.pop("events")
        elif self.modality == "events":
            dataset_dict.pop("image")

        for tfm in self.transforms:
            dataset_dict = tfm.apply_transform(dataset_dict)

        # 更新transform后的shape
        dataset_dict["instances"]._shape= dataset_dict["shape"]
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
