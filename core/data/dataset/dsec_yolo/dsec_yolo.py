import glob
import hashlib
import logging
import math
import multiprocessing as mp
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
from itertools import repeat
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import Dataset
from tqdm import tqdm as tqdm_original

from core.structures.boxes import Boxes
from core.structures.instances import Instances
from core.utils import comm

IMG_FORMATS = (
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
    "npy",
)  # image suffixes
DATASET_CACHE_VERSION = "1.0.3"
VERBOSE = True
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm bar format
LOGGER = logging.getLogger(__name__)


def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


class TQDM(tqdm_original):
    """
    Custom Ultralytics tqdm class with different default arguments.

    Args:
        *args (list): Positional arguments passed to original tqdm.
        **kwargs (dict): Keyword arguments, with custom defaults applied.
    """

    def __init__(self, *args, **kwargs):
        """Initialize custom Ultralytics tqdm class with different default arguments."""
        # Set new default values (these can still be overridden when calling TQDM)
        kwargs["disable"] = not VERBOSE or kwargs.get(
            "disable", False
        )  # logical 'and' with default value if passed
        kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)  # override default value if passed
        super().__init__(*args, **kwargs)


class YOLODataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(self, img_path, prefix):
        super().__init__()
        """Initialize BaseDataset with given configuration and options."""
        self.img_path = img_path
        self.prefix = prefix
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.ni = len(self.labels)  # number of images
        self.ims, self.im_hw0, self.im_hw = (
            [None] * self.ni,
            [None] * self.ni,
            [None] * self.ni,
        )
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]

    def get_img_files(self, img_path):  # 返回文件夹的列表
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:  # 遍历文件夹
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)  # 返回文件夹中的所有文件
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [
                            x.replace("./", parent) if x.startswith("./") else x for x in t
                        ]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{p} does not exist")
            im_files = sorted(
                x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS
            )
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"No images found in {img_path}"
        except Exception as e:
            raise FileNotFoundError(f"Error loading data from {img_path}\n") from e
        return im_files

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)  # t,h,w,3
                h0, w0 = im.shape[1:3]  # orig hw 统一读取格式

            if len(im.shape) == 4:
                im_shape = im.shape[1:3]
            else:
                im_shape = im.shape[:2]

            return im, (h0, w0)

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        image, ori_shape = self.load_image(index)
        image = torch.Tensor(image).permute(0, 3, 1, 2)
        label = deepcopy(self.labels[index])
        instances = self._preprocess_annotations(label, index, ori_shape)
        return {
            "sample_idx": index,
            "image": None,
            "events": image,
            "image_ts_0": 0,
            "ori_shape": ori_shape,
            "instances": instances,
        }

    @staticmethod
    def _preprocess_annotations(label, index, shape):
        def _preprocess_bbox(bboxes, width, height):
            new_bboxes = []
            for bbox in bboxes:
                x1_c, y1_c, w, h = bbox
                x1_c *= width
                y1_c *= height
                w *= width
                h *= height
                x1 = int(x1_c - w / 2)
                y1 = int(y1_c - h / 2)
                x2 = int(x1_c + w / 2)
                y2 = int(y1_c + h / 2)
                new_bboxes.append([x1, y1, x2, y2])
            return new_bboxes

        h, w = shape
        target = Instances(ori_image_size=shape, sample_idx=index)
        target.gt_bboxes = Boxes(_preprocess_bbox(label["bboxes"], width=w, height=h))
        classes = [int(cls) for cls in label["cls"]]
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes

        return target

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)  # 转成尾缀txt格式
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            # attempt to load a *.cache file
            cache, exists = (load_dataset_cache_file(cache_path), True)
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            # assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            # warning: 未实现cache_labels
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total

        if exists and comm.get_local_rank() in (-1, 0):
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(
                f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly."
            )
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        return labels


if __name__ == "__main__":
    dataset = YOLODataset("/data2/hzh/DSEC-YOLO/train", "train")
    item = dataset.__getitem__(0)
    pass
