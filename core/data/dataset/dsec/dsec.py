import multiprocessing as mp
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .dsec_utils import *

CLASSES = ("pedestrian", "rider", "car", "bus", "truck", "bicycle", "motorcycle", "train")
COLORS = np.array(sns.color_palette("hls", len(CLASSES)))
DSEC_CATAGORIES = {"thing_classes": ["car", "pedestrian"]}


class BaseDirectory:
    def __init__(self, root):
        self.root = root


class DSECDirectory:
    def __init__(self, root):
        self.root = root
        self.images = ImageDirectory(root / "images")
        self.events = EventDirectory(root / "events")
        self.tracks = TracksDirectory(root / "object_detections")


class ImageDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def timestamps(self):
        return np.genfromtxt(self.root / "timestamps.txt", dtype="int64")

    @property
    @lru_cache(maxsize=1)
    def image_files_rectified(self):
        return sorted(list((self.root / "left/rectified").glob("*.png")))

    @property
    @lru_cache(maxsize=1)
    def image_files_distorted(self):
        return sorted(list((self.root / "left/distorted").glob("*.png")))


# todo: 添加降采样event_2x.h5的路径
class EventDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def event_file(self):
        return self.root / "left/events.h5"


class TracksDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def tracks(self):
        return np.load(self.root / "left/tracks.npy")


class DSECDet:
    def __init__(
        self,
        root: Path,
        split: str = "train",
        sync: str = "front",
        split_config=None,
    ):
        """
        root: Root to the the DSEC dataset (the one that contains 'train' and 'test'
        split: Can be one of ['train', 'test']
        window_size: Number of microseconds of data
        sync: Can be either 'front' (last event ts), or 'back' (first event ts). Whether the front of the window or
              the back of the window is synced with the images.

        Each sample of this dataset loads one image, events, and labels at a timestamp. The behavior is different for
        sync='front' and sync='back', and these are visualized below.

        Legend:
        . = events
        | = image
        L = label

        sync='front'
        -------> time
        .......|
               L

        sync='back'
        -------> time
        |.......
               L

        """
        assert root.exists()
        assert split in ["train", "test", "val"]
        assert sync in ["front", "back"]

        self.classes = CLASSES

        self.root = root / ("train" if split in ["train", "val"] else "test")
        self.sync = sync

        self.height = 480
        self.width = 640

        self.directories = dict()
        self.img_idx_track_idxs = dict()

        if split_config is None:
            self.subsequence_directories = list(self.root.glob("*/"))
        else:
            available_dirs = list(self.root.glob("*/"))
            self.subsequence_directories = [
                self.root / s for s in split_config[split] if self.root / s in available_dirs
            ]

        self.subsequence_directories = sorted(
            self.subsequence_directories, key=self.first_time_from_subsequence
        )

        for f in self.subsequence_directories:
            directory = DSECDirectory(f)
            self.directories[f.name] = directory
            self.img_idx_track_idxs[f.name] = _compute_img_idx_to_track_idx(
                directory.tracks.tracks["t"], directory.images.timestamps
            )

    def first_time_from_subsequence(self, subsequence):
        return np.genfromtxt(subsequence / "images/timestamps.txt", dtype="int64")[0]

    def get_index_window(self, index, num_idx, sync="back"):
        if sync == "front":
            assert 0 < index < num_idx
            i_0 = index - 1
            i_1 = index
        else:
            assert 0 <= index < num_idx - 1
            i_0 = index
            i_1 = index + 1

        return i_0, i_1

    def get_tracks(self, index, mask=None, directory_name=None):

        idx, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        idx0, idx1 = img_idx_to_track_idx[idx]
        tracks = directory.tracks.tracks[idx0:idx1]

        if mask is not None:
            tracks = tracks[mask[idx0:idx1]]

        return tracks

    def get_events(self, index, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        # tag: event对应的index保证不是当前序列的最后一个index
        i_0, i_1 = self.get_index_window(index, len(img_idx_to_track_idx), sync=self.sync)
        t_0, t_1 = directory.images.timestamps[[i_0, i_1]]
        events = _extract_from_h5_by_timewindow(directory.events.event_file, t_0, t_1)
        return events

    def get_image(self, index, directory_name=None):
        index, _, directory = self.rel_index(index, directory_name)
        image_files = directory.images.image_files_distorted
        image = cv2.imread(str(image_files[index]))
        return image

    def rel_index(self, index, directory_name=None):
        assert directory_name is not None, "method `rel_index` must specify a directory name"

        img_idx_to_track_idx = self.img_idx_track_idxs[directory_name]
        directory = self.directories[directory_name]
        return index, img_idx_to_track_idx, directory


class DSEC(Dataset):
    MAPPING = dict(
        pedestrian="pedestrian",
        rider=None,
        car="car",
        bus="car",
        truck="car",
        bicycle=None,
        motorcycle=None,
        train=None,
    )

    def __init__(self, root: Path, split: str, *, debug: bool = False, split_config: dict = None):
        split_config = _yaml_file_to_dict(Path(__file__).parent / "dsec_split.yaml")
        assert split in split_config.keys(), f"'{split}' not in {list(split_config.keys())}"
        self.split = split
        self.dataset = DSECDet(root=root, split=split, sync="back", split_config=split_config)
        self.classes = ("car", "pedestrian")
        self.time_window = 1000000
        self.num_us = -1
        self.debug = debug

        self.class_mapping = _compute_class_mapping(
            classes=self.classes, all_classes=self.dataset.classes, mapping=self.MAPPING
        )

        self.image_index_pairs, self.track_masks = _filter_tracks(
            dataset=self.dataset,
            image_width=self.dataset.width,
            image_height=self.dataset.height,
            class_remapping=self.class_mapping,
            min_bbox_height=0,
            min_bbox_diag=0,
        )

        self.cache_root = root / ".cache" / split

    def set_num_us(self, num_us):
        self.num_us = num_us

    def __len__(self):
        return sum(len(d) for d in self.image_index_pairs.values())

    def _cache_data(self):
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # 使用多进程来加速缓存过程
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # 使用 Pool.map 并行处理缓存任务
            indices = range(len(self))
            list(
                tqdm(
                    pool.imap(self._cache_sample, indices),
                    total=len(indices),
                    desc=f"Caching {self.split} data",
                    unit="sample",
                )
            )

    def _cache_sample(self, index):
        image_0, image_ts_0, events, detections_0, detections_1 = self._get_data(index)
        # tag: 存储events_map而非events,能够节省load_data时间
        events_map = self._preprocess_events_map(self._preprocess_events(events))
        cache_file = self.cache_root / f"{index}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "image": image_0,
                    "events": events_map,
                    "image_ts_0": image_ts_0,
                    "ori_shape": (self.dataset.height, self.dataset.width),
                    "annotations": detections_0,
                    "annotations_next": detections_1,
                },
                f,
            )

    def __getitem__(self, index):
        if self.debug:
            return self._debug_get_item(index)

        if not self.cache_root.exists():
            return self._no_cache_get_item(index)
            # self._cache_data()

        cache_file = self.cache_root / f"{index}.pkl"
        with open(cache_file, "rb") as f:
            data = pickle.load(f)

        # todo: 截取特定时间范围内的事件流
        # events = data["events"]
        # image_ts_0 = data["image_ts_0"]
        # if self.num_us >= 0:
        #     image_ts_1 = image_ts_0 + self.num_us
        #     events = {k: v[events["t"] < image_ts_1] for k, v in events.items()}

        return data

    def _get_data(self, index):
        dataset, image_index_pairs, track_masks, idx = self._rel_index(index)
        image_index_0, image_index_1 = image_index_pairs[idx]
        image_ts_0, image_ts_1 = dataset.images.timestamps[[image_index_0, image_index_1]]
        detections_0 = self.dataset.get_tracks(
            image_index_0, mask=track_masks, directory_name=dataset.root.name
        )
        detections_1 = self.dataset.get_tracks(
            image_index_1, mask=track_masks, directory_name=dataset.root.name
        )

        detections_0 = self._preprocess_detections(detections_0)
        detections_1 = self._preprocess_detections(detections_1)

        image_0 = self.dataset.get_image(image_index_0, directory_name=dataset.root.name)
        events = self.dataset.get_events(image_index_0, directory_name=dataset.root.name)

        return image_0, image_ts_0, events, detections_0, detections_1

    def _debug_get_item(self, index):
        image_0, image_ts_0, events, detections_0, detections_1 = self._get_data(index)
        data = {
            "image": image_0,
            "events": events,
            "image_ts_0": image_ts_0,
            "ori_shape": (self.dataset.height, self.dataset.width),
            "annotations": detections_0,
            "annotations_next": detections_1,
        }

        image = (255 * (data["image"].astype("float32") / 255) ** (1 / 2.2)).astype("uint8")
        empty_image = np.zeros_like(image)
        data["debug"] = _render_events_on_image(image, x=events["x"], y=events["y"], p=events["p"])
        data["debug"] = _render_object_detections_on_image(data["debug"], data["annotations"])
        data["event_debug"] = _render_events_on_image(
            empty_image, x=events["x"], y=events["y"], p=events["p"]
        )
        data["event_debug"] = _render_object_detections_on_image(
            data["event_debug"], data["annotations"]
        )
        events = self._preprocess_events(events)
        events = self._preprocess_events_map(events)  # to events map
        data["events"] = events
        return data

    def _no_cache_get_item(self, index):
        image_0, image_ts_0, events, detections_0, detections_1 = self._get_data(index)
        events = self._preprocess_events(events)
        events = self._preprocess_events_map(events)  # to events map
        data = {
            "image": image_0,
            "events": events,
            "image_ts_0": image_ts_0,
            "ori_shape": (self.dataset.height, self.dataset.width),
            "annotations": detections_0,
            "annotations_next": detections_1,
        }
        return data

    def _preprocess_detections(self, detections):
        detections = _crop_tracks(detections, self.dataset.width, self.dataset.height)
        detections["class_id"], _ = _map_classes(detections["class_id"], self.class_mapping)
        return detections

    def _preprocess_events(self, events):
        mask = (
            (0 < events["y"])
            & (events["y"] < self.dataset.height)
            & (0 < events["x"])
            & (events["x"] < self.dataset.width)
        )  # 过滤越界的events
        events = {k: v[mask] for k, v in events.items()}
        if len(events["t"]) > 0:
            # tag: 将时间戳转换为相对时间,这里的转换公式存疑
            events["t"] = self.time_window + events["t"] - events["t"][-1]
        # 极性由0,1变为-1,1
        events["p"] = 2 * events["p"].reshape((-1, 1)).astype("int8") - 1
        return events

    def _preprocess_events_map(self, events):
        t = np.array(events["t"])
        x = np.array(events["x"])
        y = np.array(events["y"])
        p = np.array(events["p"])

        num_time_bins = 4
        ori_shape = (480, 640)

        # 归一化时间戳到 [0, num_time_bins)
        t_min, t_max = t.min(), t.max()
        t_bins = np.floor((t - t_min) / (t_max - t_min) * num_time_bins).astype(int)
        t_bins[t_bins == num_time_bins] = num_time_bins - 1
        # 初始化张量：T 时间步, C=2 极性, H 高度, W 宽度

        height, width = ori_shape
        # 压帧:3通道灰度图处理
        events_map = 127 * np.ones((num_time_bins, 3, height, width), dtype=np.float32)

        for t_i, x_i, y_i, p_i in zip(t_bins, x, y, p):
            if p_i == 1:  # 正极性
                events_map[t_i, :, y_i, x_i] = [255, 255, 255]

        events_map = torch.Tensor(events_map)
        return events_map

    def _rel_index(self, idx):
        for folder in self.dataset.subsequence_directories:
            name = folder.name
            image_index_pairs = self.image_index_pairs[name]
            directory = self.dataset.directories[name]
            track_mask = self.track_masks[name]
            if idx < len(image_index_pairs):
                return directory, image_index_pairs, track_mask, idx
            idx -= len(image_index_pairs)
        raise IndexError


def dsec_collate_fn(batch):
    images = []
    events = []
    instances = []
    for data in batch:
        if data.get("image") is not None:
            images.append(data["image"])
        if data.get("events") is not None:
            events.append(data["events"])
        instances.append(data["instances"])
    if len(images):
        images = torch.stack(images, dim=0)
    if len(events):
        events = torch.stack(events, dim=0)
        events = events.transpose(0, 1).contiguous()  # [B,T,C,H,W] -> [T,B,C,H,W]

    return {
        "image": images,
        "events": events,
        "instances": instances,
    }
