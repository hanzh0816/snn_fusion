import cv2
import numpy as np
import torch

from core.structures.boxes import Boxes
from core.structures.instances import Instances


def preprocess_events(events, ori_shape, num_time_bins):
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


def preprocess_annotations(dataset_dict):
    # todo: detection0/1 转换为annotations序列
    dataset_dict.pop("annotations_next", None)

    # 已经是Instances类型,已经是预处理过的
    if dataset_dict.get("instances", None) is not None:
        return dataset_dict

    ori_annotations = dataset_dict.pop("annotations")
    # 一个样本中的所有bbox
    bboxes = [preprocess_bbox(obj) for obj in ori_annotations]

    target = Instances(
        ori_image_size=dataset_dict["ori_shape"], sample_idx=dataset_dict["sample_idx"]
    )
    target.gt_bboxes = Boxes(bboxes)

    classes = [int(obj["class_id"]) for obj in ori_annotations]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    dataset_dict["instances"] = target
    return dataset_dict


def preprocess_bbox(bbox):
    bbox = [bbox["x"], bbox["y"], bbox["w"], bbox["h"]]

    # xywh -> xyxy_abs
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]

    return bbox


def draw_bbox(img, instances):
    bboxes = instances.gt_bboxes.tensor.cpu().numpy()
    for box in bboxes:
        x1, y1, x2, y2 = box
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return img


def display_events(events):
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
