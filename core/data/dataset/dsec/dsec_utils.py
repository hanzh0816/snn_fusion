import math
from pathlib import Path

import cv2
import h5py
import numpy as np
import seaborn as sns
import yaml

__all__ = [
    "_crop_tracks",
    "_filter_tracks",
    "_map_classes",
    "_compute_class_mapping",
    "_extract_from_h5_by_index",
    "_extract_from_h5_by_timewindow",
    "_compute_img_idx_to_track_idx",
    "_render_events_on_image",
    "_render_object_detections_on_image",
    "_draw_bbox_on_img",
    "_yaml_file_to_dict",
]

# CLASSES = ("pedestrian", "rider", "car", "bus", "truck", "bicycle", "motorcycle", "train")
CLASSES = ("car", "pedestrian")

COLORS = np.array(sns.color_palette("hls", len(CLASSES)))


def _filter_tracks(
    dataset,
    image_width,
    image_height,
    class_remapping,
    min_bbox_height=0,
    min_bbox_diag=0,
):
    image_index_pairs = {}
    track_masks = {}

    for directory_path in dataset.subsequence_directories:
        tracks = dataset.directories[directory_path.name].tracks.tracks
        image_timestamps = dataset.directories[directory_path.name].images.timestamps

        tracks_rescaled = _crop_tracks(tracks, image_width, image_height)

        _, class_mask = _map_classes(tracks_rescaled["class_id"], class_remapping)
        size_mask = _filter_small_bboxes(
            tracks_rescaled["w"], tracks_rescaled["h"], min_bbox_height, min_bbox_diag
        )
        final_mask = size_mask & class_mask

        # 1. stores indices of images which are valid, i.e. survived all filters above
        valid_image_indices = np.unique(
            np.nonzero(np.isin(image_timestamps, tracks_rescaled[final_mask]["t"]))[0]
        )
        valid_image_index_pairs = _construct_pairs(valid_image_indices, 2)
        # todo: 添加perfect_track过滤

        image_index_pairs[directory_path.name] = valid_image_index_pairs
        track_masks[directory_path.name] = final_mask

    return image_index_pairs, track_masks


def _construct_pairs(indices, n=2):
    indices = np.sort(indices)
    indices = np.stack([indices[i : i + 1 - n] for i in range(n - 1)] + [indices[n - 1 :]])
    mask = np.ones_like(indices[0]) > 0
    for i, row in enumerate(indices):
        mask = mask & (indices[0] + i == row)
    indices = indices[..., mask].T
    return indices


def _filter_small_bboxes(w, h, bbox_height=20, bbox_diag=30):
    """
    Filter out tracks that are too small.
    """
    diag = np.sqrt(h**2 + w**2)
    return (diag > bbox_diag) & (w > bbox_height) & (h > bbox_height)


def _crop_tracks(tracks, width, height):
    tracks = tracks.copy()
    x1, y1 = tracks["x"], tracks["y"]
    x2, y2 = x1 + tracks["w"], y1 + tracks["h"]

    x1 = np.clip(x1, 0, width - 1)
    x2 = np.clip(x2, 0, width - 1)

    y1 = np.clip(y1, 0, height - 1)
    y2 = np.clip(y2, 0, height - 1)

    tracks["x"] = x1
    tracks["y"] = y1
    tracks["w"] = x2 - x1
    tracks["h"] = y2 - y1

    return tracks


def _map_classes(class_ids, old_to_new_mapping):
    new_class_ids = old_to_new_mapping[class_ids]
    mask = new_class_ids > -1
    return new_class_ids, mask


def _compute_class_mapping(classes, all_classes, mapping):
    output_mapping = []
    for i, c in enumerate(all_classes):
        mapped_class = mapping[c]
        output_mapping.append(classes.index(mapped_class) if mapped_class in classes else -1)
    return np.array(output_mapping)


def _extract_from_h5_by_index(filehandle, ev_start_idx: int, ev_end_idx: int):
    events = filehandle["events"]
    x = events["x"]
    y = events["y"]
    p = events["p"]
    t = events["t"]

    x_new = x[ev_start_idx:ev_end_idx]
    y_new = y[ev_start_idx:ev_end_idx]
    p_new = p[ev_start_idx:ev_end_idx]
    t_new = t[ev_start_idx:ev_end_idx].astype("int64") + filehandle["t_offset"][()]

    output = {
        "p": p_new,
        "t": t_new,
        "x": x_new,
        "y": y_new,
    }
    return output


def _extract_from_h5_by_timewindow(h5file, t_min_us: int, t_max_us: int):
    with h5py.File(str(h5file), "r") as h5f:
        ms2idx = np.asarray(h5f["ms_to_idx"], dtype="int64")
        t_offset = h5f["t_offset"][()]

        events = h5f["events"]
        t = events["t"]

        t_ev_start_us = t_min_us - t_offset
        # assert t_ev_start_us >= t[0], (t_ev_start_us, t[0])
        t_ev_start_ms = t_ev_start_us // 1000
        ms2idx_start_idx = t_ev_start_ms
        ev_start_idx = ms2idx[ms2idx_start_idx]

        t_ev_end_us = t_max_us - t_offset
        assert t_ev_end_us <= t[-1], (t_ev_end_us, t[-1])
        t_ev_end_ms = math.floor(t_ev_end_us / 1000)
        ms2idx_end_idx = t_ev_end_ms
        ev_end_idx = ms2idx[ms2idx_end_idx]

        return _extract_from_h5_by_index(h5f, ev_start_idx, ev_end_idx)


def _compute_img_idx_to_track_idx(t_track, t_image):
    x, counts = np.unique(t_track, return_counts=True)
    i, j = (x.reshape((-1, 1)) == t_image.reshape((1, -1))).nonzero()
    deltas = np.zeros_like(t_image)

    deltas[j] = counts[i]

    idx = np.concatenate([np.array([0]), deltas]).cumsum()
    return np.stack([idx[:-1], idx[1:]], axis=-1).astype("uint64")


def _render_events_on_image(image, x, y, p):
    for x_, y_, p_ in zip(x, y, p):
        if p_ == 0:
            image[y_, x_] = np.array([0, 0, 255])
        else:
            image[y_, x_] = np.array([255, 0, 0])
    return image


def _render_object_detections_on_image(img, tracks, **kwargs):
    return _draw_bbox_on_img(
        img, tracks["x"], tracks["y"], tracks["w"], tracks["h"], tracks["class_id"], **kwargs
    )


def _draw_bbox_on_img(
    img, x, y, w, h, labels, scores=None, conf=0.5, label="", scale=1, linewidth=2, show_conf=True
):
    for i in range(len(x)):
        if scores is not None and scores[i] < conf:
            continue

        x0 = int(scale * (x[i]))
        y0 = int(scale * (y[i]))
        x1 = int(scale * (x[i] + w[i]))
        y1 = int(scale * (y[i] + h[i]))
        cls_id = int(labels[i])

        color = (COLORS[cls_id] * 255).astype(np.uint8).tolist()

        # track_id = box['track_id']
        text = f"{label}-{CLASSES[cls_id]}"

        if scores is not None and show_conf:
            text += f":{scores[i] * 100: .1f}"

        txt_color = (0, 0, 0) if np.mean(COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, linewidth)

        txt_bk_color = (COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        txt_height = int(1.5 * txt_size[1])
        cv2.rectangle(img, (x0, y0 - txt_height), (x0 + txt_size[0] + 1, y0 + 1), txt_bk_color, -1)
        cv2.putText(
            img, text, (x0, y0 + txt_size[1] - txt_height), font, 0.4, txt_color, thickness=1
        )
    return img


def _yaml_file_to_dict(yaml_file: Path):
    with yaml_file.open() as fh:
        return yaml.load(fh, Loader=yaml.UnsafeLoader)
