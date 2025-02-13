from core.config.config import get_cfg
from core.data.build import _get_detection_dataset_dicts


cfg = get_cfg()
cfg.merge_from_file("configs/spike_yolo/spike_yolo_nano.yaml")

dataset = _get_detection_dataset_dicts(cfg.DATASETS.VAL)

dataset._cache_data()