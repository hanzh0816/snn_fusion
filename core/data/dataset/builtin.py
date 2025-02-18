from doctest import debug
import logging
import os
from pathlib import Path

from core.data.catalog import DatasetCatalog, MetadataCatalog
from core.data.dataset.dsec.dsec import DSEC, DSEC_CATAGORIES
from core.data.dataset.dsec_yolo.dsec_yolo import YOLODataset


def register_dsec(root, split, *args, **kwargs):
    return DSEC(root, split, *args, **kwargs)


def register_dsec_yolo(img_path, prefix, *args, **kwargs):
    return YOLODataset(img_path, prefix, *args, **kwargs)


def register_all_dsec(root):
    root = Path(os.path.join(root, "DSEC"))
    for name, split in [("train", "train"), ("val", "val"), ("test", "test")]:
        name = f"dsec_{name}"
        DatasetCatalog.register(name, lambda x=root, y=split: register_dsec(x, y))
        MetadataCatalog.get(name).set(**DSEC_CATAGORIES, evaluator_type="dsec_coco_instance")

    name, split = "dsec_train_debug", "train"
    DatasetCatalog.register(name, lambda x=root, y=split: register_dsec(x, y, debug=True))


def register_all_dsec_yolo(root):
    root = Path(os.path.join(root, "DSEC-YOLO"))
    for name, prefix in [("train", "train"), ("val", "test")]:
        img_path = root / prefix
        name = f"dsec_yolo_{name}"
        DatasetCatalog.register(name, lambda x=img_path, y=prefix: register_dsec_yolo(x, y))
        MetadataCatalog.get(name).set(**DSEC_CATAGORIES, evaluator_type="dsec_coco_instance")


def register_all():
    _root = os.path.expanduser(os.getenv("DATASETS_ROOT", "datasets"))
    register_all_dsec(_root)
    logging.getLogger(__name__).info("Registered all DSEC datasets")

    register_all_dsec_yolo(_root)
    logging.getLogger(__name__).info("Registered all DSEC-YOLO datasets")


if __name__.endswith(".builtin"):
    register_all()
