from doctest import debug
import logging
import os
from pathlib import Path

from core.data.catalog import DatasetCatalog, MetadataCatalog
from core.data.datasets.dsec.dsec import DSEC, DSEC_CATAGORIES


def register_dsec(root, split, *args, **kwargs):
    return DSEC(root, split, *args, **kwargs)


def register_all_dsec(root):
    root = Path(os.path.join(root, "DSEC"))
    for name, split in [("train", "train"), ("val", "val"), ("test", "test")]:
        name = f"dsec_{name}"
        DatasetCatalog.register(name, lambda x=root, y=split: register_dsec(x, y))
        MetadataCatalog.get(name).set(**DSEC_CATAGORIES, evaluator_type="dsec_coco_instance")

    name, split = "dsec_train_debug", "train"
    DatasetCatalog.register(name, lambda x=root, y=split: register_dsec(x, y, debug=True))


if __name__.endswith(".builtin"):
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_all_dsec(_root)
    logging.getLogger(__name__).info("Registered all DSEC datasets")
