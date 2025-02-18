import cv2
from sympy import I
import torch
from core.config.config import get_cfg
from core.data.build import _get_detection_dataset_dicts, build_detection_train_loader
from core.data.catalog import DatasetCatalog
from core.data.dataset_mapper import DatasetMapper, MapDataset
import core.data.dataset
from core.data.dataset.dsec.dsec import dsec_collate_fn
from core.evaluation.build import build_evaluator
from core.evaluation.dsec_eval import DSECEvaluator
from core.model import build_model

cfg = get_cfg()
cfg.merge_from_file("configs/spike_yolo/spike_yolo_ultralytics.yaml")

cfg.DATALOADER.BATCH_SIZE = 1

Dataloader = build_detection_train_loader(cfg)

model = build_model(cfg=cfg.MODEL, name=cfg.MODEL.NAME)
model.to("cuda")

for batch in Dataloader:
    batch["events"] = batch["events"].to("cuda")
    outputs = model.predict(batch)
    print(outputs['loss'])
    break
