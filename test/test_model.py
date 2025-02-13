import cv2
from sympy import I
import torch
from core.config.config import get_cfg
from core.data.build import _get_detection_dataset_dicts, build_detection_train_loader
from core.data.catalog import DatasetCatalog
from core.data.dataset_mapper import DatasetMapper, MapDataset
import core.data.datasets
from core.data.datasets.dsec.dsec import dsec_collate_fn
from core.evaluation.build import build_evaluator
from core.evaluation.dsec_eval import DSECEvaluator
from core.model import build_model


# dataset = DatasetCatalog.get("dsec_train_debug")
# from tqdm import tqdm  # 导入tqdm库
# for i in tqdm(range(100), desc="Processing images"):  # 使用tqdm包装器
#     output = dataset.__getitem__(i+1000)
#     cv2.imwrite(f'./temp/output{i}.jpg',output['debug'])
#     cv2.imwrite(f'./temp/event{i}.jpg',output['event_debug'])


# dataset = _get_detection_dataset_dicts("dsec_val")

# mapper = DatasetMapper(is_train=True,modality='fusion')
# dataset = MapDataset(dataset, mapper)
# Dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=4, shuffle=True, collate_fn=dsec_collate_fn
# )


cfg = get_cfg()
cfg.merge_from_file("configs/spike_yolo/spike_yolo_nano.yaml")


model = build_model(cfg=cfg.MODEL, name=cfg.MODEL.NAME)

checkpoint = torch.load(
    "/home/hzh/code/rgb_event_fusion/snn_fusion/work_dirs/exp_n_lr1e-3_warmup4_mosaic_ep100_bs40_c3_bn_2025-02-11_16-15/ckpt/cfg.MODEL.NAME=0-epoch=21-val_loss=19.96.ckpt"
)
filtered_weights = {
    k: v for k, v in checkpoint["state_dict"].items() if not k.startswith("model.head")
}
model.load_state_dict(filtered_weights, strict=False)
print(model.backbone.layers[0].encode_conv[0][0].weight)


# evaluator: DSECEvaluator = build_evaluator(cfg)

# Dataloader = build_detection_train_loader(cfg)

# for batch in Dataloader:
#     x = batch
#     break
# print(x)
# loss = model.loss(x)
# print(loss)

# with torch.no_grad():
#     preds = model.predict(x)

# evaluator.process(x, preds)
# result = evaluator.evaluate()
# print(result)

# with torch.no_grad():
#     preds = model.predict(x)
# pass
# evaluator.process(x, preds)
# result = evaluator.evaluate()
# print(result)
