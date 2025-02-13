import torch
from core.config.config import get_cfg
from core.data.build import _get_detection_dataset_dicts, build_detection_train_loader
from core.data.dataset_mapper import DatasetMapper, MapDataset
from core.model.module.snn_module import SpikePreprocess
from core.utils.visual import plot_events_map


cfg = get_cfg()
cfg.merge_from_file("configs/spike_yolo/spike_yolo_nano.yaml")
cfg.DATALOADER.BATCH_SIZE = 1

Dataloader = build_detection_train_loader(cfg)
print(Dataloader.batch_size)

model = SpikePreprocess(spike_t=4)

for item in Dataloader:
    events = item["events"]
    events_0 = events[:, 0, ...]
    plot_events_map(events_0, item["instances"][0], "events_0.png")
    # new_events = model(events)
    # new_events_0 = new_events[:, 0, ...]
    # plot_events_map(new_events_0, "new_events_0.png")
    break


# tensor = item["events"]
# # 找到 tensor 中的非零值
# non_zero_values = tensor[tensor != 0]

# # 计算非零值的均值
# mean_non_zero = non_zero_values.mean() if non_zero_values.numel() > 0 else torch.tensor(0.0)

# print("Non-zero mean:", mean_non_zero)
