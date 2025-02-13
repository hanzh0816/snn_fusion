import itertools
import logging

import torch
import torch.utils.data as torchdata

from core.config.config import configurable
from core.data.catalog import DatasetCatalog, MetadataCatalog
from core.data.dataset_mapper import DatasetMapper, MapDataset
from core.data.datasets.dsec.dsec import dsec_collate_fn


def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = _get_detection_dataset_dicts(cfg.DATASETS.TRAIN)

    mapper = DatasetMapper(cfg=cfg, is_train=True, dataset=dataset)
    # todo: 构建sampler

    return {
        "dataset": dataset,
        "sampler": None,
        "mapper": mapper,
        "batch_size": cfg.DATALOADER.BATCH_SIZE,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "collate_fn": dsec_collate_fn,
    }


def _val_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = _get_detection_dataset_dicts(cfg.DATASETS.VAL)

    mapper = DatasetMapper(cfg=cfg, is_train=False, dataset=dataset)
    # todo: 构建sampler

    return {
        "dataset": dataset,
        "sampler": None,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "batch_size": min(4, cfg.DATALOADER.BATCH_SIZE),  # tag: 测试时batch_size为4
        "collate_fn": dsec_collate_fn,
    }


def _get_detection_dataset_dicts(names):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names

    available_datasets = DatasetCatalog.keys()
    names_set = set(names)
    if not names_set.issubset(available_datasets):
        logger = logging.getLogger(__name__)
        logger.warning(
            "The following dataset names are not registered in the DatasetCatalog: "
            f"{names_set - available_datasets}. "
            f"Available datasets are {available_datasets}"
        )

    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]

    # 注册器返回的是torchdata.Dataset对象
    if isinstance(dataset_dicts[0], torchdata.Dataset):
        if len(dataset_dicts) > 1:
            # ConcatDataset does not work for iterable style dataset.
            # We could support concat for iterable as well, but it's often
            # not a good idea to concat iterables anyway.
            return torchdata.ConcatDataset(dataset_dicts)
        return dataset_dicts[0]

    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    # 将dataset_dicts中的所有dicts合并成一个list
    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    # todo: 检查dataset元数据一致性(class label consistency)

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts


@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    batch_size,
    num_workers=0,
    collate_fn=None,
    **kwargs,
):
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    # todo: 实现自定义的Dataloader
    # tag: 这里的batch_size是每个worker的batch_size，不是整个batch_size
    return torchdata.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )


@configurable(from_config=_val_loader_from_config)
def build_detection_val_loader(
    dataset, *, mapper, sampler=None, num_workers=1, batch_size=1, collate_fn=None, **kwargs
):

    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    # todo: 实现自定义的Dataloader
    return torchdata.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
