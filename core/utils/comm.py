# Copyright (c) Facebook, Inc. and its affiliates.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import functools
import os
import numpy as np
import torch
import torch.distributed as dist


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def gather(data):
    # 获取当前进程的rank和总进程数
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 初始化汇总列表（仅在rank=0）
    gathered_list = [None] * world_size if rank == 0 else None

    # 使用gather_object收集数据到rank=0
    dist.gather_object(obj=data, object_gather_list=gathered_list, dst=0)

    # rank=0处理并返回合并后的结果，其他进程返回None
    if rank == 0:
        all_data = []
        for sublist in gathered_list:
            all_data.extend(sublist)
        return all_data
    else:
        return None


def init_process_group():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
