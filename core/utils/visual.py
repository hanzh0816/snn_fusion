import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch


def plot_events_map(events_map, instances=None, save_path="events_map.png"):
    """
    将 events_map 逐时间步绘制在一张大图上，并保存为图像文件。
    如果提供了标注信息，会在图像上绘制目标框。

    参数:
        events_map (torch.Tensor): 四维张量，形状为 (num_time_bins, C, H, W)。
        instances (Instances): 包含目标框和类别标签的对象。
        save_path (str): 保存图像的路径，默认为 "events_map.png"。
    """
    # 将 PyTorch 张量转换为 NumPy 数组
    events_map = events_map.numpy()

    # 获取张量的形状
    num_time_bins, num_channels, height, width = events_map.shape

    # 根据通道数确定子图的行数和列数
    if num_channels == 2:
        # 每个时间步绘制两个子图（正事件和负事件）
        fig, axes = plt.subplots(num_time_bins, 2, figsize=(10, 5 * num_time_bins))
        fig.suptitle("Events Map (Positive and Negative Channels)", fontsize=16)
    elif num_channels == 3:
        # 每个时间步绘制一个子图（通道 0）
        fig, axes = plt.subplots(num_time_bins, 1, figsize=(10, 5 * num_time_bins))
        fig.suptitle("Events Map (Single Channel)", fontsize=16)
    else:
        raise ValueError("Unsupported number of channels. Expected 2 or 3.")

    # 遍历每个时间步
    for t in range(num_time_bins):
        if num_channels == 2:
            # 绘制正事件通道
            ax = axes[t, 0] if num_time_bins > 1 else axes[0]
            ax.imshow(events_map[t, 0], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"Time Bin {t} - Positive Events")
            ax.axis("off")

            # 如果传入了标注信息，绘制目标框
            if instances is not None:
                plot_boxes(ax, instances)

            # 绘制负事件通道
            ax = axes[t, 1] if num_time_bins > 1 else axes[1]
            ax.imshow(events_map[t, 1], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"Time Bin {t} - Negative Events")
            ax.axis("off")

            # 如果传入了标注信息，绘制目标框
            if instances is not None:
                plot_boxes(ax, instances)

        elif num_channels == 3:
            # 绘制通道 0
            ax = axes[t] if num_time_bins > 1 else axes
            ax.imshow(events_map[t, 0], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"Time Bin {t}")
            ax.axis("off")

            # 如果传入了标注信息，绘制目标框
            if instances is not None:
                plot_boxes(ax, instances)

    # 调整子图间距
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Events map saved to {save_path}")


def plot_boxes(ax, instances):
    bboxes = instances.gt_bboxes.tensor.numpy()  # 形状为 (N, 4)
    classes = instances.gt_classes.numpy()  # 形状为 (N,)

    for bbox, cls in zip(bboxes, classes):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # 绘制矩形框
        rect = patches.Rectangle(
            (x1, y1), width, height, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

        # # 添加类别标签
        # ax.text(x1, y1 - 5, f"Class {cls}", color="r", fontsize=8, backgroundcolor="white")
