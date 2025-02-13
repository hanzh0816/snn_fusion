import torch
import torch.nn.functional as F
from torch import nn

from core.structures.instances import Instances

from .loss_utils import (TaskAlignedAssigner, bbox2dist, bbox_iou, dist2bbox,
                         make_anchors)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = (
                self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask])
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)



class v8DetectionLoss(nn.Module):
    """
    Criterion class for computing training losses
    """

    def __init__(
        self, reg_max, num_classes, num_out_channels, stride, box_weight, cls_weight, dfl_weight
    ):
        super().__init__()
        self.reg_max = reg_max
        self.num_classes = num_classes
        self.num_out_channels = num_out_channels
        self.stride = stride
        self.loss_weight = torch.tensor([box_weight, cls_weight, dfl_weight])

        self.use_dfl = self.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.num_classes, alpha=0.5, beta=6.0
        )
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl)
        self.proj = torch.arange(self.reg_max, dtype=torch.float)

    def forward(self, preds, target):
        device = preds[0].device
        self.proj = self.proj.to(device)
        loss = torch.zeros(3, device=device)

        feats = preds[1] if isinstance(preds, tuple) else preds
        batch_size = feats[0].shape[0]
        pred_distri, pred_scores = torch.cat(
            [feat.view(batch_size, self.num_out_channels, -1) for feat in feats], 2
        ).split(
            (self.reg_max * 4, self.num_classes), 1
        )  # [bs, reg_max * 4, mlvl_h*w] [bs, 80, mlvl_h*w]

        # [bs, reg_max * 4, mlvl_h*w] -> [bs, mlvl_h*w, 80]
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        # [bs, 80, mlvl_h*w] -> [bs, mlvl_h*w, 80]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        ori_shape = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        gt_labels, gt_bboxes = self.preprocess(target, batch_size, device)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self._bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        loss[1] = (
            self.bce_loss(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )  # BCE 类别损失

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.loss_weight[0]  # box gain
        loss[1] *= self.loss_weight[1]  # cls gain
        loss[2] *= self.loss_weight[2]  # dfl gain

        return {
            "loss": loss.sum() * batch_size,
            "box_loss": loss[0].detach() * batch_size,
            "cls_loss": loss[1].detach() * batch_size,
            "dfl_loss": loss[2].detach() * batch_size,
        }

    def preprocess(self, target: list["Instances"], batch_size, device):
        counts_max = max([len(k.gt_bboxes) for k in target])
        out = torch.zeros(batch_size, counts_max, 5, device=device)

        for i in range(batch_size):
            if len(target[i].gt_bboxes) == 0:
                continue
            out[i, : len(target[i].gt_bboxes)] = torch.cat(
                (target[i].gt_classes[:, None], target[i].gt_bboxes.tensor), 1
            )
        return out.split((1, 4), 2)

    def _bbox_decode(self, anchor_points, pred_dist):  # 11111
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            )
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)
