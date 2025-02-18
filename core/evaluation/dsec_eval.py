import copy
import gc
import itertools
import logging
import time

from matplotlib.pylab import annotations, f
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
import torch

from core.data.catalog import MetadataCatalog
from core.structures.boxes import BoxMode
from core.structures.instances import Instances
import core.utils.comm as comm
from core.utils.logger import create_small_table


from .evaluator import DatasetEvaluator

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval


class DSECEvaluator(DatasetEvaluator):
    # todo: 目前实现的是同步检测的evaluator，后续需要改成同步/异步兼容模式
    def __init__(
        self,
        dataset_name,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
    ):
        self._logger = logging.getLogger(__name__)
        self._distributed = True
        self._cpu_device = torch.device("cpu")
        # self._distributed = distributed
        if use_fast_impl and (COCOeval_opt is COCOeval):
            self._logger.info("Fast COCO eval is not built. Falling back to official COCO eval.")
            use_fast_impl = False
        self._use_fast_impl = use_fast_impl
        self._metadata = MetadataCatalog.get(dataset_name)

        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]
        else:
            max_dets_per_image = [1, 10, max_dets_per_image]
        self._max_dets_per_image = max_dets_per_image

        # bbox detection only
        self._task = "bbox"
        # empty coco API, when call method ``evaluate()`` it will pass results to coco API
        self._coco_api = COCO()
        self._predictions = []

    def reset(self):
        self._coco_api = COCO()
        self._predictions = []

    def process(self, inputs, outputs):
        assert "instances" in outputs, "outputs must contain the key 'instances' "
        for idx in range(len(inputs["instances"])):
            gt_instances = inputs["instances"][idx]  # type: Instances
            pred_instances = outputs["instances"][idx].clone().to(self._cpu_device)

            img, gts, preds = self._instances_to_coco_json(gt_instances, pred_instances)

            prediction = {"image": img, "preds": preds, "gts": gts}
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions)
            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[DSECEvaluator] Did not receive valid predictions.")
            return {}

        # 转为coco_dataset json格式和result json格式
        (dataset, results), img_ids = self._convert_to_coco_format(
            predictions, class_names=self._metadata.get("thing_classes")
        )
        self._coco_api.dataset = dataset
        self._coco_api.createIndex()

        coco_eval = self._evaluate_predictions_on_coco(
            coco_gt=self._coco_api,
            coco_results=results,
            iou_type=self._task,
            cocoeval_fn=COCOeval_opt if self._use_fast_impl else COCOeval,
            img_ids=img_ids,
            max_dets_per_image=self._max_dets_per_image,
        )
        # 调用derive函数处理

        res = self._derive_coco_results(
            coco_eval, self._task, class_names=self._metadata.get("thing_classes")
        )
        return res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warning("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

    def _convert_to_coco_format(
        self, predictions, class_names: str = ("car", "pedestrian"), height=480, width=640
    ):
        # tag: eval resize之后此处需要传入图像尺寸
        # 将模型输出转换为coco格式
        images = []
        annotations = []
        results = []
        img_ids = []

        ann_id = 0
        for prediction in predictions:
            images.append(prediction["image"])
            img_ids.append(prediction["image"]["id"])

            gts = prediction["gts"]
            preds = prediction["preds"]
            for gt in gts:
                gt["id"] = ann_id
                ann_id += 1
                annotations.append(gt)
            for pred in preds:
                results.append(pred)

        num_det = len(results)
        if num_det == 0:
            self._logger.warning("[DSECEvaluator] No detection for evaluation found.")

        categories = [
            {"id": id + 1, "name": class_name, "supercategory": "none"}
            for id, class_name in enumerate(class_names)
        ]

        dataset = {
            "info": {},
            "licenses": [],
            "type": "instances",
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        return (dataset, results), img_ids

    @staticmethod
    def _instances_to_coco_json(gt_instances: Instances, pred_instances: torch.Tensor):

        sample_idx = gt_instances._sample_idx
        ori_shape = gt_instances.image_size
        height, width = ori_shape

        img = {
            "date_captured": "2025",
            "file_name": "n.a",
            "id": sample_idx,
            "license": 1,
            "url": "",
            "height": height,
            "width": width,
        }
        num_instance = len(gt_instances)
        boxes = gt_instances.gt_bboxes.tensor.clone().cpu().numpy()
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        classes = gt_instances.gt_classes.tolist()

        gts = []
        for k in range(num_instance):
            gt = {
                "image_id": sample_idx,
                "category_id": int(classes[k]) + 1,
                "bbox": boxes[k],
                "area": float(boxes[k][2] * boxes[k][3]),  # w*h
                "iscrowd": False,
                "id": None,
            }
            gts.append(gt)

        preds = []
        for pred_result in pred_instances:
            x1, y1, x2, y2, confidence, class_id = pred_result
            w, h = int(x2 - x1), int(y2 - y1)
            image_result = {
                "image_id": sample_idx,
                "category_id": int(class_id) + 1,
                "score": float(confidence),
                "bbox": [float(x1), float(y1), float(w), float(h)],
            }
            preds.append(image_result)

        return img, gts, preds

    @staticmethod
    def _evaluate_predictions_on_coco(
        coco_gt,
        coco_results,
        iou_type,
        cocoeval_fn=COCOeval_opt,
        img_ids=None,
        max_dets_per_image=None,
    ):
        """
        Evaluate the coco results using COCOEval API.
        """
        assert len(coco_results) > 0

        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = cocoeval_fn(coco_gt, coco_dt, iou_type)
        # For COCO, the default max_dets_per_image is [1, 10, 100].
        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]  # Default from COCOEval
        else:
            assert (
                len(max_dets_per_image) >= 3
            ), "COCOeval requires maxDets (and max_dets_per_image) to have length at least 3"
            # In the case that user supplies a custom input for max_dets_per_image,
            # apply COCOevalMaxDets to evaluate AP with the custom input.

        if iou_type != "keypoints":
            coco_eval.params.maxDets = max_dets_per_image

        if img_ids is not None:
            coco_eval.params.imgIds = img_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval
