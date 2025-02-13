import os
from core.config.config import configurable
from core.data.catalog import MetadataCatalog
from core.evaluation.dsec_eval import DSECEvaluator


def _build_evaluator_from_config(cfg):
    dataset_name = cfg.DATASETS.TEST
    output_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
    return {"dataset_name": dataset_name, "output_dir": output_dir}


@configurable(from_config=_build_evaluator_from_config)
def build_evaluator(dataset_name, output_dir):
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    # tag: evaluator传入更多参数时需要修改这里的代码
    assert evaluator_type == "dsec_coco_instance", "Only dsec_coco_instance is supported"
    return DSECEvaluator(dataset_name, output_dir)
