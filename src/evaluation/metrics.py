from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

@dataclass
class DetectionErrors:
    per_image: pd.DataFrame
    per_class: pd.DataFrame
    matched_gt_ids: set[int]
    missed_gt_ids: set[int]
    false_positive_ids: set[int]
    reviewed_prediction_ids: set[int]
    

def evaluate_coco(ground_truth: dict, predictions: list[dict]) -> COCOeval:
    if not ground_truth["images"] or not ground_truth["categories"]:
        raise ValueError("COCO evaluation needs images and categories")
    
    gt = COCO()
    gt.dataset = deepcopy(ground_truth)
    gt.dataset.setdefault("info", {})
    gt.createIndex()
    
    unknown = {item["category_id"] for item in predictions} - set(gt.cats)
    if unknown:
        raise ValueError(f"Unknown prediction category IDs: {sorted(unknown)}")
    
    if predictions:
        dt = gt.loadRes(deepcopy(predictions))
    else:
        dt = COCO()
        dt.dataset = {
            "images": gt.dataset["images"],
            "categories": gt.dataset["categories"], "annotations": []
        }
        dt.createIndex()
        
    
    evaluator = COCOeval(gt, dt, "bbox")
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    return evaluator


def _valid_mean(values: np.ndarray) -> float:
    values = values[values >= 0]
    return float(values.mean()) if values.size else np.nan


def coco_results(evaluator: COCOeval) -> pd.DataFrame:
    area = evaluator.params.areaRngLbl.index("all")
    threshold = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.5))[0])
    rows = []
    
    