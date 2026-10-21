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
    
    for index, category_id in enumerate(evaluator.params.catIds):
        precision = evaluator.eval["precision"][:, :, index, area, -1]
        recall = evaluator.eval["recall"][:, index, area, -1]
        ids = evaluator.cocoGt.getAnnIds(
            imgIds=evaluator.params.imgIds, catIds=[category_id]
        )
        
        rows.append({
            "Class": evaluator.cocoGt.cats[category_id]["name"],
            "Instances": sum(not evaluator.cocoGt.anns[i].get("iscrowd", 0) for i in ids),
            "AP@0.50": _valid_mean(precision[threshold]),
            "AP@[.50:.95]": _valid_mean(precision),
            "AR@[.50:.95]": _valid_mean(recall)
        })
    
    table = pd.DataFrame(rows).set_index("Class")
    table.loc["All Classes"] = {
        "Instances": table["Instances"].sum(),
        **table.drop(columns="Instances").mean().to_dict(),
    }
    return table


def detection_errors(
    evaluator: COCOeval, *,
    confidence: float = 0.25, iou: float = 0.5
) -> DetectionErrors:
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be in [0, 1]")
    thresholds = np.flatnonzero(np.isclose(evaluator.params.iouThrs, iou))
    if not thresholds.size:
        raise ValueError("Review IoU must be one of the evaluated IoU thresholds")
    
    threshold = int(thresholds[0])
    area = evaluator.params.areaRng[evaluator.params.areaRngLbl.index("all")]
    gt = evaluator.cocoGt
    image_rows = {
        i: {"Image": gt.imgs[i].get("file_name", str(i)),
            "Objects": 0, "TP": 0, "FP": 0, "FN": 0}
        for i in evaluator.params.imgIds
    }
    class_rows = {
        i: {"Class": gt.cats[i]["name"], "TP": 0, "FP": 0, "FN": 0}
        for i in evaluator.params.catIds
    }
    
    matched_gt, missed_gt, false_ids, reviewed_ids = set(), set(), set(), set()
    
    for item in evaluator.evalImgs:
        if item in None or item["aRng"] != area:
            continue
        
        keep = (np.asarray(item["dtScores"]) > confidence) & ~item["dtIgnore"][threshold].astype(bool)
        matches = item["dtMatches"][threshold]
        
        prediction_ids = np.asarray(item["dtIds"], dtype=int)
        found = set(matches[keep & (matches > 0)].astype(int))
        false = set(prediction_ids[keep & (matches == 0)])
        
        expected = {
            int(i) for i, ignored in zip(item["gtIds"], item["gtIgnore"]) if not ignored
        }
        
        missed = expected - found
        image_rows[item["image_id"]]["Objects"] += len(expected)
        
        for row in (image_rows[item["image_id"]], class_rows[item["category_id"]]):
            for name, value in (("TP", len(found)), ("FP", len(false)), ("FN", len(missed))):
                row[name] += value
        matched_gt.update(found)
        missed_gt.update(missed)
        false_ids.update(false)
        reviewed_ids.update(prediction_ids[keep])
        
    
    images = pd.DataFrame.from_dict(image_rows, orient="index").rename_axis("Image ID")
    images["Errors"] = images["FP"] + images["FN"]
    images = images.sort_values(["Errors", "FN", "FP", "Image"], ascending=[False, False, False, True])
    classes = pd.DataFrame(class_rows.values()).set_index("Class")
    classes["Precision"] = classes["TP"] / (classes["TP"] + classes["FP"]).replace(0, np.nan)
    classes["Recall"] = classes["TP"] / (classes["TP"] + classes["FN"]).replace(0, np.nan)
    return DetectionErrors(images, classes, matched_gt, missed_gt, false_ids, reviewed_ids)