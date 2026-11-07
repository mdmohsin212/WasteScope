import math
import cv2
import numpy as np
from pycocotools import mask as mask_utils

def expanded_rois(boxes, width, height, config):
    rects = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(float, box)
        side = max(
            config["crop_factor"] * (x2 - x1),
            config["crop_factor"] * (y2 - y1),
            config["minimum_side"]
        )
        
        crop_w = min(side, width)
        crop_h = min(side, height)
        
        left = min(max((x1 + x2 - crop_w) / 2, 0), width - crop_w)
        top = min(max((y1 + y2 - crop_h) / 2, 0), height - crop_h)
        
        rect = (math.floor(left), math.floor(top), math.ceil(left + crop_w), math.ceil(top + crop_h))
        
        if rect[2] > rect[0] and rect[3] > rect[1]:
            rects.append(rect)
        
    return list(dict.fromkeys(rects))


def project_mask(mask, rect, image_shape):
    x1, y1, x2, y2 = rect
    height, width = image_shape
    
    if mask.shape != (y2 - y1, x2 - x1):
        mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    mask = (mask > 0).astype(np.uint8)
    left, top, crop_width, crop_height = cv2.boundingRect(mask)
    if not crop_width:
        return None
    
    mask = mask[top: top + crop_height, left: left + crop_width]
    
    x1, y1 = x1 + left, y1 + top
    flat = np.pad(mask, ((1, 1), (0, 0))).ravel(order="F")
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1

    if not edges.size:
        return None

    local_height = mask.shape[0] + 2
    starts, ends = edges[::2], edges[1::2]

    starts = ((x1 + starts // local_height) * height + y1 + starts % local_height - 1)
    ends = ((x1 + ends // local_height) * height + y1 + ends % local_height - 1)

    adjacent = starts[1:] == ends[:-1]
    starts = starts[np.r_[True, ~adjacent]]
    ends = ends[np.r_[~adjacent, True]]

    counts = np.empty(2 * len(starts) + 1,dtype=np.int64)
    counts[0] = starts[0]
    counts[-1] = height * width - ends[-1]
    counts[1:-1:2] = ends - starts
    counts[2:-1:2] = starts[1:] - ends[:-1]

    return mask_utils.frPyObjects({"size": [height, width], "counts": counts.tolist()}, height,width)


def candidate(image_id, rle, score):
    return{
        "image_id": int(image_id),
        "category_id": 1,
        "segmentation": rle,
        "score": float(score),
        "_bbox": mask_utils.toBbox(rle).tolist()   
    }


def fuse_masks(predictions, threshold, max_det):
    kept = []
    for pred in sorted(predictions, key=lambda item: -item["score"]):
        x, y, w, h = pred["_bbox"]
        overlaps = []
        
        for other in kept:
            a, b, c, d = other["_bbox"]
            if (x < a + c and a < x + w and y < b + d and b < y + h):
                overlaps.append(other["segmentation"])
        
        if overlaps and np.any(mask_utils.iou([pred["segmentation"]], overlaps, [0] * len(overlaps)) >= threshold):
            continue
        
        kept.append(pred)
        if len(kept) == max_det:
            break
        
    return kept


def coco_records(predictions):
    records = []

    for pred in predictions:
        rle = pred["segmentation"].copy()

        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("ascii")

        records.append({
            "image_id": pred["image_id"],
            "category_id": 1,
            "segmentation": rle,
            "score": pred["score"]
        })

    return records