import csv
import io
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm.auto import tqdm
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.models.yolo.segment.train import SegmentationTrainer


METRIC_KEYS = ["metrics/canonical_mask_AP50", "metrics/canonical_mask_AP50-95"]

def project_mask(mask, rect, image_shape):
    """Encode a crop mask in original coordinates without a full-image canvas."""
    x1, y1, x2, y2 = rect
    height, width = image_shape
    if mask.shape != (y2 - y1, x2 - x1):
        mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)
    left, top, crop_width, crop_height = cv2.boundingRect(mask)
    if not crop_width:
        return None
    mask = mask[top:top + crop_height, left:left + crop_width]
    x1, y1 = x1 + left, y1 + top
    flat = np.pad(mask, ((1, 1), (0, 0))).ravel(order="F")
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    if not edges.size:
        return None
    local_height = mask.shape[0] + 2
    starts, ends = edges[::2], edges[1::2]
    starts = (x1 + starts // local_height) * height + y1 + starts % local_height - 1
    ends = (x1 + ends // local_height) * height + y1 + ends % local_height - 1
    adjacent = starts[1:] == ends[:-1]
    starts, ends = starts[np.r_[True, ~adjacent]], ends[np.r_[~adjacent, True]]
    counts = np.empty(2 * len(starts) + 1, dtype=np.int64)
    counts[0], counts[-1] = starts[0], height * width - ends[-1]
    counts[1:-1:2] = ends - starts
    counts[2:-1:2] = starts[1:] - ends[:-1]
    return mask_utils.frPyObjects({"size": [height, width], "counts": counts.tolist()}, height, width)


def candidate(image_id, rle, score):
    return {
        "image_id": image_id, "category_id": 1, "segmentation": rle, "score": float(score),
        "_bbox": mask_utils.toBbox(rle).tolist(),
    }


def fuse_masks(predictions, threshold, max_det):
    """Suppress mask IoU >= threshold; S1 comes first on equal scores."""
    kept = []
    for pred in sorted(predictions, key=lambda item: -item["score"]):
        x, y, w, h = pred["_bbox"]
        overlaps = []
        for other in kept:
            a, b, c, d = other["_bbox"]
            if x < a + c and a < x + w and y < b + d and b < y + h:
                overlaps.append(other["segmentation"])
        if overlaps and np.any(mask_utils.iou([pred["segmentation"]], overlaps, [0] * len(overlaps)) >= threshold):
            continue
        kept.append(pred)
        if len(kept) == max_det:
            break
    return kept


def coco_records(predictions, *, json_safe=False):
    records = []
    for pred in predictions:
        rle = pred["segmentation"].copy()
        if json_safe and isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("ascii")
        records.append({"image_id": pred["image_id"], "category_id": 1,
                        "segmentation": rle, "score": pred["score"]})
    return records


def canonical_ap(ground_truth, predictions):
    with redirect_stdout(io.StringIO()):
        if predictions:
            detections = ground_truth.loadRes(coco_records(predictions))
        else:
            detections = COCO()
            detections.dataset = {"images": ground_truth.dataset["images"],
                                  "categories": ground_truth.dataset["categories"], "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(ground_truth, detections, "segm")
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {METRIC_KEYS[0]: float(evaluator.stats[1]), METRIC_KEYS[1]: float(evaluator.stats[0])}


class FullImageMonitor:
    def __init__(self, data, *, batch=8, encoding_workers=4):
        self.data, self.batch, self.encoding_workers = Path(data), batch, encoding_workers
        self.policy = json.loads((self.data / "roi_policy.json").read_text())
        self.index = json.loads((self.data / "val/roi_index.json").read_text())
        s1_metrics = json.loads((self.data / "support/s1_val_metrics.json").read_text())
        if s1_metrics["checkpoint_sha256"] != self.policy["full_image_checkpoint_sha256"]:
            raise ValueError("Frozen S1 predictions do not match roi_policy.json")
        with redirect_stdout(io.StringIO()):
            self.gt = COCO(str(self.data / "val/annotations.json"))
        self.gt.dataset.setdefault("info", {})
        for ann in self.gt.anns.values():
            ann["segmentation"] = self.gt.annToRLE(ann)
        self.s1 = defaultdict(list)
        for pred in json.loads((self.data / "support/s1_val_predictions.json").read_text()):
            rle = pred["segmentation"].copy()
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("ascii")
            self.s1[pred["image_id"]].append(candidate(pred["image_id"], rle, pred["score"]))

    def __call__(self, trainer):
        started = time.monotonic()
        evaluation_model = deepcopy(trainer.ema.ema).half().float().eval()
        predictor = SegmentationPredictor(overrides={
            "task": "segment", "mode": "predict", "imgsz": trainer.args.imgsz,
            "device": 0, "half": True, "batch": self.batch, "rect": False,
            "conf": self.policy["AP_confidence_min"], "max_det": self.policy["max_detections"],
            "retina_masks": True, "save": False, "verbose": False,
        })
        predictor.setup_model(evaluation_model, verbose=False)
        roi = defaultdict(list)
        with ThreadPoolExecutor(max_workers=self.encoding_workers) as pool:
            results = predictor(source=str(self.data / "val/images"), stream=True)
            for result in tqdm(results, total=len(self.index), desc="Canonical validation: ROI inference"):
                if result.masks is None:
                    continue
                info = self.index[Path(result.path).stem]
                masks = result.masks.data.byte().cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                def encode(item):
                    mask, score = item
                    rle = project_mask(mask, info["box"], info["image_shape"])
                    return candidate(info["image_id"], rle, score) if rle is not None else None

                roi[info["image_id"]].extend(
                    pred for pred in pool.map(encode, zip(masks, scores)) if pred is not None
                )
        inference_seconds = time.monotonic() - started
        del predictor, evaluation_model
        fusion_started = time.monotonic()
        fused = []
        for image_id in self.gt.imgs:
            fused.extend(fuse_masks(self.s1[image_id] + roi[image_id],
                                    self.policy["fusion_iou"], self.policy["max_detections"]))
        metrics = canonical_ap(self.gt, fused)
        fusion_seconds = time.monotonic() - fusion_started
        fitness = metrics[METRIC_KEYS[1]]
        epoch = trainer.epoch + 1
        run = Path(trainer.save_dir)
        timing = run / "validation_times.csv"
        new_file = not timing.exists()
        with timing.open("a", newline="") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(["epoch", "roi_inference_encoding_seconds", "fusion_coco_seconds"])
            writer.writerow([epoch, inference_seconds, fusion_seconds])
        print(f"Epoch {epoch}: full-image fused Mask AP50={metrics[METRIC_KEYS[0]]:.4f}, "
              f"AP50-95={fitness:.4f}; ROI={inference_seconds:.1f}s, fusion/COCO={fusion_seconds:.1f}s", flush=True)
        if trainer.best_fitness is None or fitness >= trainer.best_fitness:
            roi_only = []
            for image_id in self.gt.imgs:
                roi_only.extend(fuse_masks(roi[image_id], self.policy["fusion_iou"], self.policy["max_detections"]))
            for name, predictions in (("best_fused_predictions.json", fused), ("best_roi_predictions.json", roi_only)):
                (run / name).write_text(json.dumps(coco_records(predictions, json_safe=True)))
            record = {"epoch": epoch, "checkpoint_monitor": self.policy["checkpoint_monitor"],
                      "metrics": metrics, "roi_only_metrics": canonical_ap(self.gt, roi_only),
                      "original_val_images": len(self.gt.imgs), "original_gt_instances": len(self.gt.anns)}
            (run / "best_canonical_metrics.json").write_text(json.dumps(record, indent=2))
        return metrics, fitness


class CanonicalROITrainer(SegmentationTrainer):
    def get_validator(self):
        validator = super().get_validator()
        validator.metrics = SimpleNamespace(keys=METRIC_KEYS, results_dict={})
        return validator

    def validate(self):
        metrics, fitness = self.roi_monitor(self)
        if self.best_fitness is None or fitness > self.best_fitness:
            self.best_fitness = fitness
        self.validator.metrics.results_dict = metrics
        return metrics, fitness

    def final_eval(self):
        self.metrics = json.loads((Path(self.save_dir) / "best_canonical_metrics.json").read_text())["metrics"]
        self.validator.metrics.results_dict = self.metrics

    def plot_metrics(self):
        import matplotlib.pyplot as plt

        with Path(self.csv).open() as file:
            rows = list(csv.DictReader(file))
        epochs = [int(row["epoch"]) for row in rows]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for key in METRIC_KEYS:
            axes[0].plot(epochs, [float(row[key]) for row in rows], label=key.split("/")[-1])
        for key in rows[0]:
            if key.startswith("train/") and "loss" in key:
                axes[1].plot(epochs, [float(row[key]) for row in rows], label=key.split("/")[-1])
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(self.save_dir) / "results.png", dpi=160)
        plt.close(fig)