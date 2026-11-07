import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

os.environ["YOLO_AUTOINSTALL"] = "False"
import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detection.predictor import Detector
from src.pipeline.media import capture_video
from src.pipeline.outputs import RunOutput, file_sha256
from src.pipeline.runtime import Runtime
from src.segmentation.predictor import OverflowSegmenter


class OnnxRuntime(Runtime):
    def __init__(self, onnx_dir):
        super().__init__(device="cpu")
        self.onnx_dir = onnx_dir
        self.exports = json.loads((onnx_dir / "manifest.json").read_text())
        if self.exports["source_release_id"] != self.manifest["release_id"]:
            raise ValueError("The ONNX bundle belongs to another model release.")

    def _load(self, model_id):
        if model_id not in self._models:
            record = self.exports["models"][model_id]
            original = self.manifest["models"][model_id]
            if record["source_sha256"] != original["sha256"]:
                raise ValueError(f"ONNX source checkpoint differs: {model_id}")
            
            path = self.onnx_dir / record["filename"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError(f"ONNX SHA256 mismatch: {model_id}")
            self._models[model_id] = YOLO(str(path), task=original["task"])
            
        return self._models[model_id]


def samples(path, count):
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        yield 0, image
        
    else:
        with capture_video(path) as capture:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 1:
                raise ValueError("Video needs valid frame-count metadata.")
            
            for index in np.linspace(0, total - 1, min(count, total), dtype=int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                success, image = capture.read()
                if not success:
                    raise ValueError(f"Cannot decode video frame {index + 1}")
                yield int(index + 1), image


def match(iou, classes_a, classes_b, scores_a, scores_b, minimum, score_atol):
    na, nb = len(scores_a), len(scores_b)
    if not na or not nb:
        return {"passed": na == nb, "torch_instances": na, "onnx_instances": nb,
                "min_iou": None, "max_score_difference": None}
        
    compatible = np.asarray(classes_a)[:, None] == np.asarray(classes_b)[None, :]
    left, right = linear_sum_assignment(1 - iou + (~compatible) * 2)
    overlaps = iou[left, right]
    differences = np.abs(np.asarray(scores_a)[left] - np.asarray(scores_b)[right])
    passed = (na == nb and compatible[left, right].all()and (overlaps >= minimum).all() and (differences <= score_atol).all())
    
    return {"passed": bool(passed), "torch_instances": na, "onnx_instances": nb,
            "min_iou": float(overlaps.min()), "max_score_difference": float(differences.max())}


def compare_boxes(a, b, minimum, score_atol):
    x1y1 = np.maximum(a[:, None, :2], b[None, :, :2])
    x2y2 = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    
    intersection = np.maximum(x2y2 - x1y1, 0).prod(axis=2)
    area_a = np.maximum(a[:, 2:4] - a[:, :2], 0).prod(axis=1)
    area_b = np.maximum(b[:, 2:4] - b[:, :2], 0).prod(axis=1)
    
    iou = intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-12)
    return match(iou, a[:, 5], b[:, 5], a[:, 4], b[:, 4], minimum, score_atol)


def compare_masks(a, b, minimum, score_atol):
    iou = (mask_utils.iou([x["segmentation"] for x in a], [x["segmentation"] for x in b], [0] * len(b))
           if a and b else np.empty((len(a), len(b))))
    
    return match(iou, [x["category_id"] for x in a], [x["category_id"] for x in b], [x["score"] for x in a], [x["score"] for x in b], minimum, score_atol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--onnx-dir", type=Path, default=ROOT / "models/onnx")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models/weights")
    
    parser.add_argument("--samples", type=int, default=3, help="Evenly spaced frames per clip.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--box-iou", type=float, default=0.99)
    parser.add_argument("--mask-iou", type=float, default=0.98)
    parser.add_argument("--score-atol", type=float, default=0.001)
    
    args = parser.parse_args()
    if min(args.samples, args.threads) < 1:
        parser.error("samples and threads must be positive")
    if not (0 <= args.box_iou <= 1 and 0 <= args.mask_iou <= 1 and args.score_atol >= 0):
        parser.error("Invalid comparison tolerances")
    if version("ultralytics") != "8.4.1":
        raise RuntimeError("Use ultralytics==8.4.1.")
    
    torch.set_num_threads(args.threads)
    reference = Runtime(device="cpu", models_dir=args.models_dir)
    candidate = OnnxRuntime(args.onnx_dir)
    detectors = [Detector(runtime) for runtime in (reference, candidate)]
    segmenters = [OverflowSegmenter(runtime) for runtime in (reference, candidate)]
    configs = {"detection": detectors[0].config, "segmentation": segmenters[0].config}
    
    with RunOutput(reference, args.input[0], ROOT / "reports/onnx_checks", "conversion_check", configs, True) as run:
        rows = []
        with torch.inference_mode():
            for path in args.input:
                for frame, image in samples(path, args.samples):
                    raw = [detector.raw(image) for detector in detectors]
                    masks = [segmenter.predict(image, image_id=1, for_ap=True) for segmenter in segmenters]
                    confidence = segmenters[0].config["operating_point"]["confidence"]
                    
                    comparisons = {
                        "D1_raw": compare_boxes(raw[0].data, raw[1].data, args.box_iou, args.score_atol),
                        "D1_operating": compare_boxes(detectors[0].filter(raw[0]).data, detectors[1].filter(raw[1]).data,
                                                      args.box_iou, args.score_atol),
                        "S2_fused_AP": compare_masks(masks[0], masks[1], args.mask_iou, args.score_atol),
                        "S2_fused_operating": compare_masks(
                            [x for x in masks[0] if x["score"] > confidence],
                            [x for x in masks[1] if x["score"] > confidence], args.mask_iou, args.score_atol),
                    }
                    rows.extend({"Input": path.name, "Frame": frame, "Profile": key, **value}
                                for key, value in comparisons.items())
                    run.progress("conversion_check", len(rows) // 4)
                    
        table = pd.DataFrame(rows)
        table.to_csv(run.folder / "comparisons.csv", index=False)
        coverage = {name: int(group.torch_instances.sum()) for name, group in table.groupby("Profile")}
        covered = all(coverage.get(key, 0) > 0 for key in("D1_raw", "D1_operating", "S2_fused_AP", "S2_fused_operating"))
        
        passed = bool(table.passed.all() and covered)
        run.complete({
            "frames": len(table) // 4, "conversion_parity_passed": passed,
            "verdict": "passed" if passed else "mismatch" if not table.passed.all() else "insufficient_positive_samples",
            "scope": "CPU FP32 PyTorch vs CPU FP32 ONNX: D1 boxes and original-image fused S2 masks.",
            "coverage": coverage, "tolerances": {"box_iou": args.box_iou, "mask_iou": args.mask_iou, "score_atol": args.score_atol},
            "inputs": [{"path": str(p.resolve()), "sha256": file_sha256(p)} for p in args.input],
            "onnx_bundle": candidate.exports,
            "versions": {name: version(name) for name in ("onnx", "onnxruntime")},
            "remaining": "Historical frozen-validation metrics, full-sequence tracking parity and backend speed comparison."
        })
        
        print(table.to_string(index=False))
        print("Report:", run.folder)
        
    if not passed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()