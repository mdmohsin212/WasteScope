import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.predictor import Detector
from src.pipeline.runtime import Runtime
from src.segmentation.predictor import OverflowSegmenter
from src.tracking.tracker import Tracker

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()

    image = (
        cv2.imread(str(args.image))
        if args.image
        else np.zeros((480, 640, 3), np.uint8)
    )

    if image is None:
        raise FileNotFoundError(f"Cannot decode image: {args.image}")

    torch.set_num_threads(min(4, torch.get_num_threads()))

    runtime = Runtime(device="cpu")
    detector = Detector(runtime)

    raw = detector.raw(image)
    detections = detector.filter(raw)

    tracker = Tracker(runtime, sequence="adapter_check")
    live_tracks = tracker.update(image, raw_detections=raw)
    _, counts = tracker.finalize()

    masks = OverflowSegmenter(runtime).predict(image)
    print(json.dumps({
        "device": "cpu",
        "image_shape": list(image.shape[:2]),
        "detections": detections.records(detector.names),
        "live_tracks": live_tracks,
        "overflow_instances": len(masks),
        "final_counts": counts,
        "note": ("One-frame adapter check; counting requires 10 observed frames.")
    }, indent=2))

    print("PASS: detection, tracking/counting and fused segmentation adapters completed.")

if __name__ == "__main__":
    main()