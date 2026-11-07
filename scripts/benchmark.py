import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.predictor import Detector
from src.pipeline.media import capture_video
from src.pipeline.outputs import RunOutput
from src.pipeline.runtime import Runtime
from src.segmentation.predictor import OverflowSegmenter
from src.tracking.tracker import Tracker


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {"mean_ms": float(values.mean()), **{f"p{q}_ms": float(np.percentile(values, q)) for q in (50, 95, 99)},}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="A representative local video.")
    parser.add_argument("--device", default="auto", help="cpu, cuda:0, or auto.")
    parser.add_argument("--models-dir", type=Path)
    
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports/benchmarks")
    
    args = parser.parse_args()
    if min(args.max_frames, args.warmup, args.threads) < 1:
        parser.error("--max-frames, --warmup and --threads must be positive.")

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    
    runtime = Runtime(device=args.device, models_dir=args.models_dir)
    detector = Detector(runtime)
    segmenter = OverflowSegmenter(runtime)
    tracker = Tracker(runtime, sequence=args.input.stem)
    
    configs = {"detection": detector.config, "tracking": tracker.config, "segmentation": segmenter.config}
    cuda = runtime.device.type == "cuda"
    process = psutil.Process()

    def synchronize():
        if cuda:
            torch.cuda.synchronize(runtime.device)

    def timed(function):
        synchronize()
        started = time.perf_counter()
        value = function()
        synchronize()
        return value, (time.perf_counter() - started) * 1000

    with RunOutput(runtime, args.input, args.output_root, "benchmark", configs, True) as run:
        with capture_video(run.source) as capture:
            success, first = capture.read()
        if not success:
            raise ValueError("No video frames could be decoded.")

        with torch.inference_mode():
            run.progress("model_warmup", 0, args.warmup, force=True)
            started = time.perf_counter()
            for _ in range(args.warmup):
                detector.raw(first)
                segmenter.predict(first)
                runtime.infer(segmenter.config["models"]["roi"], [first], segmenter.config["segment_inference"], batch=1)
            synchronize()
            
            warmup_seconds = time.perf_counter() - started
            del first
            if cuda:
                torch.cuda.reset_peak_memory_stats(runtime.device)

            rows = []
            run.progress("benchmark", 0, args.max_frames, force=True)
            with capture_video(run.source) as capture:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                started = time.perf_counter()
                
                for frame in range(1, args.max_frames + 1):
                    frame_started = time.perf_counter()
                    success, image = capture.read()
                    decoded = time.perf_counter()
                    
                    if not success:
                        break
                    raw, detection_ms = timed(lambda: detector.raw(image))
                    tracks, tracking_ms = timed(lambda: tracker.update(image, raw_detections=raw))
                    masks, segmentation_ms = timed(lambda: segmenter.predict(image, image_id=frame))
                    detections = len(detector.filter(raw).data)
                    
                    frame_ms = (time.perf_counter() - frame_started) * 1000
                    rows.append({
                        "Frame": frame, "Height": image.shape[0], "Width": image.shape[1],
                        "DecodeMS": (decoded - frame_started) * 1000,
                        "DetectionMS": detection_ms, "TrackingMS": tracking_ms,
                        "SegmentationMS": segmentation_ms, "FrameMS": frame_ms,
                        "Detections": detections, "Tracks": len(tracks), "OverflowMasks": len(masks),
                        "ProcessRSSMiB": process.memory_info().rss / 2**20
                    })
                    
                    run.progress("benchmark", frame, args.max_frames)
                if not rows:
                    raise ValueError("No video frames could be decoded during measurement.")
                (tracks, counts), finalize_ms = timed(tracker.finalize)
                measured_seconds = time.perf_counter() - started

        table = pd.DataFrame(rows)
        table.to_csv(run.folder / "frame_timings.csv", index=False)
        tracks.to_csv(run.folder / "tracks.csv", index=False)
        
        stages = {
            "decode": "DecodeMS", "detection": "DetectionMS", "tracking_update": "TrackingMS",
            "segmentation_fused": "SegmentationMS", "frame": "FrameMS"
        }
        
        summary = run.complete({
            "frames": len(table), "frame_limit": args.max_frames,
            "scope": "Sequential video prefix; warmed models; decode + D1 + T3 + fused S2 + track finalization.",
            "excluded_from_processing_fps": "Model loading/warmup, overlay, encoding, and final CSV/JSON writes.",
            "processing_fps_includes": "Progress/status writes and per-frame RSS sampling.",
            "source_fps": fps if np.isfinite(fps) and fps > 0 else None,
            "device_name": torch.cuda.get_device_name(runtime.device) if cuda else "CPU",
            "cuda_build": torch.version.cuda,
            "torch_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads(),
            "warmup_iterations": args.warmup, "model_loading_and_warmup_seconds": warmup_seconds,
            "measured_processing_seconds": measured_seconds,
            "processing_fps": len(table) / measured_seconds,
            "track_finalization_ms": finalize_ms,
            "latency": {name: statistics(table[column]) for name, column in stages.items()},
            "max_sampled_process_rss_mib": float(table.ProcessRSSMiB.max()),
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(runtime.device) / 2**20 if cuda else None,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(runtime.device) / 2**20 if cuda else None,
            "counts_for_processed_frames": counts,
            "numerical_parity": "not_evaluated; compare with frozen validation references separately"
        })
        print(json.dumps(summary, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()