from collections import Counter
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np

from ..detection.predictor import Detector
from ..segmentation.predictor import OverflowSegmenter
from ..tracking.tracker import Tracker
from .media import H264Writer, annotate, capture_video
from .outputs import RunOutput, write_json
from .runtime import ROOT

def finalized_records(path, tracks, names, min_frames):
    mapping = {
        int(row.OriginalTrackID): (int(row.TrackID), bool(row.CountEligible))
        for row in tracks.drop_duplicates("OriginalTrackID").itertuples()
    }
    
    classes = {
        int(track_id): int(group.ClassID.mode().iloc[0])
        for track_id, group in tracks.groupby("TrackID")
    }
    hits, counted, counts = Counter(), set(), dict.fromkeys(names, 0)
    
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            for track in record["tracks"]:
                track["OriginalTrackID"] = track["TrackID"]
                track["TrackID"], track["CountEligible"] = mapping[track["OriginalTrackID"]]
                
            present = {track["TrackID"] for track in record["tracks"]}
            hits.update(present)
            for track_id in present - counted:
                if hits[track_id] >= min_frames:
                    counts[names[classes[track_id]]] += 1
                    counted.add(track_id)
                    
            record["counts"] = counts.copy()
            yield record
            

class WasteScopePipeline:
    def __init__(self, runtime, *, verbose=True):
        self.runtime, self.verbose = runtime, verbose
        
    def image(self, input_path, output_root=ROOT / "outputs"):
        detector = Detector(self.runtime)
        segmenter = OverflowSegmenter(self.runtime)
        configs = {"detection": detector.config, "segmentation": segmenter.config}
        
        with RunOutput(self.runtime, input_path, output_root, "image", configs, self.verbose) as run:
            image = cv2.imdecode(np.frombuffer(run.source.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError(f"Cannot decode image: {run.source}")
            
            shape = list(image.shape[:2])
            detections = detector.predict(image).records(detector.names)
            masks = segmenter.predict(image, image_id=1)
            
            record = {"schema_version": 1, "image_shape": shape,"detections": detections, "overflow": masks}
            found = Counter(item["class_name"] for item in detections)
            counts = {name: found[name] for name in detector.names}
            write_json(run.folder / "predictions.json", record)
            
            success, encoded = cv2.imencode(".png", annotate(image, record, detector.names, counts))
            
            if not success:
                raise OSError("Cannot encode annotated image.")
            
            (run.folder / "annotated.png").write_bytes(encoded.tobytes())
            return run.complete({"frames": 1, "image_shape": shape, "detection_counts": counts, "overflow_instances": len(masks)})
        
    
    def video(self, input_path, output_root=ROOT / "outputs"):
        detector = Detector(self.runtime)
        segmenter = OverflowSegmenter(self.runtime)
        tracker = Tracker(self.runtime, sequence=Path(input_path).stem)
        configs = {
            "detection": detector.config, "tracking": tracker.config,
            "segmentation": segmenter.config,
        }
        
        with RunOutput(self.runtime, input_path, output_root, "video", configs, self.verbose) as run:
            raw_path = run.folder / "frames.raw.jsonl"
            frames = overflow_frames = 0
            shape = None
            
            with capture_video(run.source) as capture, raw_path.open("w", encoding="utf-8") as stream:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError("The video has no valid FPS metadata.")
                
                reported = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                reported = int(reported) if math.isfinite(reported) and reported > 0 else None

                while True:
                    success, image = capture.read()
                    if not success:
                        break
                    
                    frames += 1
                    shape = tuple(image.shape[:2])
                    raw = detector.raw(image)
                    
                    tracks = tracker.update(image, raw_detections=raw)
                    masks = segmenter.predict(image, image_id=frames)
                    overflow_frames += bool(masks)
                    
                    record = {
                        "schema_version": 1,
                        "frame": frames,
                        "timestamp_s": (frames - 1) / fps,
                        "image_shape": list(shape),
                        "detections": detector.filter(raw).records(detector.names),
                        "tracks": tracks,
                        "overflow": masks
                    }
                    
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    if frames == 1 or time.monotonic() - run.last_update >= 2:
                        stream.flush()
                        run.progress("inference", frames, reported, force=True)

            if not frames:
                raise ValueError("No video frames could be decoded.")
            
            tracks, final_counts = tracker.finalize()
            tracks_path = run.folder / "tracks.pending.csv"
            tracks.to_csv(tracks_path, index=False)
            tracks_path.replace(run.folder / "tracks.csv")

            pending_video = run.folder / "annotated.pending.mp4"
            pending_json = run.folder / "frames.pending.jsonl"
            records = finalized_records(raw_path, tracks, detector.names, tracker.config["postprocessing"]["min_frames"])
            
            run.progress("rendering", 0, frames, force=True)
            rendered, visible_counts = 0, dict.fromkeys(detector.names, 0)
            
            with (
                capture_video(run.source) as capture,
                H264Writer(pending_video, fps, shape) as writer,
                pending_json.open("w", encoding="utf-8") as stream
            ):
                for rendered, record in enumerate(records, 1):
                    success, image = capture.read()
                    if not success:
                        raise ValueError("Video decoding changed between inference and rendering.")
                    
                    visible_counts = record["counts"]
                    writer.write(annotate(image, record, detector.names, visible_counts, video=True))
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    run.progress("rendering", rendered, frames)
                    
                if capture.read()[0]:
                    raise ValueError("Video contains extra frames on the second decoding pass.")

            if rendered != frames or visible_counts != final_counts:
                raise RuntimeError("Rendered frames or cumulative counts differ from finalized results.")
            
            pending_video.replace(run.folder / "annotated.mp4")
            pending_json.replace(run.folder / "frames.jsonl")
            raw_path.unlink()

            warnings = []
            if reported not in (None, frames):
                warnings.append(f"Metadata reported {reported} frames; the decoder returned {frames}.")
                
            return run.complete({
                "frames": frames,
                "source_fps": fps,
                "image_shape": list(shape),
                "encoded_shape": [value + value % 2 for value in shape],
                "audio_included": False,
                "count_mode": "postprocessed cumulative unique tracks",
                "final_counts": final_counts,
                "counted_containers": sum(final_counts.values()),
                "tracks_before_duration_filter": int(tracks.TrackID.nunique()),
                "frames_with_overflow_predictions": overflow_frames,
                "overflow_frame_fraction": overflow_frames / frames,
                "warnings": warnings
            })