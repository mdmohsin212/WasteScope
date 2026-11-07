import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from src.pipeline import media, runner
from src.segmentation.masks import expanded_rois

NAMES = [
    "container_yellow", "container_green", "container_default", "container_blue",
    "container_oil", "container_battery", "container_biodegradable",
]
COLUMNS = ["Sequence", "Frame", "TrackID", "ClassID", "x1", "y1", "x2", "y2", "Score"]


class FixedTracker:
    def __init__(self, runtime, sequence):
        self.config = {"postprocessing": {"min_frames": 10}}
        self.sequence, self.frame, self.rows = sequence, 0, []

    def update(self, image, raw_detections=None):
        self.frame += 1
        native_id = 1 if self.frame <= 6 else 2
        rows = [[self.sequence, self.frame, native_id, 0, 40., 130., 80., 170., .9]]
        if self.frame in (3, 4, 5):
            rows.append([self.sequence, self.frame, 3, 4, 200., 130., 240., 180., .8])
        self.rows.extend(rows)
        return [dict(zip(COLUMNS, row)) for row in rows]

    def finalize(self):
        tracks = pd.DataFrame(self.rows, columns=COLUMNS)
        tracks["OriginalTrackID"] = tracks.TrackID
        tracks.loc[tracks.TrackID.eq(2), "TrackID"] = 1
        hits = tracks.groupby("TrackID").Frame.transform("nunique")
        tracks["CountEligible"] = hits.ge(10)
        counts = dict.fromkeys(NAMES, 0)
        counts[NAMES[0]] = int(tracks.CountEligible.any())
        return tracks, counts


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "outputs"
        
        runtime = SimpleNamespace(device=SimpleNamespace(type="cpu"), manifest={"release_id": "test-fixture"})
        prediction = {"class_id": 0, "class_name": NAMES[0], "confidence": .9, "xyxy": [40., 130., 80., 170.]}
        raw = SimpleNamespace(records=lambda names: [prediction])
        
        self.detector = SimpleNamespace(
            names=NAMES, config={"inference": {"half": True, "rect": True}},
            raw=Mock(return_value=raw), filter=Mock(side_effect=lambda value: value),
            predict=Mock(return_value=raw),
        )
        self.segmenter = SimpleNamespace(
            config={"segment_inference": {"half": True}},
            predict=Mock(return_value=[]),
        )
        
        for name, value in (("Detector", self.detector), ("OverflowSegmenter", self.segmenter)):
            mock = patch.object(runner, name, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)
            
        mock = patch.object(runner, "Tracker", FixedTracker)
        mock.start()
        self.addCleanup(mock.stop)
        self.pipeline = runner.WasteScopePipeline(runtime, verbose=False)

    def image_source(self):
        path = self.root / "sample.png"
        success, encoded = cv2.imencode(".png", np.full((240, 320, 3), 80, np.uint8))
        self.assertTrue(success)
        path.write_bytes(encoded.tobytes())
        return path

    def video_source(self, frames=12):
        path = self.root / "sample.mp4"
        with media.H264Writer(path, 5, (240, 320)) as writer:
            for frame in range(frames):
                writer.write(np.full((240, 320, 3), 60 + frame, np.uint8))
        return path

    def assert_failed(self, state="failed"):
        folder = next(self.output.iterdir())
        status = json.loads((folder / "status.json").read_text())
        self.assertEqual(status["state"], state)
        
        self.assertFalse((folder / "summary.json").exists())
        self.assertFalse((folder / "annotated.mp4").exists())

    def test_image_outputs_masks_config_and_unique_folders(self):
        mask = np.zeros((240, 320), np.uint8)
        mask[150:170, 150:170] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))
        
        rle["counts"] = rle["counts"].decode("ascii")
        self.segmenter.predict.return_value = [{"image_id": 1, "category_id": 1, "score": .8, "segmentation": rle}]
        source = self.image_source()
        summary = self.pipeline.image(source, self.output)
        
        folder = Path(summary["output_dir"])
        self.assertEqual(summary["detection_counts"][NAMES[0]], 1)
        self.assertEqual(summary["overflow_instances"], 1)
        self.assertEqual(cv2.imread(str(folder / "annotated.png")).shape, (240, 320, 3))
        record = json.loads((folder / "predictions.json").read_text())
        
        np.testing.assert_array_equal(mask_utils.decode(record["overflow"][0]["segmentation"]), mask)
        config = json.loads((folder / "run_config.json").read_text())
        self.assertFalse(config["effective_configs"]["detection"]["inference"]["half"])
        self.assertTrue(self.detector.config["inference"]["half"])
        
        self.assertEqual(json.loads((folder / "status.json").read_text())["state"], "complete")
        self.assertFalse((folder / "status.json.tmp").exists())
        self.assertNotEqual(self.pipeline.image(source, self.output)["output_dir"], str(folder))

    def test_video_final_ids_counts_and_playback(self):
        summary = self.pipeline.video(self.video_source(), self.output)
        folder = Path(summary["output_dir"])
        records = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines()]
        tracks = pd.read_csv(folder / "tracks.csv")
        
        self.assertEqual(len(records), 12)
        self.assertEqual(records[8]["counts"][NAMES[0]], 0)
        self.assertEqual(records[9]["counts"][NAMES[0]], 1)
        self.assertEqual(records[6]["tracks"][0]["OriginalTrackID"], 2)
        self.assertEqual(records[6]["tracks"][0]["TrackID"], 1)
        
        self.assertEqual(records[-1]["counts"], summary["final_counts"])
        self.assertEqual(summary["counted_containers"], 1)
        self.assertEqual(len(tracks), 15)
        
        self.assertFalse(tracks.loc[tracks.TrackID.eq(3), "CountEligible"].any())
        self.assertEqual(self.detector.raw.call_count, 12)
        self.assertEqual(self.segmenter.predict.call_count, 12)
        self.assertFalse(any("pending" in p.name or "raw.jsonl" in p.name for p in folder.iterdir()))
        
        with media.capture_video(folder / "annotated.mp4") as capture:
            decoded = 0
            while capture.read()[0]:
                decoded += 1
        self.assertEqual(decoded, 12)

    def test_empty_video_predictions_complete(self):
        empty = pd.DataFrame(columns=COLUMNS + ["OriginalTrackID", "CountEligible"])
        tracker = SimpleNamespace(
            config={"postprocessing": {"min_frames": 10}}, update=Mock(return_value=[]),
            finalize=Mock(return_value=(empty, dict.fromkeys(NAMES, 0))),
        )
        
        self.detector.raw.return_value = SimpleNamespace(records=lambda names: [])
        with patch.object(runner, "Tracker", return_value=tracker):
            summary = self.pipeline.video(self.video_source(2), self.output)
            
        self.assertEqual(summary["counted_containers"], 0)
        self.assertEqual(summary["frames_with_overflow_predictions"], 0)
        self.assertEqual(summary["status"], "complete")

    def test_invalid_image_records_failure(self):
        path = self.root / "bad.png"
        path.write_bytes(b"invalid image")
        with self.assertRaises(ValueError):
            self.pipeline.image(path, self.output)
        self.assert_failed()

    def test_keyboard_interrupt_records_cancellation(self):
        source = self.video_source(2)
        with patch.object(FixedTracker, "update", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.pipeline.video(source, self.output)
        self.assert_failed("cancelled")

    def test_encoder_failure_does_not_publish_completed_video(self):
        source = self.video_source(2)
        with patch.object(media.imageio_ffmpeg, "get_ffmpeg_exe", return_value=str(self.root / "missing-ffmpeg")):
            with self.assertRaises(OSError):
                self.pipeline.video(source, self.output)
        self.assert_failed()

    def test_odd_noncontiguous_frame_is_encoded_with_padding(self):
        path = self.root / "odd.mp4"
        frame = np.zeros((241, 321, 3), np.uint8)[:, ::-1]
        self.assertFalse(frame.flags.c_contiguous)
        
        with media.H264Writer(path, 5, frame.shape[:2]) as writer:
            writer.write(frame)
        with media.capture_video(path) as capture:
            success, decoded = capture.read()
            
        self.assertTrue(success)
        self.assertEqual(decoded.shape[:2], (242, 322))

    def test_encoder_timeout_kills_process_and_closes_log(self):
        process = Mock()
        process.wait.side_effect = [subprocess.TimeoutExpired("ffmpeg", 60), -9]
        with patch.object(media.subprocess, "Popen", return_value=process):
            writer = media.H264Writer(self.root / "slow.mp4", 5, (240, 320))
            with self.assertRaisesRegex(RuntimeError, "60 seconds"):
                with writer:
                    pass
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)
        self.assertTrue(writer.log.closed)


class MaskGeometryTests(unittest.TestCase):
    def test_roi_bounds_and_duplicates(self):
        config = {"crop_factor": 3, "minimum_side": 512}
        boxes = [[100, 100, 200, 200]] * 2
        self.assertEqual(
            expanded_rois(boxes, 640, 480, config),
            [(0, 0, 512, 480)]
        )
        self.assertEqual(expanded_rois([], 640, 480, config), [])

if __name__ == "__main__":
    unittest.main(verbosity=2)