from threading import RLock
from types import SimpleNamespace
import numpy as np
import pandas as pd
from ultralytics.engine.results import Boxes
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.trackers.bot_sort import BOTSORT
from ..detection.predictor import Detector
from ..evaluation.tracking import postprocess_tracks
from ..pipeline.runtime import check_image

_ID_LOCK = RLock()
COLUMNS = [
    "Sequence", "Frame", "TrackID", "ClassID",
    "x1", "y1", "x2", "y2", "Score",
]

class Tracker:
    def __init__(self, runtime, sequence="video"):
        self.config = runtime.config("tracking")
        self.detector = Detector(runtime)
        self.names = self.detector.names
        self.sequence = str(sequence)
        
        self._lock = RLock()
        self._rows = []
        self._shape = None
        self._final = None
        self._frame = 0
        self._id_count = 0
        
        with _ID_LOCK:
            previous = BaseTrack._count
            try:
                self._tracker = BOTSORT(
                    SimpleNamespace(**self.config["tracker"]),
                    frame_rate=self.config["tracker_frame_rate_argument"]
                )
            finally:
                BaseTrack._count = previous
        
        
    def update(self, image, raw_detections=None):
        with self._lock:
            if self._final is not None:
                raise RuntimeError("This video is finalized; create a new Tracker for the next video.")

            check_image(image)
            shape = tuple(image.shape[:2])

            if self._shape is not None and shape != self._shape:
                raise ValueError("Video frame dimensions changed during tracking.")

            raw = raw_detections if raw_detections is not None else self.detector.raw(image)
            if raw.image_shape != shape:
                raise ValueError("Detections and frame have different image dimensions.")

            with _ID_LOCK:
                previous = BaseTrack._count
                try:
                    BaseTrack._count = self._id_count
                    result = self._tracker.update(Boxes(raw.data, orig_shape=shape), image)
                    self._id_count = BaseTrack._count
                finally:
                    BaseTrack._count = previous

            self._shape = shape
            self._frame += 1
            height, width = shape
            rows = []

            for (x1, y1, x2, y2, track_id, score, class_id, _,) in result:
                box = np.clip([x1, y1, x2, y2], [0, 0, 0, 0], [width, height, width, height])
                rows.append([
                    self.sequence,
                    self._frame,
                    int(track_id),
                    int(class_id),
                    *map(float, box),
                    float(score)
                ])
                
            self._rows.extend(rows)

            return [dict(zip(COLUMNS, row)) for row in rows]

    def finalize(self):
        with self._lock:
            if self._final is None:
                tracks = pd.DataFrame(self._rows,columns=COLUMNS,)
                frames = pd.DataFrame(columns=["Sequence", "Width", "Height"])

                if self._shape is not None:
                    frames.loc[0] = [self.sequence, self._shape[1], self._shape[0]]

                params = self.config["postprocessing"]
                tracks = postprocess_tracks(
                    tracks,
                    frames,
                    min_frames=params["min_frames"],
                    max_gap=params["max_gap"],
                    max_distance=params["max_distance"]
                )

                eligible = tracks.loc[tracks["CountEligible"]]
                classes =  eligible.groupby("TrackID")["ClassID"].agg(lambda values: int(values.mode().iloc[0]))
                
                counts = {
                    name: int(classes.eq(index).sum())
                    for index, name in enumerate(self.names)
                }
                self._final = tracks, counts

            return (self._final[0].copy(), self._final[1].copy())