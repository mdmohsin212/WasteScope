from contextlib import contextmanager, suppress
import subprocess
import cv2
import imageio_ffmpeg
import numpy as np
from pycocotools import mask as mask_utils

COLORS = [
    (0, 210, 255), (0, 180, 0), (180, 180, 180), (255, 100, 20),
    (40, 100, 150), (200, 60, 200), (40, 140, 60)
]

@contextmanager
def capture_video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        yield capture
    finally:
        capture.release()
        

def annotate(image, record, names, counts, *, video=False):
    result = image.copy()
    height, width = result.shape[:2]
    foreground = np.zeros((height, width), dtype=bool)
    
    contours = []
    for prediction in record["overflow"]:
        mask = mask_utils.decode(prediction["segmentation"])
        foreground |= mask.astype(bool)
        contours.extend(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        
    result[foreground] = (0.65 * result[foreground] + 0.35 * np.array([30, 30, 240])).astype(np.uint8)
    cv2.drawContours(result, contours, -1, (30, 30, 240), 2)
    scale = max(0.4, min(0.8, width / 1800))
    
    for item in record["tracks"] if video else record["detections"]:
        if video:
            class_id, score = int(item["ClassID"]), item["Score"]
            box = [item[key] for key in ("x1", "y1", "x2", "y2")]
            prefix = f"ID {item['TrackID']} | "
        else:
            class_id, score = int(item["class_id"]), item["confidence"]
            box, prefix = item["xyxy"], ""
            
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        color = COLORS[class_id % len(COLORS)]
        label = names[class_id].removeprefix("container_")
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        cv2.putText(
            result, f"{prefix}{label} {score:.2f}", (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
        )

    items = [f"{name.removeprefix('container_')}: {counts[name]}" for name in names]
    header = (f"Frame {record['frame']} | {record['timestamp_s']:.2f}s" if video else "Image detections")
    
    lines = [
        f"{header} | Overflow masks: {len(record['overflow'])}",
        f"{'Postprocessed counts' if video else 'Detections'}: {sum(counts.values())}",
        " | ".join(items[:4]),
        " | ".join(items[4:])
    ]
    
    panel_height = 12 + 24 * len(lines)
    result[:panel_height] = (result[:panel_height] * 0.35).astype(np.uint8)
    
    for index, line in enumerate(lines):
        cv2.putText(result, line, (8, 22 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return result


class H264Writer:
    def __init__(self, path, fps, shape):
        height, width = shape
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0",
            "-movflags", "+faststart", str(path)
        ]
        self.log = (path.parent / "encoding.log").open("wb")
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        except BaseException:
            self.log.close()
            raise

    def __enter__(self):
        return self

    def write(self, frame):
        self.process.stdin.write(frame.tobytes())

    def __exit__(self, error_type, error, traceback):
        try:
            if error_type is not None:
                self.process.kill()
            with suppress(BrokenPipeError):
                self.process.stdin.close()
            try:
                code = self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("Video encoder did not finish within 60 seconds.")
            if error_type is None and code:
                raise RuntimeError("H.264 encoding failed; inspect encoding.log in the output folder")
            
        finally:
            self.log.close()