import logging
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from ..api.jobs import FILES
from .outputs import STATUS_SINK

_pipeline = None

def validate_media(source, mode):
    from PIL import Image
    if mode == "image":
        with Image.open(source) as image:
            if image.width * image.height > 4_000_000:
                raise ValueError("Images must be at most 4 megapixels.")
            image.verify()
        return

    import cv2
    import math
    capture = cv2.VideoCapture(str(source))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError("The video has no valid FPS metadata.")

        frames = 0
        while True:
            success, image = capture.read()
            if not success:
                break

            frames += 1
            h, w = image.shape[:2]

            if h * w > 1920 * 1080 or max(h, w) > 1920:
                raise ValueError("Video resolution must be at most 1080p.")
            if frames > 300 or frames / fps > 30:
                raise ValueError("Videos must be at most 30 seconds and 300 frames.")

        if not frames:
            raise ValueError("No video frames could be decoded.")
    finally:
        capture.release()


def process(job_id, source_name, states, data_volume, model_volume):
    global _pipeline
    job = states[job_id]

    def publish(**fields):
        job.update(fields, updated_at=time.time())
        states[job_id] = job.copy()

    token = None
    try:
        if job["expires_at"] <= time.time():
            raise ValueError("This job has expired.")

        publish(state="running", progress={"phase": "setup", "frames_processed": 0})
        data_volume.reload()

        with TemporaryDirectory(prefix="wastescope-") as temporary:
            source = Path(temporary) / source_name
            shutil.copy2(Path("/data/jobs") / job_id / source_name, source)
            validate_media(source, job["mode"])

            if _pipeline is None:
                import torch
                from .runner import WasteScopePipeline
                from .runtime import Runtime

                torch.set_num_threads(4)
                model_volume.reload()
                runtime = Runtime(models_dir="/models/weights", device="cuda:0")
                _pipeline = WasteScopePipeline(runtime)

            token = STATUS_SINK.set(lambda progress: publish(progress=progress))
            summary = getattr(_pipeline, job["mode"])(source, output_root=Path(temporary) / "results")

            folder = Path(summary["output_dir"])
            destination = Path("/data/jobs") / job_id / "results"
            shutil.copytree(folder, destination, dirs_exist_ok=True)
            data_volume.commit()

            publish(
                state="complete",
                files=sorted(name for name in FILES if (folder / name).is_file()),
                expires_at=time.time() + 24 * 60 * 60
            )
    except Exception as error:
        logging.exception("Cloud inference job %s failed", job_id)
        publish(state="failed", error=f"{type(error).__name__}: {error}")
        
    finally:
        if token is not None:
            STATUS_SINK.reset(token)

    return job