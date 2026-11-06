from contextlib import suppress
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
from uuid import uuid4

PACKAGES = [
    "torch", "ultralytics", "numpy", "opencv-python",
    "pycocotools", "lap", "imageio-ffmpeg"
]

def _now():
    return datetime.now(timezone.utc).isoformat()

def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RunOutput:
    def __init__(self, runtime, source, output_root, mode, configs, verbose):
        self.started = time.perf_counter()
        self.source = Path(source).expanduser().resolve()
        if not self.source.is_file():
            raise FileNotFoundError(self.source)
            
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.source.stem)[:48]
            
        self.folder = Path(output_root).expanduser().resolve() / f"{stem}_{stamp}_{uuid4().hex[:8]}"
        self.folder.mkdir(parents=True)
        self.verbose, self.last_update = verbose, 0.0
        self.mode = mode
        self.progress("setup", 0, force=True)
            
        try:
            effective = json.loads(json.dumps(configs))
            for config in effective.values():
                for key in ("inference", "proposal_inference", "segment_inference"):
                    if key in config:
                        config[key]["half"] = bool(
                            config[key].get("half") and  runtime.device.type == "cuda"
                        )
                
            write_json(self.folder / "run_config.json", {
            "schema_version": 1,
            "release_id": runtime.manifest["release_id"],
            "mode": mode,
            "input": str(self.source),
            "input_sha256": file_sha256(self.source),
            "created_utc": _now(),
            "device": str(runtime.device),
            "model_manifest": runtime.manifest,
            "effective_configs": effective,
            "environment": {
                "python": platform.python_version(),
                **{name: version(name) for name in PACKAGES}
                }
            })
        
        except BaseException as error:
            self.fail(error)
            raise
    
    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        if error is not None:
            self.fail(error)

    def _status(self, **fields):
        write_json(self.folder / "status.json", {**fields, "updated_utc": _now()})
            
    def progress(self, phase, frames, total=None, *, force=False):
        now = time.monotonic()
        if force or now - self.last_update >= 2:
            self._status(state="running", phase=phase, frames_processed=frames, total_frames=total)
            if self.verbose:
                print(f"{phase}: {frames}/{'?' if total is None else total} frames | {self.folder}", flush=True)
                
            self.last_update = now
            
    def complete(self, summary):
        seconds = time.perf_counter() - self.started
        summary.update(
            schema_version=1,
            status="complete",
            mode=self.mode,
            input=str(self.source),
            output_dir=str(self.folder),
            end_to_end_seconds=seconds,
            timing_scope="Local file through saved outputs; model loading included when cold."
        )
        if self.mode == "video":
            summary["end_to_end_fps"] = summary["frames"] / max(seconds, 1e-9)
            
        write_json(self.folder / "summary.json", summary)
        self._status(state="complete", phase="complete", frames_processed=summary["frames"], total_frames=summary["frames"])
        return summary
    
    def fail(self, error):
        with suppress(OSError):
            self._status(
                state="cancelled" if isinstance(error, KeyboardInterrupt) else "failed", error_type=type(error).__name__, error=str(error)
            )