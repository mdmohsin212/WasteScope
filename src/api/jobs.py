from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import shutil
from threading import BoundedSemaphore, Lock
from uuid import uuid4
from ..pipeline.outputs import write_json

FILES = {
    "annotated.png", "annotated.mp4", "predictions.json", "frames.jsonl",
    "tracks.csv", "summary.json", "run_config.json", "status.json", "encoding.log"
}

class BusyError(Exception):
    pass

class Jobs:
    def __init__(self, root, pipeline_factory):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.factory, self.pipeline = pipeline_factory, None
        self.slots, self.lock = BoundedSemaphore(2), Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        
        for path in self.root.glob("*/job.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            
            if job["state"] in {"uploading", "queued", "running"}:
                self._update(path.parent, state="interrupted", error="Server stopped before completion." )
                
    def folder(self, job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise FileNotFoundError(job_id)
        
        folder = self.root / job_id
        if not (folder / "job.json").is_file():
            raise FileNotFoundError(job_id)
        
        return folder

    def read(self, job_id):
        return json.loads((self.folder(job_id) / "job.json").read_text(encoding="utf-8"))
    
    def _update(self, folder, **fields):
        with self.lock:
            path = folder / "job.json"
            job = (json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})
            job.update(fields, updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(path, job)
    
    def reserve(self, filename, mode):
        if not self.slots.acquire(blocking=False):
            raise BusyError("One job is running and another is waiting. Try again later")
        
        folder = self.root / uuid4().hex
        try:
            folder.mkdir()
            self._update(folder, job_id=folder.name, filename=filename, mode=mode, state="uploading")
            return folder
        except BaseException:
            self.discard(folder)
            raise
        
    def discard(self, folder):
        try:
            shutil.rmtree(folder, ignore_errors=True)
        finally:
            self.slots.release()
    
    def submit(self, folder, source):
        self._update(folder, state="queued")
        self.executor.submit(self._run, folder, source)
        
    def _run(self, folder, source):
        try:
            self._update(folder, state="running")
            if self.pipeline is None:
                self.pipeline = self.factory()
            
            mode = self.read(folder.name)["mode"]
            getattr(self.pipeline, mode)(source, output_root=folder / "results")
            self._update(folder, state="complete")
            
        except Exception as error:
            logging.exception("Inference job %s failed", folder.name)
            self._update(folder, state="failed", error=f"{type(error).__name__}: {error}")
            
        finally:
            self.slots.release()
    
    def result_folder(self, job_id):
        if self.read(job_id)["state"] != "complete":
            raise RuntimeError("The job has not completed successfully.")

        folders = list((self.folder(job_id) / "results").iterdir())
        if len(folders) != 1 or not folders[0].is_dir():
            raise RuntimeError("The completed output folder is missing or ambiguous")
        
        return folders[0]

    def status(self, job_id):
        job = self.read(job_id)
        statuses = list((self.folder(job_id) / "results").glob("*/status.json"))

        if statuses:
            job["progress"] = json.loads(statuses[0].read_text(encoding="utf-8"))

        job["status_url"] = f"/jobs/{job_id}"
        
        if job["state"] == "complete":
            folder = self.result_folder(job_id)
            job["result_url"] = f"/jobs/{job_id}/result"
            job["files"] = {
                name: f"/jobs/{job_id}/files/{name}"
                for name in sorted(FILES)
                if (folder / name).is_file()
            }
        return job

    def close(self):
        self.executor.shutdown(wait=True)