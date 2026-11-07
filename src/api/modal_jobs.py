import re
import time
import shutil
from uuid import uuid4
from pathlib import Path
from threading import Lock
from datetime import datetime, timezone
from .jobs import BusyError, FILES

RETENTION = 24 * 60 * 60

class ModalJobs:
    pipeline = None
    def __init__(self, root, states, volume, worker,
        get_result, daily_limit=10):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.states, self.volume, self.worker = states, volume, worker
        self.get_result, self.daily_limit = get_result, daily_limit
        self.lock, self.download_lock = Lock(), Lock()
        
    def read(self, job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise FileNotFoundError(job_id)

        job, now = self.states.get(job_id), time.time()
        if not job or job["expires_at"] <= now:
            raise FileNotFoundError(job_id)
        
        if job["state"] == "uploading" and now - job["updated_at"] > 300:
            job.update(state="failed", error="Upload did not finish.", updated_at=now)
            self.states[job_id] = job
            
            call_id = self.states.get("call:" + job_id)
            if (call_id and job["state"] in {"queued", "running"} and now - job["updated_at"] > 30):
                result = self.get_result(call_id)
            if result:
                job.update(result, updated_at=now)
                self.states[job_id] = job

        return job
    
    def reserve(self, filename, mode):
        with self.lock:
            active = []
            for job_id in self.states.get("_active", []):
                try:
                    if self.read(job_id)["state"] in {"uploading", "queued", "running"}:
                        active.append(job_id)
                except FileNotFoundError:
                    pass

            if len(active) >= 2:
                raise BusyError("One job is running and another is waiting; Try again later.")

            day = datetime.now(timezone.utc).date().isoformat()
            usage = self.states.get("_usage", {"day": day, "count": 0})
            used = usage["count"] if usage["day"] == day else 0

            if used >= self.daily_limit:
                raise BusyError("The demo's daily upload limit has been reached; Please try again later.")

            folder = self.root / uuid4().hex
            folder.mkdir()
            now = time.time()

            self.states[folder.name] = {
                "job_id": folder.name,
                "filename": filename,
                "mode": mode,
                "state": "uploading",
                "updated_at": now,
                "expires_at": now + RETENTION
            }
            self.states["_active"] = active + [folder.name]
            self.states["_usage"] = {"day": day, "count": used + 1}
            return folder

    def discard(self, folder):
        job = self.states.get(folder.name)
        if job and job["state"] == "uploading":
            job.update(state="failed", error="Upload was rejected or interrupted.")
            self.states[folder.name] = job
        shutil.rmtree(folder, ignore_errors=True)

    def submit(self, folder, source):
        try:
            remote = f"/jobs/{folder.name}/{source.name}"
            with self.volume.batch_upload() as batch:
                batch.put_file(source, remote)

            job = self.read(folder.name)
            job.update(state="queued", updated_at=time.time())
            self.states[folder.name] = job

            call = self.worker.spawn(folder.name, source.name)
            self.states["call:" + folder.name] = call.object_id
            
        except Exception:
            job = self.states.get(folder.name)
            if job and job["state"] in {"uploading", "queued"}:
                job.update(state="failed", error="Could not submit the cloud job.")
                self.states[folder.name] = job
            raise
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def status(self, job_id):
        job = self.read(job_id).copy()
        job["status_url"] = f"/jobs/{job_id}"

        if job["state"] == "complete":
            job["result_url"] = f"/jobs/{job_id}/result"
            job["files"] = {
                name: f"/jobs/{job_id}/files/{name}"
                for name in job["files"]
                if name in FILES
            }
        return job

    def artifact(self, job_id, name):
        job = self.read(job_id)
        if job["state"] != "complete":
            raise RuntimeError("The job has not completed successfully.")
        if name not in FILES or name not in job["files"]:
            raise FileNotFoundError(name)

        with self.download_lock:
            path = self.root / job_id / name
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                pending = path.with_suffix(path.suffix + ".part")
                
                try:
                    remote = f"/jobs/{job_id}/results/{name}"
                    with pending.open("wb") as stream:
                        for block in self.volume.read_file(remote):
                            stream.write(block)
                    pending.replace(path)
                    
                finally:
                    pending.unlink(missing_ok=True)
            return path

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def cleanup(states, volume):
    for job_id, job in list(states.items()):
        if not re.fullmatch(r"[0-9a-f]{32}", str(job_id)):
            continue
        if job["expires_at"] > time.time():
            continue

        try:
            volume.remove_file(f"/jobs/{job_id}", recursive=True)
        except FileNotFoundError:
            pass

        states.pop(job_id, None)
        states.pop("call:" + job_id, None)