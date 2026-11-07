from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.api.modal_jobs import ModalJobs, cleanup
from src.api.jobs import BusyError
from src.pipeline.outputs import STATUS_SINK, RunOutput


class States(dict):
    def __getitem__(self, key):
        return deepcopy(super().__getitem__(key))

    def __setitem__(self, key, value):
        super().__setitem__(key, deepcopy(value))

    def get(self, key, default=None):
        return deepcopy(super().get(key, default))


class Volume:
    def __init__(self):
        self.files, self.reads = {}, []

    def batch_upload(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def put_file(self, source, remote):
        self.files[remote] = source.read_bytes()

    def read_file(self, remote):
        self.reads.append(remote)
        yield self.files[remote]

    def remove_file(self, prefix, recursive=False):
        self.files = {
            k: v for k, v in self.files.items()
            if not k.startswith(prefix + "/")
        }


class ModalJobTests(unittest.TestCase):
    def test_outputs_survive_api_restart(self):
        states, volume = States(), Volume()

        def spawn(job_id, source_name):
            job = states[job_id]
            job.update(state="complete", files=["summary.json", "annotated.mp4"])
            states[job_id] = job
            prefix = f"/jobs/{job_id}/results/"
            volume.files[prefix + "summary.json"] = b'{"frames":24}'
            volume.files[prefix + "annotated.mp4"] = b"video"
            return SimpleNamespace(object_id="fc-test")

        worker = SimpleNamespace(spawn=spawn)
        with TemporaryDirectory() as root:
            factory = lambda path, _: ModalJobs(path, states, volume, worker, lambda _: None)
            app = create_app(output_root=Path(root) / "first", jobs_factory=factory)
            
            with TestClient(app) as client:
                response = client.post("/jobs", files={"file": ("clip.mp4", b"input")})
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["job_id"]

            app = create_app(output_root=Path(root) / "second", jobs_factory=factory)
            
            with TestClient(app) as client:
                job = client.get(f"/jobs/{job_id}").json()
                
                self.assertEqual(job["state"], "complete")
                self.assertEqual(client.get(job["result_url"]).json(), {"frames": 24})
                self.assertEqual(len(volume.reads), 1)
                
                client.get(job["result_url"])
                self.assertEqual(len(volume.reads), 1)
                self.assertEqual(client.get(job["files"]["annotated.mp4"] + "?inline=true").content, b"video")

    def test_queue_daily_limit_and_abandoned_upload(self):
        with TemporaryDirectory() as root:
            states = States()
            jobs = ModalJobs(
                root, states, Volume(), None,
                lambda _: None, daily_limit=3,
            )
            first = jobs.reserve("a.png", "image")
            second = jobs.reserve("b.png", "image")

            with self.assertRaises(BusyError):
                jobs.reserve("c.png", "image")

            job = states[first.name]
            job["updated_at"] -= 301
            states[first.name] = job

            third = jobs.reserve("c.png", "image")
            self.assertEqual(states[first.name]["state"], "failed")
            jobs.discard(second)
            jobs.discard(third)

            with self.assertRaisesRegex(BusyError, "daily"):
                jobs.reserve("d.png", "image")

    def test_failed_call_and_expiration_cleanup(self):
        with TemporaryDirectory() as root:
            states, volume = States(), Volume()
            jobs = ModalJobs(
                root, states, volume, None,
                lambda _: {"state": "failed", "error": "Timeout"},
            )
            folder = jobs.reserve("a.png", "image")
            job = states[folder.name]
            job.update(state="running", updated_at=time.time() - 31)
            states[folder.name] = job
            states["call:" + folder.name] = "fc-test"

            self.assertEqual(jobs.status(folder.name)["state"], "failed")
            job = states[folder.name]
            job["expires_at"] = time.time() - 1
            states[folder.name] = job
            volume.files[f"/jobs/{folder.name}/input.png"] = b"input"

            with self.assertRaises(FileNotFoundError):
                jobs.status(folder.name)

            cleanup(states, volume)
            self.assertNotIn(folder.name, states)
            self.assertFalse(volume.files)

    def test_progress_preserves_disk_status(self):
        with TemporaryDirectory() as root:
            run = RunOutput.__new__(RunOutput)
            run.folder = Path(root)
            seen = []
            token = STATUS_SINK.set(seen.append)

            try:
                run._status(state="running", frames_processed=4)
            finally:
                STATUS_SINK.reset(token)

            saved = json.loads((run.folder / "status.json").read_text())
            self.assertEqual(seen[0], saved)
            self.assertIsNone(STATUS_SINK.get())

if __name__ == "__main__":
    unittest.main()