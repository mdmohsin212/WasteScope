import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest
from fastapi.testclient import TestClient
from src.api.main import create_app


class FakePipeline:
    def __init__(self, gate=None):
        self.gate = gate

    def image(self, source, output_root):
        if self.gate:
            self.gate.wait(timeout=5)
        if source.read_bytes() == b"bad":
            raise ValueError("Cannot decode image.")

        folder = output_root / "run"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text('{"frames": 1}', encoding="utf-8")
        (folder / "annotated.png").write_bytes(b"preview")
        (folder / "status.json").write_text('{"state": "complete"}', encoding="utf-8")
        
    video = image

class ApiTests(unittest.TestCase):
    def terminal(self, client, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = client.get(f"/jobs/{job_id}").json()
            if result["state"] in {"complete", "failed", "interrupted"}:
                return result
            time.sleep(0.01)
        self.fail("The job did not finish.")

    def test_image_and_video_results(self):
        with TemporaryDirectory() as root:
            app = create_app(output_root=root, factory=FakePipeline)
            
            with TestClient(app) as client:
                self.assertFalse(client.get("/health").json()["pipeline_loaded"])
                for filename, mode in (("sample.png", "image"), ("sample.mp4", "video")):
                    response = client.post("/jobs", files={"file": (filename, b"valid")})
                    self.assertEqual(response.status_code, 202)
                    job = self.terminal(client, response.json()["job_id"])
                    
                    self.assertEqual((job["state"], job["mode"]), ("complete", mode))
                    self.assertEqual(client.get(job["result_url"]).json(), {"frames": 1})
                    self.assertEqual(client.get(job["files"]["annotated.png"]).content, b"preview")
                    
                    self.assertEqual(client.get(job["status_url"] + "/files/input.png").status_code, 404)
                    
                self.assertEqual(client.get("/jobs/not-a-job").status_code, 404)

    def test_rejections_failures_and_slot_reuse(self):
        with TemporaryDirectory() as root:
            app = create_app(output_root=root, factory=FakePipeline, max_upload_mb=1)
            
            with TestClient(app) as client:
                for filename, data, expected in (
                    ("a.txt", b"x", 415),
                    ("a.png", b"", 400),
                    ("a.png", b"x" * (1024 * 1024 + 1), 413)
                ):
                    response = client.post("/jobs", files={"file": (filename, data)})
                    self.assertEqual(response.status_code, expected)

                with self.assertLogs(level="ERROR"):
                    response = client.post("/jobs", files={"file": ("a.png", b"bad")})
                    job = self.terminal(client, response.json()["job_id"])

                self.assertEqual(job["state"], "failed")
                self.assertEqual(client.get(job["status_url"] + "/result").status_code, 409)
                response = client.post("/jobs", files={"file": ("a.png", b"valid")})
                self.assertEqual(self.terminal(client, response.json()["job_id"])["state"], "complete")

    def test_queue_limit_and_restart(self):
        gate = Event()
        with TemporaryDirectory() as root:
            app = create_app(output_root=root, factory=lambda: FakePipeline(gate))
            with TestClient(app) as client:
                try:
                    responses = [
                        client.post("/jobs", files={"file": ("a.png", b"valid")})                        
                        for _ in range(3)
                    ]
                    self.assertEqual([r.status_code for r in responses], [202, 202, 429])
                finally:
                    gate.set()

            folder = Path(root) / ("f" * 32)
            folder.mkdir()
            (folder / "job.json").write_text(json.dumps({"job_id": folder.name, "state": "running"}),encoding="utf-8")
            app = create_app(output_root=root, factory=FakePipeline)
            
            with TestClient(app) as client:
                self.assertEqual(client.get(f"/jobs/{folder.name}").json()["state"],"interrupted")
    
    def test_web_ui_and_preview(self):
        with TemporaryDirectory() as root:
            app = create_app(output_root=root, factory=FakePipeline)

            with TestClient(app) as client:
                response = client.get("/")
                self.assertEqual(response.status_code, 200)
                self.assertIn("WasteScope", response.text)

                for filename in ("app.css", "app.js"):
                    self.assertEqual(client.get(f"/static/{filename}").status_code, 200)

                response = client.post("/jobs", files={"file": ("sample.png", b"valid")})
                job = self.terminal(client, response.json()["job_id"])
                url = job["files"]["annotated.png"]

                self.assertIn("attachment", client.get(url).headers["content-disposition"])
                self.assertIn("inline", client.get(url + "?inline=true").headers["content-disposition"])
                self.assertEqual(client.get(url, headers={"Range": "bytes=0-2"}).status_code, 206)
                

if __name__ == "__main__":
    unittest.main()