from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from .jobs import BusyError, FILES, Jobs
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv"}


def pipeline_factory():
    from ..pipeline.runner import WasteScopePipeline
    from ..pipeline.runtime import Runtime

    runtime = Runtime(device=os.getenv("WASTESCOPE_DEVICE", "cpu"))
    return WasteScopePipeline(runtime)


def create_app(*, output_root=None, factory=pipeline_factory, max_upload_mb=200, jobs_factory=Jobs):
    @asynccontextmanager
    async def lifespan(app):
        root = output_root or os.getenv("WASTESCOPE_OUTPUTS", str(ROOT / "outputs/api"))
        app.state.jobs = jobs_factory(root, factory)
        try:
            yield
        finally:
            app.state.jobs.close()

    app = FastAPI(title="WasteScope",version="1.0.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(WEB / "index.html")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "pipeline_loaded": app.state.jobs.pipeline is not None
        }

    def job_status(job_id):
        try:
            return app.state.jobs.status(job_id)
        except FileNotFoundError:
            raise HTTPException(404, "Job not found.") from None

    @app.post("/jobs", status_code=202)
    def submit(file: UploadFile = File(...)):
        jobs, folder = app.state.jobs, None

        try:
            filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
            suffix = Path(filename).suffix.lower()

            if suffix not in IMAGE_TYPES | VIDEO_TYPES:
                raise HTTPException(415, "Upload JPG, PNG, WEBP, MP4, MOV, AVI or MKV.")

            try:
                folder = jobs.reserve(filename, "image" if suffix in IMAGE_TYPES else "video")
            except BusyError as error:
                raise HTTPException(429, str(error), headers={"Retry-After": "5"}) from None

            source, size = folder / f"input{suffix}", 0
            with source.open("wb") as stream:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, f"Maximum file size is {max_upload_mb} MiB.")
                    stream.write(chunk)

            if not size:
                raise HTTPException(400, "The uploaded file is empty.")

            jobs.submit(folder, source)

        except BaseException:
            if folder is not None:
                jobs.discard(folder)
            raise
        
        finally:
            file.file.close()

        return job_status(folder.name)

    @app.get("/jobs/{job_id}")
    def status(job_id: str):
        return job_status(job_id)

    def completed_file(job_id, name):
        job_status(job_id)
        try:
            path = app.state.jobs.artifact(job_id, name)
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from None
        except FileNotFoundError:
            raise HTTPException(404, "File not found.") from None

        if not path.is_file():
            raise HTTPException(404, "File not found.")
        return path

    @app.get("/jobs/{job_id}/result")
    def result(job_id: str):
        path = completed_file(job_id, "summary.json")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/jobs/{job_id}/files/{name}")
    def download(job_id: str, name: str, inline: bool = False):
        if name not in FILES:
            raise HTTPException(404, "File not found.")

        path = completed_file(job_id, name)
        disposition = "inline" if inline else "attachment"
        return FileResponse(path, filename=name, content_disposition_type=disposition)

    return app

app = create_app()