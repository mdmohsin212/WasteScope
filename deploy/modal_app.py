import json
import modal
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/app"
GPU, DAILY_UPLOADS = "T4", 10

app = modal.App("wastescope")
models = modal.Volume.from_name("wastescope-models-v1", create_if_missing=True, version=2)
data = modal.Volume.from_name("wastescope-jobs-v1", create_if_missing=True, version=2)
states = modal.Dict.from_name("wastescope-jobs-v1", create_if_missing=True)

base = modal.Image.debian_slim(python_version="3.12").env({
    "PYTHONPATH": REMOTE,
    "PYTHONUNBUFFERED": "1",
    "YOLO_CONFIG_DIR": "/tmp/ultralytics",
    "YOLO_AUTOINSTALL": "False"
})

api_image = base.pip_install(
    "fastapi==0.128.8",
    "starlette==0.52.1",
    "python-multipart==0.0.22"
)

gpu_image = (
    base.apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        index_url="https://download.pytorch.org/whl/cu128"
    )
    .add_local_file(ROOT / "requirements.txt", "/tmp/requirements.txt", copy=True)
    .add_local_file(ROOT / "requirements-video.txt", "/tmp/requirements-video.txt", copy=True)
    .run_commands("python -m pip install -r /tmp/requirements-video.txt", "python -m pip check")
)

api_image = api_image.add_local_dir(ROOT / "src", REMOTE + "/src",ignore=["**/__pycache__/**", "**/*.pyc"])
gpu_image = (
    gpu_image
    .add_local_dir(ROOT / "src", REMOTE + "/src", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(ROOT / "configs", REMOTE + "/configs")
    .add_local_file(ROOT / "models/manifest.json", REMOTE + "/models/manifest.json")
)

for filename in ("index.html", "app.css", "app.js"):
    api_image = api_image.add_local_file(ROOT / "web" / filename, REMOTE + "/web/" + filename)
    

@app.function(
    image=gpu_image, gpu=GPU, volumes={"/data": data, "/models": models},
    timeout=15 * 60,
    min_containers=0, max_containers=1, scaledown_window=30
)
def infer(job_id: str, source_name: str):
    from src.pipeline.modal_worker import process
    return process(job_id, source_name, states, data, models)


def get_result(call_id):
    try:
        return modal.FunctionCall.from_id(call_id).get(timeout=0)
    except (
        TimeoutError, 
        modal.exception.ConnectionError,
        modal.exception.InternalError,
        modal.exception.ServiceError
    ):
        return None
    except Exception as error:
        return {
            "state": "failed",
            "error": f"{type(error).__name__}: {error}",
        }


@app.function(
    image=api_image, cpu=0.5, memory=1024,
    min_containers=0, max_containers=1, scaledown_window=30,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()

def web():
    from fastapi.responses import JSONResponse
    from src.api.main import create_app
    from src.api.modal_jobs import ModalJobs

    api = create_app(
        output_root="/tmp/wastescope-api",
        jobs_factory=lambda root, _: ModalJobs(root, states, data, infer, get_result, DAILY_UPLOADS)
    )

    @api.middleware("http")
    async def upload_header_limit(request, call_next):
        if request.method == "POST" and request.url.path == "/jobs":
            length = request.headers.get("content-length", "")
            if not length.isdigit() or int(length) > 201 * 1024 * 1024:
                return JSONResponse({"detail": "Upload a file up to 200 MiB."}, status_code=413)
        return await call_next(request)

    return api


@app.function(
    image=api_image, cpu=0.25, memory=512,
    schedule=modal.Period(hours=6), timeout=300
)
def remove_expired():
    from src.api.modal_jobs import cleanup
    cleanup(states, data)


@app.local_entrypoint()
def main():
    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8-sig"))
    verified = []
    for record in manifest["models"].values():
        path = ROOT / "models/weights" / record["filename"]
        digest = hashlib.sha256()

        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)

        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Checkpoint SHA256 mismatch: {path.name}")
        verified.append(path)

    with models.batch_upload(force=True) as batch:
        for path in verified:
            batch.put_file(path, "/weights/" + path.name)
            print("Verified and staged:", path.name)

    print("Three checkpoints staged. No inference or training was started.")