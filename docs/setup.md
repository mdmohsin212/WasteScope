# Setup, inference and deployment

Run commands from the repository root. Examples use Windows PowerShell and Python 3.12. Linux users can use `python3.12 -m venv venv` and `source venv/bin/activate`.

## Install

```powershell
git clone https://github.com/mdmohsin212/WasteScope.git
cd WasteScope
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-api.txt
python -m pip check
```

For a compatible NVIDIA GPU, replace the CPU PyTorch install with:

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

`requirements-api.txt` includes base inference and video dependencies. Evaluation and ONNX dependencies are optional.

## Selected checkpoints

The three files under `models/weights/` are `D1_yolo26n_960.pt`, `S1_yolo26n_seg_960.pt` and `S2_yolo26n_seg_roi_960.pt`. Their expected hashes and source paths are in `models/manifest.json`. Authorized access is required.

Use an existing rclone remote:

```powershell
python scripts/setup_models.py --remote gdrive:WasteScope --rclone-config "C:\private\rclone.conf"
```

Or use a local/mounted WasteScope folder containing the original `runs/` hierarchy:

```powershell
python scripts/setup_models.py --source-root "D:\WasteScope"
```

For Colab, use `/content/drive/MyDrive/WasteScope`. Existing matching files are reused. Then verify:

```powershell
python scripts/setup_models.py --check --smoke-test
```

The smoke test checks hashes, class/task order and tiny CPU forwards, not accuracy. Keep credentials outside Git; do not replace expected hashes to accept other checkpoints.

## CLI and outputs

```powershell
python scripts/infer.py image --input "C:\media\image.jpg" --device cpu
python scripts/infer.py video --input "C:\media\clip.mp4" --device cpu
```

Use `--device cuda:0` on a configured GPU, or `--device auto`. CPU uses full precision. Optional `--models-dir` and `--output-root` select other folders. Short media is more practical on CPU.

| Output | Image | Video |
| --- | --- | --- |
| Annotated media | `annotated.png` | `annotated.mp4`, H.264 without audio |
| Predictions | `predictions.json` | `frames.jsonl`, `tracks.csv` |
| Run evidence | `summary.json`, `run_config.json`, `status.json` | Same, plus `encoding.log` |

Video tracking resets per input. Finalized IDs/counts are applied during a second decoding/rendering pass. Odd frame dimensions receive right/bottom padding for encoding; predictions retain original coordinates. CLI outputs go into separate run folders under `outputs/` by default.

## Local UI and API

```powershell
$env:WASTESCOPE_DEVICE = "cpu"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> or `/docs`. Models load on the first job. Use one server worker: local jobs use an in-process queue. `WASTESCOPE_OUTPUTS` optionally changes API storage.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness; `pipeline_loaded` does not establish GPU/model readiness |
| `POST /jobs` | Multipart field `file`; HTTP 202 with job/status links |
| `GET /jobs/{job_id}` | State/progress; result and artifact links when complete |
| `GET /jobs/{job_id}/result` | Final summary |
| `GET /jobs/{job_id}/files/{name}` | Allowed artifact; `?inline=true` enables preview |

```powershell
curl.exe -X POST "http://127.0.0.1:8000/jobs" -F "file=@C:\media\image.jpg"
```

Poll the returned status link until complete, then follow its result/file links. HTTP 429 means queue/budget limits; 413 means file size; 415 means unsupported extension. Decode/media-limit errors may appear as failed jobs after acceptance. Local jobs do not survive server shutdown.

## Modal deployment

```powershell
python -m pip install modal==1.6.3
python -m modal setup
python -m modal run deploy/modal_app.py
python -m modal deploy deploy/modal_app.py
```

The `run` entrypoint verifies/uploads the three local checkpoints to the model volume. It does not train or run a prediction. Repeat staging only if the volume is missing or checkpoints intentionally change. For code updates, use only the `deploy` command. Open the `web` URL printed by deployment; it belongs to the active workspace/environment.

```powershell
python -m modal app logs wastescope
python -m modal app dashboard wastescope
```

The API stores job state in Modal Dict and starts a background GPU call; media/results and weights use separate volumes. Artifacts are committed before completion is published. Both web/GPU functions have zero minimum containers, a maximum of one each, and a configured 30-second idle window. Cold starts are expected. The T4 call timeout is 15 minutes.

See the [README](../README.md#hosted-demo) for media and daily limits. The daily budget counts reservations, including rejected attempts. Results expire after 24 hours; physical removal can wait until the next 6-hour cleanup pass. The general CLI does not enforce these cloud-demo media caps.

## Benchmarks and ONNX

```powershell
python scripts/benchmark.py --input "C:\media\clip.mp4" --device cpu --max-frames 100 --warmup 5
```

Use `--device cuda:0` for GPU measurement. Match clip/device/timing scope when comparing results. Warmed processing FPS differs from rendered end-to-end FPS. [Recorded benchmark](../reports/benchmarks/README.md).

Optional CPU FP32 export and sampled comparison:

```powershell
python -m pip install -r requirements-onnx.txt
python scripts/export_onnx.py
python scripts/check_onnx.py --input "C:\media\clip.mp4"
```

These tools do not switch the deployed PyTorch runtime to ONNX. Sampled conversion success does not establish full accuracy or tracking parity.

## Training and evaluation

Use the task notebooks in `notebooks/` and S2 training/monitor files in `scripts/segmentation/`. Historical notebooks contain cloud paths and task-specific dependencies. Optional tracking evaluation packages: `python -m pip install -r requirements-evaluation.txt`.

Follow the checkpoint identities, split manifests and policies in [reports](../reports/README.md). Select policies on validation; do not retune on the held-out test.

## Verification and scope

```powershell
python -m compileall -q src scripts
python -m unittest discover -s tests -p "test_*.py" -v
```

The CPU suite covers setup integrity, jobs, cloud adapters, geometry, outputs/count consistency, encoding and failure handling. Pipeline/API tests mock model adapters; passing them does not establish model accuracy or GPU numerical parity.

The selected checkpoints passed owner-run CPU forward checks. Reports contain the sampled CPU FP32 ONNX check and T4 benchmark/demos; the owner confirmed deployed inference works. Full frozen-validation replay in the deployed environment and full-sequence backend parity remain follow-up validation. Manifest config/environment text hashes normalize UTF-8 without BOM and LF line endings; checkpoint hashes always cover original binary bytes.

## Troubleshooting

| Problem | Action |
| --- | --- |
| Missing/wrong checkpoint | Stage the selected weights and verify again |
| `rclone` not found | Install it on PATH or use `--source-root` |
| Slow first job | Check status/logs; cold initialization and CPU inference take time |
| Detection but no count | Eligibility requires 10 observed frames after reconnection |
| Invalid video/FPS | Re-encode with valid constant-FPS metadata |
| Encoding failure | Inspect the run's `encoding.log` |
| Cloud result unavailable | Download results before their expiration |