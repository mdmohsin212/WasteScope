# Setup and usage

Run WasteScope locally to analyze images and videos, or try the [live demo](https://mohsin416m--wastescope-web.us-east.modal.run).

Examples use **Windows PowerShell and Python 3.12**. Run all commands from the repository root.

## 1. Install

```powershell
git clone https://github.com/mdmohsin212/WasteScope.git
cd WasteScope

py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-api.txt
python -m pip check
```

This installs the CPU version, including the web interface and video dependencies.

On Linux, create and activate the environment with:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

<details>
<summary>Using an NVIDIA GPU?</summary>

In the installation commands above, replace the CPU PyTorch command with:

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

A compatible NVIDIA GPU and driver are required. Use `--device cuda:0` for CLI inference.

For the web interface, set this before starting the server:

```powershell
$env:WASTESCOPE_DEVICE = "cuda:0"
```

</details>

## 2. Add the trained models

The checkpoints are stored separately from GitHub. You need authorized access to the selected weights.

With an existing rclone remote and rclone installed on PATH:

```powershell
python scripts/setup_models.py --remote gdrive:WasteScope --rclone-config "C:\private\rclone.conf"
```

Replace the remote and config path with your own.

<details>
<summary>Using a local folder or mounted Google Drive instead?</summary>

Point to the WasteScope folder containing the original `runs/` hierarchy:

```powershell
python scripts/setup_models.py --source-root "D:\WasteScope"
```

For mounted Google Drive in Colab, use `/content/drive/MyDrive/WasteScope`.

</details>

Then verify the models:

```powershell
python scripts/setup_models.py --check --smoke-test
```

Setup places these files in `models/weights/`:

- `D1_yolo26n_960.pt`
- `S1_yolo26n_seg_960.pt`
- `S2_yolo26n_seg_roi_960.pt`

Existing matching files are reused. Setup verifies checkpoint hashes and performs small CPU forward checks. It does not download datasets or train models.

Checkpoint identities are recorded in `models/manifest.json`. Keep rclone credentials outside Git.

## 3. Start the app

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**, upload an image or short video, and select **Analyze media**.

The app uses CPU by default. Models load on the first prediction, so the first job can take longer. Keep the terminal open while using the local app, and use one server worker.

API documentation is available at **http://127.0.0.1:8000/docs**.

## Run from the command line

```powershell
python scripts/infer.py image --input "C:\media\image.jpg" --device cpu
python scripts/infer.py video --input "C:\media\clip.mp4" --device cpu
```

For GPU inference, replace `cpu` with `cuda:0`. Use short clips when running on CPU.

Each CLI run creates a separate folder under `outputs/`.

| Output | Image | Video |
| --- | --- | --- |
| Annotated media | `annotated.png` | `annotated.mp4` |
| Predictions | `predictions.json` | `frames.jsonl`, `tracks.csv` |
| Run details | `summary.json`, `run_config.json`, `status.json` | Same, plus `encoding.log` |

Video output uses H.264 without audio. Tracking resets for each video, and final IDs/counts are applied after the whole clip is processed.

Local web/API results are stored under `outputs/api/`. Downloads are available through the interface.

## Deploy on Modal

Complete the model setup above before the first deployment.

### First deployment

```powershell
python -m pip install modal==1.6.3
python -m modal setup

python -m modal run deploy/modal_app.py
python -m modal deploy deploy/modal_app.py
```

The `run` command verifies and uploads the three checkpoints to the Modal model volume. The `deploy` command publishes the application.

Open the **web URL** printed by deployment.

### Update an existing deployment

After changing application code:

```powershell
python -m modal deploy deploy/modal_app.py
```

Repeat model staging only when the model volume needs to be populated or the selected checkpoints intentionally change.

### View logs

```powershell
python -m modal app logs wastescope
python -m modal app dashboard wastescope
```

The current deployment uses an **L4 GPU**. Web and GPU functions scale to zero when idle, so cold starts can delay the first visit or prediction. The hosted app runs independently of your local PC.

Cloud results expire after 24 hours. See [hosted-demo limits](../README.md#hosted-demo) before uploading media.

## Optional tools

<details>
<summary>Use the API directly</summary>

Submit a file:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/jobs" -F "file=@C:\media\image.jpg"
```

Follow the returned status link until the job completes, then use its result and download links.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Check whether the API is running |
| `POST /jobs` | Submit an image or video |
| `GET /jobs/{job_id}` | Check progress |
| `GET /jobs/{job_id}/result` | Get the final summary |
| `GET /jobs/{job_id}/files/{name}` | Download an output file |

Local jobs do not survive server shutdown. A successful health response does not verify model or GPU readiness.

</details>

<details>
<summary>Benchmark inference</summary>

```powershell
python scripts/benchmark.py --input "C:\media\clip.mp4" --device cpu --max-frames 100 --warmup 5
```

Use `--device cuda:0` for GPU measurement. Compare results using the same input, device and timing scope.

See [recorded benchmarks](../reports/benchmarks/README.md) for existing results and measurement details.

</details>

<details>
<summary>Export and check ONNX models</summary>

```powershell
python -m pip install -r requirements-onnx.txt
python scripts/export_onnx.py
python scripts/check_onnx.py --input "C:\media\clip.mp4"
```

These tools perform CPU FP32 export and sampled comparison. They do not change the deployed PyTorch runtime or establish full accuracy/tracking parity.

</details>

<details>
<summary>Run tests</summary>

```powershell
python -m compileall -q src scripts
python -m unittest discover -s tests -p "test_*.py" -v
```

The CPU tests cover application behavior and failure handling. Pipeline/API tests use mocked model adapters; model accuracy is documented separately in [experiment reports](../reports/README.md).

</details>

<details>
<summary>Training and evaluation</summary>

Use the task notebooks under `notebooks/` and the S2 training helpers under `scripts/segmentation/`.

For tracking evaluation dependencies:

```powershell
python -m pip install -r requirements-evaluation.txt
```

Notebooks contain environment-specific paths. Follow the split manifests, checkpoints and policies in [experiment reports](../reports/README.md). Select settings on validation data before evaluating on test data.

</details>

## Troubleshooting

| Problem | What to do |
| --- | --- |
| Missing or incorrect weights | Repeat model setup, then run `--check --smoke-test` |
| `rclone` not found | Install rclone on PATH or use `--source-root` |
| Slow first prediction | Allow time for model loading; CPU video inference is slower |
| Detections appear but counts stay zero | Counting requires 10 observed frames after reconnection |
| Video cannot be processed | Re-encode it with valid constant-FPS metadata |
| Video encoding fails | Check `encoding.log` in the run folder |
| Upload rejected | Check file type, size and hosted-demo limits |
| Cloud result expired | Run the prediction again and download the outputs before expiry |