# WasteScope — setup

Use Python 3.12. Run commands from the repository root.

## Windows CPU environment

These commands do not require virtual-environment activation:

```powershell
py -3.12 -m venv .venv

.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m pip check
```

Optional evaluation dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt
```

## Linux GPU environment

Use a compatible NVIDIA driver:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt
python -m pip check
```

The common environment is verified on Linux CPU. GPU inference parity
and the final CUDA deployment lock remain to be verified.

## Prepare weights

Choose one source.

### Existing rclone installation

Use your actual rclone.conf path:

```bash
python scripts/setup_models.py --remote gdrive:WasteScope --rclone-config scripts/segmentation/rclone.conf
```

The script does not install rclone or create credentials.

### Mounted Google Drive

```bash
python scripts/setup_models.py --source-root /content/drive/MyDrive/WasteScope
```

### Manual download

Download only these three files from Drive and rename them:

| Drive source | Destination in repository |
| --- | --- |
| runs/detect/D1_yolo26n_960/weights/best.pt | models/weights/D1_yolo26n_960.pt |
| runs/segment/S1_yolo26n_seg_960/weights/best.pt | models/weights/S1_yolo26n_seg_960.pt |
| runs/segment/S2_yolo26n_seg_roi_960/weights/best.pt | models/weights/S2_yolo26n_seg_roi_960.pt |

## Verify models

```bash
python scripts/setup_models.py --check --smoke-test
```

On Windows without environment activation:

```powershell
.\.venv\Scripts\python.exe scripts/setup_models.py --check --smoke-test
```

This verifies hashes, tasks, class order and a small CPU forward pass.
It does not measure production accuracy or latency.

Existing matching weights are reused. A downloaded file replaces its
destination only after its SHA256 matches.

## Frozen task behaviour

- Detection uses rect=True. Final scores must exceed 0.35,
  except oil scores must exceed 0.20.
- Tracking uses the detector output before the detection operating-point
  filter. Counts require 10 distinct observed frames after same-class
  reconnection, with max_gap=5 and max_distance=0.05.
- Duration filtering affects counting only. Finalize counts at the end
  of each video and reset tracker state between videos.
- Segmentation proposals use rect=False and confidence 0.05.
- S1/S2 segmentation use retina masks and the recorded ROI geometry.
- Fused masks must exceed confidence 0.20.
- Images without proposals retain the S1 full-image branch.
- half=True records GPU reference precision; CPU execution uses FP32.

Keep the separate inference profiles until validation establishes parity
for any shared-inference optimization.

## Offline setup tests

```bash
python -m unittest discover -s tests -p test_model_setup.py -v
```

See phase1-verification.md for the completed checks.