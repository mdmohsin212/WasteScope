# WasteScope reports

This folder contains compact evidence from completed experiments. Original dataset archives, images, predictions, weights and videos remain on Drive.

| Area | Evidence |
| --- | --- |
| Detection D0/D1 validation | [Comparison](detection/comparison.md) |
| Detection D1 final test | [Final report](detection/D1_yolo26n_960/test_960/README.md) |
| Detection operating threshold | [Validation selection](detection/D1_yolo26n_960/val_960/confidence_tuning/selected_operating_point.json) |
| Tracking T0–T3 and final test | [Final comparison and frozen settings](tracking/final/README.md) |
| Tracking corrected preparation | [Preparation](tracking/preparation/README.md) |
| Segmentation S0/S1/S2 | [Canonical comparison and final test](segmentation/README.md) |
| GPU performance | [T4 benchmark](benchmarks/README.md) |
| End-to-end demonstrations | [Demo evidence](pipeline/README.md) |
| ONNX export comparison | [Sampled conversion check](onnx_checks/README.md) |

## Final selected systems

| Task | Selected system | Held-out test evidence |
| --- | --- | --- |
| Detection | D1 YOLO26n, 960; confidence > 0.35, oil > 0.20 | Canonical box AP50–95 0.7266; operating macro F1 0.9221 |
| Tracking | T3 BoT-SORT + reconnection and counting rules | HOTA 66.2731%; IDF1 78.3097%; count MAE 0.5714; SAD 8 |
| Overflow segmentation | D1 proposals + S1 full image + S2 ROI, fused; confidence > 0.20 | Canonical mask AP50–95 0.2737; operating F1 0.4760 |

These are single-run task evaluations on their respective task splits, not one common end-to-end accuracy score. Configurations were selected on validation. Detection native Ultralytics and canonical COCO scores are separate; segmentation comparisons use canonical COCO. S0/S1 and S2 used different checkpoint-selection monitors.

## Layout and scope

Per-run training/ contains arguments, training history/plots, available software-environment records and checkpoint-selection evidence. Evaluation folders keep aggregate/per-class/per-video/per-image numeric tables, configuration, frozen policies and non-image-source plots. Tracking has no trainable tracker stage.

Reports are copied from Drive; summaries and navigation added here do not represent reruns. The original absolute paths and environment versions in configurations describe the source runs. Missing environment records are not reconstructed. Training and validation folders keep their original names, including the S2 confidence-0.25 validation folder.

Restricted source-image failure galleries, batch previews, raw masks/predictions, complete ground-truth JSON, full tracks/frame records, checkpoints and annotated videos are omitted. Numeric error evidence remains included. Source code belongs in src/ or scripts/, not reports/.

- [File provenance, source hashes and packaged hashes](sources.csv)
- [Editorial changes and derived summaries](package_notes.json)
