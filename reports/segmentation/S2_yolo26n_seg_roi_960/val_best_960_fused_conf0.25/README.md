# S2_yolo26n_seg_roi_960 — val / fused

- Checkpoint: `best.pt`; selected epoch: 1.
- Original images: 1030; GT instances: 861.
- Canonical Mask AP50–95: 0.3650; AP50: 0.6649.
- Supporting confidence: >0.25; matching mask IoU: 0.5.
- TP / FP / FN: 444 / 115 / 417.
- Foreground IoU: 0.3725; Dice: 0.5428.
- Mean inference latency (ms): Not measured.
- Checkpoint selection: canonical full-original-image S1+ROI mask AP50-95.
- Selection limitation: S0/S1 used native checkpoint selection; S2 uses canonical fused AP.

AP uses pycocotools COCOeval, confidence floor 0.001, IoU .50:.05:.95 and maxDets=[1,10,100].
Boundary IoU averages matched instances. These are single-run val results.
Confidence sweeps run on validation only; the report above keeps confidence 0.25.
Source-image examples in `private/` remain private.

[Metrics](metrics.csv) · [Per-video errors](per_video.csv) · [Confidence sweep](confidence_sweep.csv)
