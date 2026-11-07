# S2_yolo26n_seg_roi_960 — frozen final test

System: D1 container proposals + frozen S1 full image + S2 ROI, fused at original image resolution.
S2 checkpoint: epoch 1. Evaluation image size: 960. Images: 1110; instances: 846.
Operating confidence: **> 0.2**, chosen on validation before test. No test tuning.

| Metric | Value |
| --- | --- |
| mask_AP50 | 0.5150 |
| mask_mAP50_95 | 0.2737 |
| mask_AR50_95 | 0.4897 |
| TP | 307.0000 |
| FP | 137.0000 |
| FN | 539.0000 |
| precision | 0.6914 |
| recall | 0.3629 |
| F1 | 0.4760 |
| overall_foreground_IoU | 0.5022 |
| Dice | 0.6686 |
| boundary_IoU_matched | 0.8066 |
| false_positive_mask_area_pct | 0.0299 |
| negative_false_positive_mask_area_pct | 0.0100 |
| missed_overflow_rate_pct | 63.7116 |
| background_images | 306.0000 |
| background_images_with_false_alarms | 53.0000 |
| background_false_alarm_rate_pct | 17.3203 |

## Protocol

Canonical COCO mask AP uses confidence 0.001, IoU 0.50:0.05:0.95, maxDets=[1,10,100].
P/R/F1 and pixel metrics use confidence > 0.2; instance matching uses mask IoU >= 0.50.
Boundary IoU covers matched pairs only. Background false alarm rate counts negative images with at least one FP.
Foreground IoU/Dice pool pixels across the split. AP is independent of the chosen operating threshold.

ROI proposals come from D1 predictions, with the frozen crop policy and no GT proposals or test jitter.
No-container images keep the S1 branch. S1 precedes S2 on score ties in mask-IoU NMS.
S0/S1 used native checkpoint selection; S2 used canonical fused AP. Results represent one run.

## Saved evidence

- [Frozen system and checkpoint hashes](selected_system.json)
- [Metrics and software environments](metrics.json)
- [Per-class results](per_class.csv), [per-video results](per_video.csv), [per-image results](per_image.csv)
- [Experiment summary](experiment_summary.csv): validation comparisons and final test are separate rows.
- [Prediction metadata](prediction_metadata.json). Full predictions and original test ground truth remain on Drive.
- [AP collection timings](ap_collection_times.csv): includes initialization; **not deployment latency**.
- ROI policy and video manifest are included. Evaluator/helper hashes are recorded in selected_system.json; source snapshots remain on Drive and implementation code lives in the repository.

## Error evidence


Numeric per-image and per-video errors document limitations; test results do not change this frozen configuration.
Pipeline and performance evidence is indexed in the reports root README.
S3 is deferred.
