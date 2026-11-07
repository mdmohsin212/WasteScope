# D0_yolo26n_640 — validation

Model: `yolo26n.pt` | Image size: 640
Epochs: 75
Images: 5018 | Instances: 10203

## Overall metrics — Ultralytics

| Run | Split | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| D0_yolo26n_640 | val | 0.8866 | 0.7973 | 0.8533 | 0.6414 |

## Per-class COCO AP/AR

| Class | Instances | AP@0.50 | AP@[.50:.95] | AR@[.50:.95] |
| --- | --- | --- | --- | --- |
| container_yellow | 1351 | 0.7448 | 0.5344 | 0.6942 |
| container_green | 1478 | 0.8867 | 0.6426 | 0.7015 |
| container_default | 4850 | 0.8597 | 0.6182 | 0.6809 |
| container_blue | 1606 | 0.8188 | 0.6064 | 0.7055 |
| container_oil | 249 | 0.8479 | 0.6408 | 0.7016 |
| container_battery | 461 | 0.7888 | 0.5471 | 0.5946 |
| container_biodegradable | 208 | 0.8976 | 0.7018 | 0.7587 |
| All Classes | 10203 | 0.8349 | 0.6130 | 0.6910 |

## Next investigation

| Area | Observed evidence | Next investigation |
| --- | --- | --- |
| Weak classes | container_yellow: 0.534; container_battery: 0.547; container_blue: 0.606 | Inspect FP/FN examples. |
| Object size | Missed: 19.52 px; detected: 33.42 px | Compare sizes within each class. |
| Localization | container_green: gap=0.24 | Inspect box alignment. |
| Background false alarms | 12/2529 images; rate=0.47% | Review false detections. |
| Video variation | Recall: 0.05–0.97 | Review low-recall videos. |

## Evaluation protocol

- Ultralytics: 8.4.1.
- Evaluated checkpoint: `best.pt`; split: validation.
- AP predictions: confidence ≥ 0.001.
- Native confusion matrix: confidence > 0.25, matching IoU > 0.45;
  class-agnostic IoU matching.
- Error CSV and failure gallery: confidence ≥ 0.25,
  matching IoU ≥ 0.50; COCO class-aware, score-ordered matching.
- COCO AP/AR: IoU 0.50:0.05:0.95, maxDets=[1,10,100].
- Results are from one run; no mean ± std is reported.

Confusion-matrix counts and error-CSV counts use different matching protocols.
Ultralytics and COCO AP use separate implementations.

## Figures

- [BoxF1_curve.png](figures/BoxF1_curve.png)
- [BoxPR_curve.png](figures/BoxPR_curve.png)
- [BoxP_curve.png](figures/BoxP_curve.png)
- [BoxR_curve.png](figures/BoxR_curve.png)
- [confusion_matrix.png](figures/confusion_matrix.png)
- [confusion_matrix_normalized.png](figures/confusion_matrix_normalized.png)
- [training_curves.png](figures/training_curves.png)

## Failure evidence

Numeric error tables are included. Source-image galleries remain on Drive.
