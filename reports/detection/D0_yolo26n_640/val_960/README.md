# D0_yolo26n_640 — validation at 960

Model: `yolo26n.pt`
Training image size: 640 | Evaluation image size: 960
Training epochs: 75
Images: 5018 | Instances: 10203

## Overall metrics — Ultralytics

| Run | Split | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| D0_yolo26n_640 | val | 0.8759 | 0.8300 | 0.8804 | 0.6615 |

## Per-class COCO AP/AR

| Class | Instances | AP@0.50 | AP@[.50:.95] | AR@[.50:.95] |
| --- | --- | --- | --- | --- |
| container_yellow | 1351 | 0.8161 | 0.5801 | 0.7241 |
| container_green | 1478 | 0.8820 | 0.6527 | 0.7186 |
| container_default | 4850 | 0.8767 | 0.6347 | 0.7035 |
| container_blue | 1606 | 0.8335 | 0.6117 | 0.7128 |
| container_oil | 249 | 0.8248 | 0.6161 | 0.7189 |
| container_battery | 461 | 0.9075 | 0.6108 | 0.6646 |
| container_biodegradable | 208 | 0.9148 | 0.7272 | 0.7937 |
| All Classes | 10203 | 0.8651 | 0.6333 | 0.7195 |

## Next investigation

| Area | Observed evidence | Next investigation |
| --- | --- | --- |
| Weak classes | container_yellow: 0.580; container_battery: 0.611; container_blue: 0.612 | Inspect FP/FN examples. |
| Object size | Missed: 35.28 px; detected: 49.00 px | Compare sizes within each class. |
| Localization | container_battery: gap=0.30 | Inspect box alignment. |
| Background false alarms | 33/2529 images; rate=1.30% | Review false detections. |
| Video variation | Recall: 0.08–1.00 | Review low-recall videos. |

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
