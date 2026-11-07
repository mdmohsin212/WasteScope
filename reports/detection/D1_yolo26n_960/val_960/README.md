# D1_yolo26n_960 — validation at 960

Model: YOLO26n | Evaluated checkpoint: `best.pt`
Training image size: 960 | Evaluation image size: 960
Result stage: final
Training status: early stopping at epoch 92, patience 20.
Training epochs: 92/100
Evaluated checkpoint epoch: Not recorded
Images: 5018 | Instances: 10203

## Overall metrics — Ultralytics

| Run | Split | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| D1_yolo26n_960 | val | 0.8869 | 0.8337 | 0.8836 | 0.6706 |

## Per-class COCO AP/AR

| Class | Instances | AP@0.50 | AP@[.50:.95] | AR@[.50:.95] |
| --- | --- | --- | --- | --- |
| container_yellow | 1351 | 0.7977 | 0.5880 | 0.7442 |
| container_green | 1478 | 0.9277 | 0.6829 | 0.7373 |
| container_default | 4850 | 0.8999 | 0.6630 | 0.7153 |
| container_blue | 1606 | 0.8327 | 0.6206 | 0.7172 |
| container_oil | 249 | 0.7837 | 0.5689 | 0.6795 |
| container_battery | 461 | 0.9185 | 0.6346 | 0.6844 |
| container_biodegradable | 208 | 0.8988 | 0.7070 | 0.7683 |
| All Classes | 10203 | 0.8656 | 0.6379 | 0.7209 |

## Next investigation

| Area | Observed evidence | Next investigation |
| --- | --- | --- |
| Weak classes | container_oil: 0.569; container_yellow: 0.588; container_blue: 0.621 | Inspect FP/FN examples. |
| Object size | Missed: 32.66 px; detected: 48.42 px | Compare sizes within each class. |
| Localization | container_battery: gap=0.28 | Inspect box alignment. |
| Background false alarms | 27/2529 images; rate=1.07% | Review false detections. |
| Video variation | Recall: 0.24–1.00 | Review low-recall videos. |

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
