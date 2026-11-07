# D1_yolo26n_960 — final test
Model: YOLO26n | Image size: 960
Images: 5036 | Instances: 10028

## Native Ultralytics metrics

| Run            | Split   |   Precision |   Recall |   mAP50 |   mAP50-95 |
|:---------------|:--------|------------:|---------:|--------:|-----------:|
| D1_yolo26n_960 | test    |      0.9487 |   0.9193 |  0.9655 |     0.7554 |

Native precision/recall use Ultralytics' operating-point calculation.
The fixed-policy metrics below are reported separately.

## Per-class COCO AP/AR

| Class                   |   Instances |   AP@0.50 |   AP@[.50:.95] |   AR@[.50:.95] |
|:------------------------|------------:|----------:|---------------:|---------------:|
| container_yellow        |        1475 |    0.9305 |         0.6966 |         0.7557 |
| container_green         |        1380 |    0.9592 |         0.6908 |         0.7707 |
| container_default       |        4171 |    0.9632 |         0.7511 |         0.7887 |
| container_blue          |        1635 |    0.9657 |         0.7549 |         0.8125 |
| container_oil           |         182 |    0.9532 |         0.7044 |         0.7918 |
| container_battery       |         600 |    0.9669 |         0.7511 |         0.8433 |
| container_biodegradable |         585 |    0.9651 |         0.737  |         0.7891 |
| All Classes             |       10028 |    0.9577 |         0.7266 |         0.7931 |

## Fixed operating policy

Default confidence > 0.35;
Oil confidence > 0.2.

| Run            | Split   |   Macro Precision |   Macro Recall |   Macro F1 |   TP |   FP |   FN |   Oil Recall |
|:---------------|:--------|------------------:|---------------:|-----------:|-----:|-----:|-----:|-------------:|
| D1_yolo26n_960 | test    |            0.9004 |         0.9498 |     0.9221 | 9474 |  808 |  554 |            1 |

| Class                   |   TP |   FP |   FN |   Precision |   Recall |     F1 |   Confidence |
|:------------------------|-----:|-----:|-----:|------------:|---------:|-------:|-------------:|
| container_yellow        | 1352 |  126 |  123 |      0.9147 |   0.9166 | 0.9157 |         0.35 |
| container_green         | 1298 |   95 |   82 |      0.9318 |   0.9406 | 0.9362 |         0.35 |
| container_default       | 3988 |  240 |  183 |      0.9432 |   0.9561 | 0.9496 |         0.35 |
| container_blue          | 1527 |  116 |  108 |      0.9294 |   0.9339 | 0.9317 |         0.35 |
| container_oil           |  182 |   33 |    0 |      0.8465 |   1      | 0.9169 |         0.2  |
| container_battery       |  589 |  185 |   11 |      0.761  |   0.9817 | 0.8574 |         0.35 |
| container_biodegradable |  538 |   13 |   47 |      0.9764 |   0.9197 | 0.9472 |         0.35 |

## Protocol

- Checkpoint: best.pt; policy selected on validation and unchanged on test.
- AP predictions: confidence 0.001.
- COCO AP/AR: IoU 0.50:0.05:0.95, maxDets=[1,10,100].
- Operating errors: COCO class-aware, score-ordered matching, IoU 0.50.
- Native confusion matrix uses confidence > 0.25 and matching IoU > 0.45;
  its matching protocol differs from the operating-error tables.
- Single-run results; no mean ± std.

## Figures

- [BoxF1_curve.png](figures/BoxF1_curve.png)
- [BoxPR_curve.png](figures/BoxPR_curve.png)
- [BoxP_curve.png](figures/BoxP_curve.png)
- [BoxR_curve.png](figures/BoxR_curve.png)
- [confusion_matrix.png](figures/confusion_matrix.png)
- [confusion_matrix_normalized.png](figures/confusion_matrix_normalized.png)

## Saved outputs

- [Metrics CSV files](metrics/)
- [Frozen operating policy](operating_policy.json)
- [Evaluation configuration](evaluation_config.json)
- [Frozen video split](video_split.json)
