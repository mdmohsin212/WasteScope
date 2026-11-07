# Detection comparison

| Run | Split | Precision | Recall | mAP50 | mAP50-95 | Model | ImgSize | Evaluator |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_yolo26n_640 | val | 0.8866 | 0.7973 | 0.8533 | 0.6414 | yolo26n.pt | 640 | 8.4.1 |
| D0_yolo26n_640 | val | 0.8759 | 0.8300 | 0.8804 | 0.6615 | yolo26n.pt | 960 | 8.4.1 |
| D1_yolo26n_960 | val | 0.8869 | 0.8337 | 0.8836 | 0.6706 | YOLO26n (best.pt) | 960 | 8.4.1 |

ImgSize is the evaluation image size.

- [D0_yolo26n_640 / val / 640](D0_yolo26n_640/val/README.md)
- [D0_yolo26n_640 / val / 960](D0_yolo26n_640/val_960/README.md)
- [D1_yolo26n_960 / val / 960](D1_yolo26n_960/val_960/README.md)

These comparison AP columns use the native Ultralytics evaluator. Canonical COCO tables are stored in each evaluation folder; do not mix the two AP implementations.