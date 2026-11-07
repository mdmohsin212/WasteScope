# S0_yolo26n_seg_640 — validation

Original validation images: 1030; overflow instances: 861. Checkpoint: best.pt.

| Metric | Value |
| --- | ---: |
| Canonical mask AP50 | 0.2660 |
| Canonical mask AP50–95 | 0.1478 |
| TP / FP / FN at confidence > 0.25 | 137 / 1 / 724 |
| Matched-pair Boundary IoU | 0.7764 |

AP uses pycocotools COCOeval, mask IoU 0.50:0.05:0.95, confidence floor 0.001 and maxDets [1,10,100]. Supporting mask matches use IoU 0.50 and strict confidence > 0.25. Boundary IoU is calculated only on matched pairs.

Native training metrics selected the saved checkpoint. Its best canonical epoch over all training epochs has not been verified. Final canonical reporting and native training-monitor scores are separate. Comparison with S2 is a system comparison with a different checkpoint-selection rule, not an architecture-only ablation.

- [Metrics, protocol, hashes and environment](metrics.json)
- [Per-class metrics](per_class.csv)
- [Per-image errors](per_image.csv)
- [Training evidence](../training/)

Source-image galleries and raw predictions remain on Drive. Latencies in this evaluation cover the scope recorded in metrics.json and are not full deployed-pipeline timings.
