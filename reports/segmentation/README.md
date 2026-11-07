# Segmentation evidence

S2 fused was selected on validation: D1 proposals + frozen S1 full image + S2 ROI, with original-image mask fusion. S3 is deferred.

| System | Split | Canonical mask AP50 | Canonical mask AP50–95 |
| --- | --- | ---: | ---: |
| S0 | val | 0.2660 | 0.1478 |
| S1 | val | 0.5300 | 0.2775 |
| S2 fused | val | 0.6649 | 0.3650 |
| S2 fused | test | 0.5150 | 0.2737 |

- [Exact comparison and report paths](comparison.csv)
- [S2 validation and confidence sweep](S2_yolo26n_seg_roi_960/val_best_960_fused_conf0.25/README.md)
- [Frozen final test system](S2_yolo26n_seg_roi_960/test_best_960_fused_conf0.2/README.md)
- [S2 ROI preparation coverage](S2_yolo26n_seg_roi_960/preparation/preparation_summary.json)
- [Epoch-1 canonical checkpoint selection](S2_yolo26n_seg_roi_960/training/best_canonical_metrics.json)

The validation folder retains its original confidence-0.25 name and supporting metrics. The final operating confidence is > 0.20, selected on validation and frozen before test; its validation evidence is in the sweep and selected_system.json. AP uses confidence floor 0.001 and is separate from either operating threshold.

S0/S1 used native checkpoint selection; S2 used canonical full-image fused mask AP. Native training curves are monitoring evidence, not the canonical comparison above. S2 includes three checkpoint branches; its component count and runtime differ from a standalone segmenter. Boundary IoU averages matched pairs only.
