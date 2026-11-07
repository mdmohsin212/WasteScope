# Data access and preparation

WasteScope uses [StreetView-Waste](https://streetview-waste.di.ubi.pt/) for container detection, tracking/counting and overflow instance segmentation. Request access through the [authors' dataset repository](https://github.com/DiogoJPaulo/StreetView-Waste); access is managed through a data agreement.

Dataset archives, extracted images, derived clips and credentials are not included in this repository. Model checkpoint access is separate from dataset access. Only compact report/configuration evidence is committed.

| Workflow | Entry point |
| --- | --- |
| Archive audit | [01_dataset_audit.ipynb](../notebooks/01_dataset_audit.ipynb) |
| Video-level split | [02_video_distribution_and_split.ipynb](../notebooks/02_video_distribution_and_split.ipynb) |
| Detection preparation/training | [Detection notebooks](../notebooks/detection/) |
| Tracking preparation and reviewed identity corrections | [Tracking notebooks](../notebooks/tracking/) and [preparation evidence](../reports/tracking/preparation/README.md) |
| Segmentation preparation and expanded ROI experiment | [Segmentation notebooks](../notebooks/segmentation/) |

Cloud notebooks generally use `/content/drive/MyDrive/WasteScope` with `raw_archives/`, `prepared/`, `manifests/video_split.json`, `runs/` and `reports/`. Use your own authorized storage paths. Retain the frozen split; do not mix videos across train/val/test or tune on the test split.

The tracking task uses reviewed corrected annotations. Use the corrected preparation summaries and the annotation hash in the selected tracking configuration, rather than the archived original tables.