# Tracking preparation

Use `split_summary.csv`, `sequences.csv` and `class_summary.csv`: these were regenerated from the corrected annotation file whose SHA256 is recorded in the final selected configuration. The compact `split_summary_corrected.csv` is the original Drive export of corrected totals.

Corrected totals: train 23,802 instances / 264 tracks; val 5,698 / 56; test 4,697 / 55.
The correction log records two removed stale boxes, restored rear-bin identity in 0103_1, and the second test bin assigned ID 54 in 0115_1. Track IDs are sequence-scoped.

`original/` contains the three unchanged pre-correction tables for audit history. `identity_review/` describes the original conflicts, not the remaining quality of corrected annotations. Source images and full annotation/frame tables remain on Drive.

- [Corrected split summary](split_summary.csv)
- [Correction log](annotation_corrections.csv)
- [Class mapping and quality checks](data_config.json)
- [Original conflict summary](identity_review/issue_summary.csv)
