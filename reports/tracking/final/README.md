# Tracking final report

Selected configuration: **T3_botsort**, chosen on validation before the final test.
T3 has the highest validation HOTA and IDF1, with counting MAE tied with T2.
T2 has fewer identity switches; this is the selection trade-off.

## Validation comparison

| Run | Split | HOTA | IDF1 | MOTA | IDSW | MAE | SAD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T0_bytetrack | val | 60.2081 | 75.0568 | 62.3903 | 38 | 3.6923 | 48 |
| T1_bytetrack_calibrated | val | 62.1487 | 78.6632 | 63.1801 | 30 | 2.2308 | 29 |
| T2_bytetrack_postprocessed | val | 62.2186 | 78.7734 | 63.4082 | 17 | 1.0769 | 14 |
| T3_botsort | val | 67.3233 | 80.9069 | 67.1639 | 27 | 1.0769 | 14 |

## Final held-out test

| Run | Split | HOTA | IDF1 | MOTA | IDSW | MAE | SAD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T3_botsort | test | 66.2731 | 78.3097 | 64.9351 | 32 | 0.5714 | 8 |

Exact sequence counts: **8/14**.
GT tracks: **55**; counted tracks: **57**.
Sum of absolute per-sequence count errors: **8**.
Settings were frozen before the test; the test was not used for tuning.

## Measured limitations

The two lowest-HOTA test sequences and the sequence with the most ID switches:

| Sequence | GT | RawTracks | Predicted | HOTA | IDF1 | MOTA | IDSW |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0124_1 | 5 | 5 | 5 | 11.6992 | 9.8361 | -73.719 | 1 |
| 0108_1 | 2 | 3 | 0 | 13.2271 | 21.8182 | 12.766 | 2 |
| 0115_1 | 9 | 15 | 11 | 64.5432 | 80.8571 | 75.3906 | 11 |

Count agreement can coexist with poor tracking quality. These results document remaining errors without assigning unverified causes.

## Protocol and counting

- Class-aware TrackEval: HOTA IoU 0.05:0.05:0.95; IDF1/MOTA IoU 0.50.
- HOTA, DetA, AssA, IDF1, MOTA and MAPE are percentages. IDSW is a count.
- Counting metrics measure unique objects per sequence. RawTracks counts IDs after reconnection; Predicted counts IDs meeting the minimum-duration rule.
- Minimum duration is 10 distinct observed frames, after same-class reconnection with at most 5 missing frame steps and normalized center distance 0.05.
- Tracking metrics retain all emitted boxes; duration filtering affects counting only. Counts are finalized after the complete sequence.
- MAE/RMSE/SAD/MAPE use per-sequence count errors. The difference between total GT and total predicted counts can hide errors that cancel.
- Results use the corrected annotations and frozen video split. These are single-run results.
- Saved tracking wall times cover tracker/postprocessing, with detections already cached; they are not end-to-end pipeline FPS.

## Saved configuration and sources

- [Selected configuration](selected_config.json): detector inference settings, tracker, postprocessing, dependency versions, and recorded data/checkpoint hashes. File paths are relative to the WasteScope Drive root.
- [Full comparison](comparison.csv)
- [T0_bytetrack / val](../T0_bytetrack/val/README.md)
- [T1_bytetrack_calibrated / val](../T1_bytetrack_calibrated/val/README.md)
- [T2_bytetrack_postprocessed / val](../T2_bytetrack_postprocessed/val/README.md)
- [T3_botsort / val](../T3_botsort/val/README.md)
- [T3_botsort / test](../T3_botsort/test/README.md)