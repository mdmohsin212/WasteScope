# GPU pipeline benchmark

- [Pooled result](20261106T142641Z_5e651b4c/benchmark_aggregate.json)
- [Per-clip comparison](20261106T142641Z_5e651b4c/benchmark_comparison.csv)
- [All measured frame timings](20261106T142641Z_5e651b4c/all_frame_timings.csv)

Tesla T4; three clips / 616 frames. Pooled processing FPS: 1.6436. Frame latency P50 / P95 / P99: 498.54 / 959.77 / 3109.38 ms.

Processing covers decoding, D1 detection, T3 tracking, fused S2 segmentation and final track postprocessing. It excludes model loading/warmup, overlay rendering, encoding and final output writes. Each clip resets tracking. Timing scopes and measured memory are in the per-clip summary.json and run_config.json files.

These are recorded benchmark measurements, not accuracy/parity tests, cold-start measurements, or real-time throughput claims. Demo end-to-end FPS has a different scope. Benchmark helper code and source video data remain outside the project; input and code hashes are retained as evidence.
