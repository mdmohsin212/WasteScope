# ONNX conversion check

- [Summary](demo_containers_20261107T053955Z_a97bb108/summary.json)
- [All comparisons](demo_containers_20261107T053955Z_a97bb108/comparisons.csv)

Existing GitHub evidence is preserved: CPU FP32 PyTorch versus CPU FP32 ONNX Runtime on six sampled frames from two clips. All 24 profile comparisons passed the recorded box/mask IoU and score tolerances. Empty/empty outputs have undefined IoU and may appear as blank CSV cells; they are not comparison failures.

This establishes sampled conversion parity only. It does not establish full validation-set accuracy, full-sequence tracking parity, CUDA/FP16 parity or a backend speed comparison. Official held-out task metrics remain the frozen task reports.
