from pathlib import Path
import modal

GPU = "RTX-PRO-6000"
RUN_NAME = "S2_yolo26n_seg_roi_960"
IMGSZ, BATCH, VAL_BATCH, WORKERS = 960, 64, 32, 8
HERE = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("rclone", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.11.0", "torchvision==0.26.0",index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("ultralytics==8.4.1", "pycocotools==2.0.11", "tqdm==4.67.1")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file(str(HERE / "s2_monitor.py"), "/root/s2_monitor.py")
    .add_local_file(str(HERE / "rclone.conf"), "/root/rclone.conf")
)
app = modal.App("wastescope-segmentation-s2")

@app.function(image=image, gpu=GPU, timeout=24 * 60 * 60)
def train_s2():
    import hashlib
    import json
    import os
    import platform
    import shutil
    import subprocess
    import time
    from importlib.metadata import version
    from zipfile import ZipFile

    root = Path("/tmp/wastescope")
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    for key, folder in {"XDG_CACHE_HOME": "cache", "TORCH_HOME": "cache/torch",
                        "YOLO_CONFIG_DIR": "cache/ultralytics"}.items():
        directory = root / folder
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)

    import cv2
    import torch
    import ultralytics
    import yaml
    from tqdm.auto import tqdm
    from ultralytics import YOLO
    from s2_monitor import CanonicalROITrainer, FullImageMonitor

    cv2.setNumThreads(1)
    config = root / "rclone.conf"
    shutil.copy2("/root/rclone.conf", config)
    config.chmod(0o600)
    data = root / "segmentation_roi"
    prepared = "gdrive:WasteScope/prepared/segmentation_roi"
    drive_run = f"gdrive:WasteScope/runs/segment/{RUN_NAME}"

    def rclone(*args, check=True, **kwargs):
        return subprocess.run(["rclone", "--config", str(config), *map(str, args)], check=check, **kwargs)

    existing = rclone("lsf", drive_run, "--max-depth", "1", check=False, capture_output=True, text=True)
    if existing.returncode == 0 and existing.stdout.strip():
        raise FileExistsError(f"{drive_run} already contains a run. Choose a new RUN_NAME.")
    
    if existing.returncode not in (0, 3):
        existing.check_returncode()
        
    data.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        archive_path = root / f"{split}.zip"
        print(f"Downloading ROI {split}.zip", flush=True)
        rclone("copyto", f"{prepared}/{split}.zip", archive_path,
               "--progress", "--stats", "10s", "--buffer-size", "16M",
               "--multi-thread-streams", "4", "--retries", "5")
        with ZipFile(archive_path) as archive:
            for member in tqdm(archive.infolist(), desc=f"Extracting {split}"):
                archive.extract(member, data)
        archive_path.unlink()

    dataset = yaml.safe_load((data / "data.yaml").read_text())
    dataset.update(path=str(data), train="train/images", val="val/images")
    (data / "data.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    monitor = FullImageMonitor(data, batch=VAL_BATCH, encoding_workers=4)
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name}; VRAM: {gpu.total_memory / 1024**3:.1f} GiB", flush=True)
    print(f"PyTorch: {torch.__version__}; Ultralytics: {ultralytics.__version__}", flush=True)

    def backup(trainer, force=False):
        started = time.monotonic()
        try:
            rclone("copy", trainer.save_dir, drive_run, "--retries", "5", "--stats", "0", timeout=180)
            print(f"Drive backup complete ({time.monotonic() - started:.1f}s)", flush=True)
        except (OSError, subprocess.SubprocessError) as error:
            print(f"Drive backup failed: {error}; local run: {trainer.save_dir}", flush=True)
            if force:
                raise

    def save_context(trainer):
        trainer.roi_monitor = monitor
        run = Path(trainer.save_dir)
        for name in ("modal_train_s2.py", "s2_monitor.py"):
            shutil.copy2(Path("/root") / name, run / name)
        for name in ("roi_policy.json", "video_split.json", "data.yaml", "val/annotations.json",
                     "train/preparation_summary.json", "val/preparation_summary.json",
                     "support/s1_val_metrics.json", "support/s1_val_predictions.json"):
            destination = run / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data / name, destination)
        context = {
            "prepared_source": prepared,
            "checkpoint_selection": "canonical full-original-image S1+ROI mask AP50-95",
            "initialization": "COCO-pretrained yolo26n-seg.pt; independent of S1 weights",
            "optimizer_policy": "Match S1 auto-selected MuSGD: lr0=0.002, momentum=0.9, warmup_bias_lr=0.0",
            "comparison_limitations": "S0/S1 used native checkpoint selection; S2 uses canonical fused AP and more ROI training samples",
            "s1_branch": "frozen cached validation predictions from the recorded S1 checkpoint",
            "fusion_suppression": "mask IoU >= roi_policy.fusion_iou; S1 first on score ties",
            "roi_validation": {"batch": VAL_BATCH, "imgsz": IMGSZ, "half": True,
                               "retina_masks": True, "rect": False,
                               "projection": "nearest-neighbour binary mask resize into original ROI"},
            "effective_train_crops": len(trainer.train_loader.dataset),
            "original_validation_images": len(monitor.gt.imgs),
            "original_validation_instances": len(monitor.gt.anns),
            "optimizer": {"name": type(trainer.optimizer).__name__, "defaults": trainer.optimizer.defaults},
            "sha256": {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                       for name in ("roi_policy.json", "video_split.json", "val/annotations.json",
                                    "val/roi_index.json", "support/s1_val_predictions.json")},
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "torchvision": version("torchvision"), "ultralytics": ultralytics.__version__,
                            "pycocotools": version("pycocotools"), "numpy": version("numpy"),
                            "opencv": cv2.__version__, "cuda": torch.version.cuda, "gpu": gpu.name},
        }
        (run / "run_context.json").write_text(json.dumps(context, indent=2, default=str))
        backup(trainer, force=True)

    model = YOLO(str(root / "yolo26n-seg.pt"))
    model.add_callback("on_train_start", save_context)
    model.add_callback("on_model_save", backup)
    completed = False
    try:
        model.train(
            trainer=CanonicalROITrainer,
            data=str(data / "data.yaml"), epochs=100, patience=20,
            imgsz=IMGSZ, batch=BATCH, device=0, workers=WORKERS,
            optimizer="MuSGD", lr0=0.002, momentum=0.9, warmup_bias_lr=0.0,
            seed=42, deterministic=True, amp=True,
            cache="disk", val=True, plots=True, save=True,
            project=str(root / "runs/segment"), name=RUN_NAME, exist_ok=False
        )
        completed = True
    finally:
        trainer = getattr(model, "trainer", None)
        if trainer is not None and Path(trainer.save_dir).is_dir():
            status = {"status": "complete" if completed else "interrupted_or_failed",
                      "last_epoch_started": getattr(trainer, "epoch", -1) + 1,
                      "target_epochs": 100}
            (Path(trainer.save_dir) / "training_status.json").write_text(json.dumps(status, indent=2))
            backup(trainer, force=True)
    print(f"Best checkpoint: {drive_run}/weights/best.pt", flush=True)
    print(f"Canonical results: {drive_run}/best_canonical_metrics.json", flush=True)


@app.local_entrypoint()
def main():
    call = train_s2.spawn()
    print("Job submitted:", call.object_id)