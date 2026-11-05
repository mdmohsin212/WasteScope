import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path, expected):
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    if sha256(path) != expected:
        raise ValueError(f"Checkpoint SHA256 mismatch: {path}")


def stage(models, output, *, source_root=None, remote=None, rclone_config=None):
    rclone = shutil.which("rclone") if remote else None
    if remote and rclone is None:
        raise FileNotFoundError(
            "Install rclone or use --source-root "
            "with a local/Drive-mounted folder."
        )

    output.mkdir(parents=True, exist_ok=True)
    for model_id, record in models.items():
        target = output / record["filename"]

        if target.is_file() and sha256(target) == record["sha256"]:
            print(f"{model_id}: already verified", flush=True)
            continue

        descriptor, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=output)
        os.close(descriptor)
        temporary = Path(name)

        try:
            if remote:
                command = [rclone]
                if rclone_config:
                    command.extend(["--config", str(rclone_config)])

                command.extend(["copyto", f"{remote.rstrip('/')}/{record['source_path']}", str(temporary)])
                subprocess.run(command, check=True)
            else:
                shutil.copyfile(source_root / record["source_path"], temporary)

            verify(temporary, record["sha256"])
            temporary.replace(target)

        finally:
            temporary.unlink(missing_ok=True)
        print(f"{model_id}: copied and verified", flush=True)


def smoke_test(models, output):
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    for model_id, record in models.items():
        model = YOLO(str(output / record["filename"]))
        names = [model.names[key] for key in sorted(model.names)]

        if model.task != record["task"] or names != record["classes"]:
            raise ValueError(f"{model_id}: task or class order differs from the manifest")

        model.predict(
            image,
            imgsz=64,
            conf=0.99,
            device="cpu",
            half=False,
            verbose=False,
            save=False
        )
        print(f"{model_id}: CPU load/forward passed", flush=True)
        del model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    
    source.add_argument("--source-root", type=Path, help="Local or mounted WasteScope folder containing runs/.",)
    source.add_argument("--remote", help="Existing rclone remote root, e.g. gdrive:WasteScope.",)
    source.add_argument("--check", action="store_true", help="Verify existing local checkpoints; do not copy.",)

    parser.add_argument("--rclone-config", type=Path, help="Optional existing rclone.conf for --remote.",)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models/weights",)
    parser.add_argument("--smoke-test", action="store_true", help="Load each model and run a tiny CPU forward pass.",)

    args = parser.parse_args(argv)
    
    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    models = manifest["models"]

    if args.source_root or args.remote:
        stage(models, args.models_dir, source_root=args.source_root, remote=args.remote, rclone_config=args.rclone_config,)

    for model_id, record in models.items():
        verify(args.models_dir / record["filename"], record["sha256"])
        print(f"{model_id}: SHA256 verified", flush=True)

    if args.smoke_test:
        smoke_test(models, args.models_dir)
        
    print("Three checkpoints ready. No dataset downloads or training performed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        raise SystemExit(1)