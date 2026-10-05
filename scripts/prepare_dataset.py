import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

parser = argparse.ArgumentParser()
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

output = args.output.reslove()
manifest = json.loads(args.manifest.read_text())

video_split = {}
for split, videos in manifest["splits"].items():
    for video in videos:
        if video in video_split:
            raise ValueError(f"Video appears in multiple splits: {video}")
        video_split[video] = split

names = [
    "container_yellow", "container_green", "container_default",
    "container_blue", "container_ash", "container_oil",
    "container_battery", "container_biodegradable",
]


def convert_labels(text, filename):
    result = []
    
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{filename}:{number}: expected five fields")
        
        cid = int(fields[0])
        if not 0 <= cid <= 8:
            raise ValueError(f"{filename}:{number}: unknown class {cid}")
        if cid == 0:
            continue
        
        box = [float(value) for value in fields[1:]]
        if not (
            all(math.isfinite(v) and 0 <= v <= 1 for v in box) and box[2] > 0 and box[3] > 0
        ):
            raise ValueError(f"{filename}:{number}: invalid normalized box")

        result.append(f"{cid - 1} {' '.join(fields[1:])}")

    return "\n".join(result) + ("\n" if result else "")


output.mkdir(parents=True, exist_ok=True)

with ZipFile(args.archive) as zf:
    entries, images, labels = [], set(), set()
    
    for item in zf.infolist():
        path = PurePosixPath(item.filename)
        
        if item.is_dir() or path.suffix.lower() not in (".png", ".txt"):
            continue
        
        split = video_split[path.stem.split("_")[0]]
        if split not in ("train", "val"):
            continue
        
        is_image = path.suffix.lower() == ".png"
        seen = images if is_image else labels
        if path.stem in seen:
            raise ValueError(f"Duplicate filename stem: {path.stem}")
        seen.add(path.stem)
        
        target = output / split / ("images" if is_image else "labels") / path.name
        entries.append((item, target, is_image))
    
    if not images or images != labels:
        raise ValueError("Selected images and labels do not match")
    
    entries.sort(key=lambda entry: entry[0].header_offset)
    source = {
        "manifest": manifest,
        "archive_index": hashlib.sha256(json.dumps([
            (item.filename, item.CRC, item.file_size)
            for item, _, _ in entries
        ]).encode()).hexdigest(),
        "label_conversion": "drop_0_then_subtract_1_v1"
    }
    
    marker = output / "_source.json"
    if marker.exists():
        if json.loads(marker.read_text()) != source:
            raise ValueError("Output folder belongs to different data or split")
        
    elif any(output.iterdir()):
        raise ValueError("Use an empty output folder for the first run")
    
    def complete(target, item, is_image):
        return (is_image and target.is_file() and target.stat().st_size == item.file_size)
    
    required = sum(
        item.file_size
        for item, target, is_image in entries
        if not complete(target, item, is_image)
    )
    
    free = shutil.disk_usage(output).free
    print(
        f"Train/val images: {len(images)}; "
        f"remaining: {required / 1024**3:.2f} GiB; "
        f"free: {free / 1024**3:.2f} GiB",
        flush=True,
    )

    if required + 10 * 1024**3 > free:
        raise RuntimeError(
            "Insufficient disk with a 10 GiB reserve. No extraction started."
        )

    marker.write_text(json.dumps(source, indent=2))
    
    for number, (item, target, is_image) in enumerate(entries, 1):
        if not complete(target, item, is_image):
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            
            if is_image:
                with zf.open(item) as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024**2)
            else:
                text = zf.read(item).decode("utf-8-sig")
                temporary.write_text(
                    convert_labels(text, item.filename), encoding="utf-8"
                )
            
            temporary.replace(target)
        
        if number % 5000 == 0:
            print(f"Prepared {number}/{len(entries)} files", flush=True)
    

yaml = (
    f"path: {json.dumps(str(output))}\n"
    "train: train/images\nval: val/images\nnames:\n"
)
yaml += "".join(f"  {i}: {name}\n" for i, name in enumerate(names))
(output / "data.yaml").write_text(yaml, encoding="utf-8")

print(f'Ready: {output / "data.yaml"}')