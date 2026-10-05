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


