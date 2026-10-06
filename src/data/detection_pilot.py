import hashlib
import json
import math
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from PIL import Image, ImageDraw

GIB = 1024**3
CLASSES = [
    "container_yellow", "container_green", "container_default", "container_blue",
    "container_ash", "container_oil", "container_battery", "container_biodegradable",
]
VARIANTS = {"original_png": ".png", "jpeg_q95": ".jpg"}

def read_splits(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = manifest["splits"]
    
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Expected train, val and test splits.")
    
    assignment = {}
    for split, videos in splits.items():
        if not isinstance(videos, list) or not videos:
            raise ValueError(f"Empty or invalid split: {split}")
        
        for video in videos:
            if not isinstance(video, str) or not video.isdigit() or video in assignment:
                raise ValueError(f"Invalid or repeated video ID: {video!r}")
            assignment[video] = split
    
    for video, split in manifest.get("fixed_assignments", {}).items():
        if assignment.get(video) != split:
            raise ValueError(f"Fixed assignment changed: {video}")
    return assignment