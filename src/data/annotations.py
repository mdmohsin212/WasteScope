import json
from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal
from .boxes import (
    Box, _box, coco_xywh_to_xyxy, xywhn_to_xyxy,
    xyxy_to_coco_xywh, xyxy_to_xywhn
)
from .splits import SPLITS, SplitManifest, video_id

ClassMap = Mapping[int, int | None]
__all__ = [
    "Box", "ClassMap", "SPLITS", "BoxAnnotation", "CocoOverflow", "SplitManifest",
    "build_class_map", "coco_xywh_to_xyxy", "decode_coco_mask",
    "parse_yolo_labels", "read_yolo_labels", "video_id", "xywhn_to_xyxy",
    "xyxy_to_coco_xywh", "xyxy_to_xywhn"   
]

def _integer(value: Any, name: str) -> int:
    try:
        number = Decimal(str(value))
        if number != number.to_integral_value() or number < 0:
            raise ValueError
        return int(number)
    
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}") from error
    
    
def build_class_map(
    categories: Iterable[Mapping[str, Any]],
    class_names: Sequence[str], *,
    task: Literal["detection", "tracking", "coco"]
) -> dict[int, int | None]:
    records = [(category["id"], category["name"]) for category in categories]
    
    if not set(class_names).issubset(name for _, name in records):
        raise ValueError("Target class names must be present in source categories")
    
    offset = 1 if task == "detection" else 0
    if task == "tracking":
        records = [(cid, name) for cid, name in records if name.startswith("container_")]
        offset = min(cid for cid, _ in records)
    
    target_ids = {name: index for index, name in enumerate(class_names)}
    return {cid - offset: target_ids.get(name) for cid, name in records}


@dataclass(frozen=True, slots=True)
class BoxAnnotation:
    class_id: int
    xywhn: Box
    source_class_id: int
    track_id: int | None = None
    confidence: float | None = None
    

def parse_yolo_labels(
    text: str | bytes, *,
    class_map: ClassMap,
    format: Literal["detection", "tracking"] = "detection",
    source: str = "<labels>"
) -> list[BoxAnnotation]:
    text = text.decode("utf-8-sig") if isinstance(text, bytes) else text.lstrip("\ufeff")
    tracking = format == "tracking"
    expected =  7 if tracking else 5
    result = []
    
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        try:
            if len(fields) != expected:
                raise ValueError(f"Expected {expected} columns, got {len(fields)}")
            
            raw_id = _integer(fields[0], "class ID")
            if raw_id not in class_map:
                raise ValueError(f"Unknown source class ID {raw_id}")
            
            box = _box(fields[1:5], normalized=True)
            target_id = class_map[raw_id]
            if target_id is None:
                continue
            
            confidence = float(fields[5]) if tracking else None
            track_id = _integer(fields[6], "track ID") if tracking else None
            result.append(BoxAnnotation(target_id, box, raw_id, track_id, confidence))
        
        except ValueError as error:
            raise ValueError(f"{source}:{line_number}: {error}") from error
    return result