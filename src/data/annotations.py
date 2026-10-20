import json
import numpy as np
from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pycocotools import mask as mask_utils
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


def read_yolo_labels(
    path: str | Path, *,
    class_map : ClassMap,
    format: Literal["detection", "tracking"] = "detection"
) -> list[BoxAnnotation]:
    path = Path(path)
    return parse_yolo_labels(path.read_bytes(), source=str(path), class_map=class_map, format=format)


def decode_coco_mask(
    annotation: Mapping[str, Any],
    width: int, height: int
) -> Any:
    segmantation = annotation["segmentation"]
    if isinstance(segmantation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmantation, height, width))
    else:
        counts = segmantation["counts"]
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(
                {"size": [height, width], "counts": [int(value) for value in counts]}, height, width
            )
        else:
            rle = {"size": [height, width], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
    return mask_utils.decode(rle).astype(np.bool_, copy=False)


class CocoOverflow:
    def __init__(self, document: Mapping[str, Any], *, category_name: str = "Overflow") -> None:
        self._document = deepcopy(dict(document))
        self.categories = self._document["categories"]
        selected = [item for item in self.categories if item["name"].casefold() == category_name.casefold()]
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one COCO category named {category_name!r}")
        
        self.category = selected[0]
        self.images: dict[int, dict[str, Any]] = {image["id"]: image for image in self._document["images"]}
        self._annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        
        for annotation in self._document["annotations"]:
            if annotation["category_id"] == self.category["id"]:
                self._annotations[annotation["image_id"]].append(annotation)
                
    
    @classmethod
    def from_file(cls, path: str | Path, *, category_name: str = "Overflow") -> CocoOverflow:
        return cls(json.loads(Path(path).read_text(encoding="utf-8-sig")), category_name=category_name)
    
    def for_image(self, image_id: int) -> tuple[dict[str, Any], ...]:
        if image_id not in self.images:
            raise KeyError(f"Unknown COCO image ID {image_id}")
        return tuple(self._annotations.get(image_id, ()))
    
    def select_images(self, manifest: SplitManifest, split: str) -> list[int]:
        return [key for key, image in self.images.items() if manifest.split_for(image["file_name"]) == split]
    
    def masks_for_image(self, image_id: int) -> Any:
        image = self.images[image_id]
        masks = [decode_coco_mask(item, image["width"], image["height"]) for item in self.for_image(image_id)]
        return np.stack(masks) if masks else np.empty((0, image["height"], image["width"]), dtype=np.bool_)

    def to_coco(self) -> dict[str, Any]:
        return deepcopy({
            **self._document,
            "categories": [self.category],
            "annotations": [item for item in self._document["annotations"] if item["category_id"] == self.category["id"]]
        })