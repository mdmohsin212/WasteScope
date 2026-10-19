import math
from collections.abc import Sequence

Box = tuple[float, float, float, float]

def _box(values: Sequence[tuple], *, normalized: bool = False) -> Box:
    if len(values) != 4:
        raise ValueError("A box needs exactly four coordinates")
    
    box = tuple(float(value) for value in values)
    return box


def xywhn_to_xyxy(box: Sequence[float], width: int, height: int, *, clip: bool = False) -> Box:
    cx, cy, w, h = _box(box, normalized=True)
    corners = ((cx - w / 2) * width, (cy - h / 2) * height,
            (cx + w / 2) * width, (cy + h / 2) * height)
    if clip:
        return tuple(max(0.0, min(value, limit)) for value, limit in zip(corners, (width, height, width, height)))
    
    return corners