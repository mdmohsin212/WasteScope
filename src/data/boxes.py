from collections.abc import Sequence

Box = tuple[float, float, float, float]

def _box(values: Sequence[float], *, normalized: bool = False) -> Box:
    box = tuple(float(value) for value in values)

    if normalized and (any(value < 0 or value > 1 for value in box) or min(box[2:]) <= 0):
        raise ValueError("Normalized xywh requires values in [0, 1] and positive width/height")
    return box


def xywhn_to_xyxy(box: Sequence[float], width: int, height: int, *, clip: bool = False) -> Box:
    cx, cy, w, h = _box(box, normalized=True)
    corners = (
        (cx - w / 2) * width, (cy - h / 2) * height,
        (cx + w / 2) * width, (cy + h / 2) * height
    )
    
    if clip:
        return tuple(max(0.0, min(value, limit)) for value, limit in zip(corners, (width, height, width, height)))
    return corners


def xyxy_to_xywhn(box: Sequence[float], width: int, height: int, *, clip: bool = False) -> Box:
    x1, y1, x2, y2 = _box(box)
    if clip:
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Pixel xyxy must have positive area; clipping may remove the entire box")
    
    return _box((
        (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
        (x2 - x1) / width, (y2 - y1) / height
    ), normalized=True)
    

def coco_xywh_to_xyxy(box: Sequence[float]) -> Box:
    x, y, width, height = _box(box)
    return x, y, x + width, y + height


def xyxy_to_coco_xywh(box: Sequence[float]) -> Box:
    x1, y1, x2, y2 = _box(box)
    return x1, y1, x2 - x1, y2 - y1