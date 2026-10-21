from __future__ import annotations
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SPLITS = ("train", "val", "test")

def video_id(filename: str | Path) -> str:
    prefix = PurePosixPath(str(filename).replace("\\", "/")).stem.split("_", 1)[0]
    if len(prefix) != 4 or not prefix.isdigit():
        raise ValueError(f"Expected a four-digit video prefix: {filename!s}")
    return prefix


@dataclass(frozen=True, slots=True)
class SplitManifest:
    
    classes: tuple[str, ...]
    splits: Mapping[str, frozenset[str]]
    
    def __post_init__(self) -> None:
        groups = {split: frozenset(self.splits[split]) for split in SPLITS}
        if sum(map(len, groups.values())) != len(frozenset().union(*groups.values())):
            raise ValueError("A video is repeated across the manifest splits")
        
        object.__setattr__(self, "classes", tuple(self.classes))
        object.__setattr__(self, "splits", groups)
    
    @classmethod
    def from_file(cls, path: str | Path) -> SplitManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(tuple(data["classes"]), data["splits"])
    
    def split_for(self, filename: str | Path) -> str:
        key = video_id(filename)
        for split, videos in self.splits.items():
            if key in videos:
                return split
        raise ValueError(f"Video {key} is absent from the manifest")
 
    def select(self, filenames: Iterable[str | Path], split: str) -> list[str | Path]:
        return [name for name in filenames if self.split_for(name) == split]