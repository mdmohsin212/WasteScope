from __future__ import annotations
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SPLITS = ("train", "val", "test")

def video_id(filename: str | Path) -> str:
    prefix = PurePosixPath(str(filename).replace("\\", "/")).stem.split("_", 1)[0]
    return prefix


@dataclass(frozen=True, slots=True)
class SplitManifest:
    classes: tuple[str, ...]
    splits: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        names = tuple(self.classes)
        if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Class names must be non-empty and unique")
        if set(self.splits) != set(SPLITS):
            raise ValueError("The manifest needs train, val and test video lists")
        groups, seen = {}, set()
        for split in SPLITS:
            videos = self.splits[split]
            if isinstance(videos, str) or not videos:
                raise ValueError(f"{split} must be a non-empty video list")
            for value in videos:
                if not isinstance(value, str) or video_id(value) != value:
                    raise ValueError(f"Invalid video ID in {split}: {value!r}")
                if value in seen:
                    raise ValueError(f"Video {value} is repeated in the manifest")
                seen.add(value)
            groups[split] = frozenset(videos)
        object.__setattr__(self, "classes", names)
        object.__setattr__(self, "splits", groups)

    @classmethod
    def from_file(cls, path: str | Path) -> SplitManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if data.get("group_by", "video_id") != "video_id":
            raise ValueError("A video-level manifest is required")
        return cls(tuple(data["classes"]), data["splits"])

    @property
    def class_map(self) -> dict[int, int]:
        return {index: index for index in range(len(self.classes))}

    def split_for(self, filename: str | Path) -> str:
        key = video_id(filename)
        for split, videos in self.splits.items():
            if key in videos:
                return split
        raise ValueError(f"Video {key} is absent from the manifest")

    def select(self, filenames: Iterable[str | Path], split: str) -> list[str | Path]:
        if split not in self.splits:
            raise ValueError(f"Unknown split: {split!r}")
        return [name for name in filenames if self.split_for(name) == split]
