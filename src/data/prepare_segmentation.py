import json
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

def polygon_to_yolo(annotation: dict[str, Any], image: dict[str, Any]) -> str:
    segments = annotation["segmentation"]
    if len(segments) != 1:
        raise ValueError(f"Expected one polygon: annotation {annotation['id']}")
    
    polygon = segments[0]
    if len(polygon) < 6 or len(polygon) % 2:
        raise ValueError(f"Invalid polygon: annotation {annotation['id']}")
    
    coordinates: list[float] = [
        value / (image["width"] if index % 2 == 0 else image["height"]) for index, value in enumerate(polygon)
    ]
    
    if not  all(0 <= value <=1 for value in coordinates):
        raise ValueError(f"Out-of-bounds polygon: annotation {annotation['id']}")
    return "0 " + " ".join(f"{value:.8f}" for value in coordinates)


def prepare_segmentation(source_zip: str | Path, manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source_zip, manifest_path = Path(source_zip), Path(manifest_path)
    output_dir = Path(output_dir)
    manifest_bytes = manifest_path.read_bytes()
    manifest: dict[str, Any] = json.loads(manifest_bytes)
    splits: tuple[str, str, str] = ("train", "val", "test")
    video_to_split: dict[str, str] = {}
    
    for split in splits:
        for video in manifest["splits"][split]:
            if video in video_to_split:
                raise ValueError(f"Duplicate video assignment: {video}")
            
            video_to_split[video] = split
    
    with zipfile.ZipFile(source_zip) as source:
        data: dict[str, Any] = json.loads(source.read("Segmentation Task/labels.json"))
        category: dict[str, Any] = next(c for c in data["categories"] if c["name"].lower() == "overflow")
        
        members: dict[str, zipfile.ZipInfo] = {
            PurePosixPath(info.filename).name: info
            for info in source.infolist()
            if PurePosixPath(info.filename).suffix.lower() in {".png", ".jpg", ".jpeg"}
        }
        
        by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in data["annotations"]:
            if item["category_id"] == category["id"]:
                by_image[item["image_id"]].append(item)
                
        grouped: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
        labels: dict[int, str] = {}
        
        for image in data["images"]:
            name = PurePosixPath(image["file_name"]).name
            grouped[video_to_split[name.split("_")[0]]].append(image)
            rows: list[str] = [polygon_to_yolo(item, image) for item in by_image[image["id"]]]
            labels[image["id"]] = "\n".join(rows) + ("\n" if rows else "")

        summary: dict[str, Any] = {
            "source_archive": source_zip.name,
            "source_annotations": len(data["annotations"]),
            "yolo_class": {"id": 0, "name": category["name"]},
            "coco_category_id": category["id"],
            "split_policy": "existing video_split.json; no reassignment",
            "splits": {}
        }
        for split, records in grouped.items():
            positive = sum(bool(by_image[image["id"]]) for image in records)
            summary["splits"][split] = {
                "images": len(records), "positive": positive,
                "negative": len(records) - positive,
                "overflow_instances": sum(len(by_image[image["id"]]) for image in records),
                "videos": len({PurePosixPath(image["file_name"]).name.split("_")[0] for image in records})
            }
            
        summary_text = json.dumps(summary, indent=2)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with tempfile.TemporaryDirectory(prefix="wastescope_seg_") as temporary:
            for split, records in grouped.items():
                local_zip = Path(temporary) / f"{split}.zip"
                records = sorted(records, key=lambda im: members[PurePosixPath(im["file_name"]).name].header_offset)
                coco: dict[str, Any] = {
                    "info": data.get("info", {}), "licenses": data.get("licenses", []),
                    "images": records, "categories": [category],
                    "annotations": [item for im in records for item in by_image[im["id"]]]
                }
                print(f"Building {split}.zip: {len(records)} images", flush=True)
                
                with zipfile.ZipFile(local_zip, "w", compression=zipfile.ZIP_STORED) as target:
                    for index, image in enumerate(records, 1):
                        name = PurePosixPath(image["file_name"]).name
                        target.writestr(f"{split}/images/{name}", source.read(members[name]))
                        target.writestr(f"{split}/labels/{Path(name).stem}.txt", labels[image["id"]])
                        
                        if index % 500 == 0 or index == len(records):
                            print(f"  {split}: {index}/{len(records)}", flush=True)
                    target.writestr(f"{split}/annotations.json", json.dumps(coco))
                    
                    if split == "train":
                        target.writestr("data.yaml", "train: train/images\nval: val/images\ntest: test/images\nnames:\n  0: Overflow\n")
                        target.writestr("video_split.json", manifest_bytes)
                        target.writestr("preparation_summary.json", summary_text)
                        
                partial = output_dir / f"{split}.zip.part"
                print(f"Copying {split}.zip to {output_dir}", flush=True)
                shutil.copyfile(local_zip, partial)
                
                partial.replace(output_dir / f"{split}.zip")
                local_zip.unlink()
                print(f"Saved {split}.zip", flush=True)

    (output_dir / "preparation_summary.json").write_text(summary_text, encoding="utf-8")