import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZIP_STORED, ZipFile

GIB = 1024 ** 3
CLASSES = [
    "container_yellow", "container_green", "container_default", "container_blue",
    "container_ash", "container_oil", "container_battery", "container_biodegradable",
]

def save_json(path, data):
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def convert_label(raw, filename):
    lines = []
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        try:
            if len(fields) != 5:
                raise ValueError("Expected five fields.")
            category = int(fields[0])
            x, y, w, h = map(float, fields[1:])
            
            if not (0 <= category <= 8 and all(math.isfinite(v) for v in (x, y, w, h))
                    and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("Invalid class or box.")
            
        except ValueError as exc:
            raise ValueError(f"Invalid detection label: {filename}:{number}") from exc
        if category != 0:
            lines.append(" ".join([str(category - 1), *fields[1:]]))
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def prepare(archive, manifest_path, output, workdir, shard_gib=2):
    if not math.isfinite(shard_gib) or shard_gib <= 0:
        raise ValueError("--shard-gib must be positive.")
    
    if output.resolve() == workdir.resolve():
        raise ValueError("Output and temporary work folders must be different.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignment = {}
    for split in ("train", "val", "test"):
        
        for video in manifest["splits"][split]:
            
            if not isinstance(video, str) or video in assignment:
                raise ValueError("Video IDs must be strings and appear in exactly one split.")
            assignment[video] = split

    target = max(1, int(shard_gib * GIB))
    output.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.json"

    with ZipFile(archive) as source:
        members = [member for member in source.infolist() if not member.is_dir()]
        images, labels = {}, {}
        
        for member in members:
            path = PurePosixPath(member.filename)
            if path.suffix.lower() not in (".png", ".txt"):
                continue
            parts = path.stem.split("_")
            
            if len(parts) != 5 or not all(part.isdigit() for part in parts):
                raise ValueError(f"Unexpected filename: {member.filename}")
            
            table = images if path.suffix.lower() == ".png" else labels
            if path.stem in table:
                raise ValueError(f"Duplicate filename stem: {path.stem}")
            table[path.stem] = member

        if images.keys() != labels.keys() or not images:
            raise ValueError("PNG/TXT pairs are missing, or the archive is empty.")
        
        if {stem.split("_")[0] for stem in images} != set(assignment):
            raise ValueError("Archive video IDs do not match the frozen split manifest.")

        recipe = {
            "version": 1, "classes": CLASSES, "prepared_splits": ["train", "val", "test"],
            "video_splits": manifest["splits"], "shard_bytes": target,
        }
        source_info = [(m.filename, m.CRC, m.file_size, m.header_offset) for m in members]
        
        signature = hashlib.sha256(
            json.dumps([recipe, source_info], sort_keys=True).encode()
        ).hexdigest()

        if index_path.exists():
            state = json.loads(index_path.read_text(encoding="utf-8"))
            if state["signature"] != signature:
                raise ValueError("Source, split or settings changed. Use a new output folder.")
        else:
            if list(output.glob("*.zip")):
                raise ValueError("Output contains ZIPs without an index. Use a new folder.")
            
            state = {**recipe, "signature": signature, "complete": False, "shards": {}}
            save_json(index_path, state)

        plans = []
        for split in ("train", "val", "test"):
            selected = sorted(
                (s for s in images if assignment[s.split("_")[0]] == split),
                key=lambda s: images[s].header_offset,
            )
            
            group, size, number = [], 0, 0
            for stem in selected:
                required = images[stem].file_size + labels[stem].file_size + 512
                if group and size + required > target:
                    plans.append((f"{split}-{number:04d}.zip", split, group, size))
                    group, size, number = [], 0, number + 1
                    
                group.append(stem)
                size += required
                
            if group:
                plans.append((f"{split}-{number:04d}.zip", split, group, size))
            print(f"{split}: {len(selected):,} images", flush=True)

        pending = []
        for plan in plans:
            name = plan[0]
            record = state["shards"].get(name)
            path = output / name
            
            if record and path.is_file() and path.stat().st_size == record["bytes"]:
                (workdir / f"{name}.part").unlink(missing_ok=True)
            else:
                pending.append(plan)

        print(f"Shards: {len(plans) - len(pending)} saved, {len(pending)} remaining", flush=True)
        
        if pending:
            state["complete"] = False
            save_json(index_path, state)

        needed = [stem for _, _, group, _ in pending for stem in group]
        converted = {}
        
        if needed:
            print(f"Converting {len(needed):,} labels...", flush=True)
            
        for count, stem in enumerate(sorted(needed, key=lambda s: labels[s].header_offset), 1):
            converted[stem] = convert_label(source.read(labels[stem]), labels[stem].filename)
            if count % 5000 == 0:
                print(f"Labels: {count:,}/{len(needed):,}", flush=True)

        for name, split, group, estimated in pending:
            local = workdir / f"{name}.part"
            local.unlink(missing_ok=True)
            
            if shutil.disk_usage(workdir).free < estimated + GIB:
                raise OSError("Insufficient local disk space for the next shard plus 1 GiB reserve.")

            print(f"Building {name}: {len(group):,} images", flush=True)
            
            with ZipFile(local, "w", compression=ZIP_STORED) as destination:
                for stem in group:
                    with source.open(images[stem]) as src:
                        with destination.open(f"{split}/images/{stem}.png", "w", force_zip64=True) as dst:
                            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                    destination.writestr(f"{split}/labels/{stem}.txt", converted[stem])


            temporary = output / f"{name}.part"
            digest = hashlib.sha256()
            
            with local.open("rb") as src, temporary.open("wb") as dst:
                while block := src.read(8 * 1024 * 1024):
                    dst.write(block)
                    digest.update(block)

            size = local.stat().st_size
            
            if temporary.stat().st_size != size:
                raise OSError(f"Incomplete copy: {temporary}")
            
            temporary.replace(output / name)
            state["shards"][name] = {
                "split": split, "images": len(group), "bytes": size,
                "sha256": digest.hexdigest(),
            }
            
            save_json(index_path, state)
            local.unlink()
            print(f"Saved {name}: {size / GIB:.2f} GiB", flush=True)

        state["complete"] = True
        save_json(index_path, state)
        print(f"Prepared dataset saved: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("/content/wastescope_tmp"))
    parser.add_argument("--shard-gib", type=float, default=2)
    
    args = parser.parse_args()
    prepare(args.archive, args.manifest, args.output, args.workdir, args.shard_gib)