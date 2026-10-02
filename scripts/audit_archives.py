from __future__ import annotations
import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


def gib(byte_count: int) -> float:
    return round(byte_count / (1024 ** 3), 3)

def suspicious_name(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    
    return path.is_absolute() or ".." in path.parts or (len(name) >= 2 and name[1] == ":")

def inspect_archive(archive: Path) -> dict:
    if not archive.is_file():
        raise FileNotFoundError(f"Completed ZIP not found: {archive}")
    
    if archive.name.endswith(".part"):
        raise ValueError(f"Partial download is not ready: {archive}")
    
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        files = [entry for entry in members if not entry.is_dir()]
        names = [entry.filename for entry in members]
        
        top_level = Counter()
        extensions = Counter()
        candidate_annotations = []
        
        for entry in files:
            posix_name = entry.filename.replace("\\", "/")
            parts = [part for part in posix_name.split("/") if part]
            
            top_level[parts[0] if parts else "<empty>"] += 1
            extensions[PurePosixPath(posix_name).suffix.lower() or "<none>"] += 1
            
            lowered = posix_name.lower()
            suffix = PurePosixPath(lowered).suffix
            if len(candidate_annotations) < 20 and (
                suffix in {".json", ".csv", ".xml", ".yaml", ".yml"}
                or any(token in lowered for token in ("/gt/", "annotation", "label", "seqinfo"))
            ):
              candidate_annotations.append(posix_name)
        
        
        expanded = sum(entry.file_size for entry in files)
        compressed = sum(entry.compress_size for entry in files)
        duplicates = len(names) - len(set(names))
        suspicious = [name for name in names if suspicious_name(name)]
        encrypted_count = sum(bool(entry.flag_bits & 1) for entry in files)
    
    return {
        "archive": str(archive),
        "archive_gib": gib(archive.stat().st_size),
        "member_count": len(members),
        "file_count": len(files),
        "expanded_gib": gib(expanded),
        "compressed_members_gib": gib(compressed),
        "top_level_file_counts": top_level.most_common(20),
        "file_extensions": extensions.most_common(20),
        "annotation_candidates_first_20": candidate_annotations,
        "first_20_member_names": names[:20],
        "duplicate_member_names": duplicates,
        "suspicious_paths_count": len(suspicious),
        "suspicious_paths_first_10": suspicious[:10],
        "encrypted_file_count": encrypted_count,
        "integrity_scope": "ZIP directory was read; contents and CRCs were not tested"
    }
    

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--archive", action="append", type=Path, required=True)
    
    parser.add_argument("--output-dir", type=Path, default=Path("reports/private"))
    
    args = parser.parse_args()
    reports = []
    for archive in args.archive:
        try:
            report = inspect_archive(archive)
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile, ValueError) as exc:
            print(f"ERROR: {archive}: {exc}", file=sys.stderr)
            return 1
        
        reports.append(reports)
        print(
            f"{archive.name}: {report['file_count']} files, "
            f"{report['archive_gib']} GiB ZIP, "
            f"{report['expanded_gib']} GiB expanded"
        )
        print("Top-level:", report["top_level_file_counts"][:8])
        print("Extensions:", report["file_extensions"][:8])
        print(
            "Path checks:", report["suspicious_paths_count"], "suspicious,", report["duplicate_member_names"], "duplicate,", report["encrypted_file_count"], "encrypted"
        )
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for report, archive in zip(reports, args.archive):
        destination = args.output_dir / f"{archive.stem}_audit.json"
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("Private inventory saved:", destination)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())