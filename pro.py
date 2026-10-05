"""Create the WasteScope project skeleton without overwriting existing files."""

import argparse
import json
from pathlib import Path


def create_project(root: Path) -> None:
    files = {
        "README.md": (
            "# WasteScope — Urban Waste Intelligence\n\n"
            "Container detection, tracking and counting, and Overflow instance segmentation.\n\n"
            "Keep datasets and model weights outside this code repository.\n"
            "Use the frozen video_split_v1.json for all three tasks.\n"
            "This scaffold contains placeholders; existing files are preserved.\n"
        ),
        "LICENSE": "",
        "requirements.txt": "# Add tested dependency versions as each component is implemented.\n",
        "Dockerfile": "# Add build instructions when the API is implemented.\n",
        ".gitignore": (
            "__pycache__/\n*.py[cod]\n.venv/\nvenv/\n.ipynb_checkpoints/\n"
            ".env\n.env.*\n!.env.example\n"
            "raw_archives/\ndata/raw/\ndata/processed/\n"
            "runs/\ncheckpoints/\n*.pt\n*.pth\n*.onnx\n"
            "*.zip\n*.tar\n*.tar.gz\nreports/private/\n"
        ),
        "data/README.md": (
            "# Data\n\n"
            "StreetView-Waste requires authorized access. Do not commit raw data or credentials.\n"
            "Archives and the frozen split manifest are stored separately in Google Drive.\n"
            "Prepared training data is staged on the active runtime's local disk.\n"
        ),
        "configs/detection.yaml": "# Container detector training settings.\n",
        "configs/tracking.yaml": "# Tracker and counting settings.\n",
        "configs/segmentation.yaml": "# Overflow instance segmentation settings.\n",
        "model_card/MODEL_CARD.md": "# WasteScope Model Card\n\nComplete after model evaluation.\n",
        "src/__init__.py": "",
    }

    modules = {
        "data/annotations.py": "Annotation parsing and class mapping.",
        "detection/detector.py": "Container detection inference.",
        "tracking/tracker.py": "Container tracking across frames.",
        "tracking/counter.py": "Count containers within a sequence.",
        "segmentation/segmenter.py": "Overflow instance segmentation inference.",
        "pipeline/video.py": "Combine detection, tracking, counting, and segmentation.",
        "evaluation/metrics.py": "Task evaluation and metric reporting.",
        "api/app.py": "FastAPI application.",
    }
    for module, description in modules.items():
        files[f"src/{module}"] = f'"""{description}"""\n'
        files[f"src/{Path(module).parent.as_posix()}/__init__.py"] = ""

    scripts = {
        "prepare_dataset.py": "Prepare task data using the frozen split manifest.",
        "train_detector.py": "Train the container detector.",
        "train_segmenter.py": "Train the Overflow instance segmenter.",
        "evaluate.py": "Evaluate a trained model or tracker.",
        "infer_video.py": "Run the combined pipeline on a video sequence.",
        "export_onnx.py": "Export and verify an ONNX model.",
    }
    for filename, description in scripts.items():
        files[f"scripts/{filename}"] = (
            f'"""{description}"""\n\n'
            'if __name__ == "__main__":\n'
            f'    raise SystemExit("{filename} is a placeholder. Add its implementation before running.")\n'
        )

    notebooks = {
        "01_dataset_audit.ipynb": "Dataset audit",
        "02_video_distribution_and_split.ipynb": "Video distribution and split",
        "03_detection_baseline.ipynb": "Detection baseline",
        "04_tracking_evaluation.ipynb": "Tracking and counting evaluation",
        "05_segmentation_baseline.ipynb": "Overflow segmentation baseline",
    }
    for filename, title in notebooks.items():
        notebook = {
            "cells": [{"cell_type": "markdown", "id": "intro", "metadata": {},
                       "source": [f"# {title}\n"]}],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4, "nbformat_minor": 5,
        }
        files[f"notebooks/{filename}"] = json.dumps(notebook, indent=2) + "\n"

    for folder in ("tests", "reports/figures", "reports/metrics", "reports/failure_cases", "assets/demo"):
        files[f"{folder}/.gitkeep"] = ""

    created = preserved = 0
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
            created += 1
        except FileExistsError:
            if path.is_dir():
                raise IsADirectoryError(f"Expected a file: {path}")
            preserved += 1

    print(f"Project: {root}")
    print(f"Created: {created} files | Preserved: {preserved} existing files")
    print("Existing code and notebooks were not changed. Empty files and stubs are placeholders.")
    print("LICENSE is left blank until you choose a code license.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("wastescope-urban-intelligence"))
    create_project(parser.parse_args().root.expanduser().resolve())