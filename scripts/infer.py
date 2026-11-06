import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.runner import WasteScopePipeline
from src.pipeline.runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("mode", choices=["image", "video"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models-dir", type=Path)

    args = parser.parse_args()

    runtime = Runtime(device=args.device, models_dir=args.models_dir)
    pipeline = WasteScopePipeline(runtime)

    summary = getattr(pipeline, args.mode)(args.input,output_root=args.output_root,)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()