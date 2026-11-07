import argparse
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import sys
import tempfile

os.environ["YOLO_AUTOINSTALL"] = "False"
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.outputs import file_sha256, write_json
from src.pipeline.runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models/weights")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/onnx")
    parser.add_argument("--threads", type=int, default=4)
    
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if version("ultralytics") != "8.4.1":
        raise RuntimeError("Use the project's pinned ultralytics==8.4.1 environment.")
    
    torch.set_num_threads(args.threads)
    runtime = Runtime(device="cpu", models_dir=args.models_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(format="onnx", imgsz=960, dynamic=True, half=False, batch=1, opset=17, simplify=False, nms=False, device="cpu", max_det=300)
    
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    exported = {}
    
    for model_id, record in runtime.manifest["models"].items():
        source = args.models_dir / record["filename"]
        runtime._load(model_id)
        with tempfile.TemporaryDirectory(prefix="wastescope_onnx_") as temporary:
            staged = Path(temporary) / source.name
            shutil.copy2(source, staged)
            result = YOLO(str(staged)).export(**settings)
            
            if not result:
                raise RuntimeError(f"Export failed: {model_id}")
            
            path = Path(result)
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
                raise RuntimeError("Expected a self-contained ONNX file.")
            
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            shapes = [(1, 3, 960, 960)]
            
            if record["task"] == "detect":
                shapes.append((1, 3, 544, 960))
            if model_id == "segmenter_roi":
                shapes.append((2, 3, 960, 960))
            for shape in shapes:
                outputs = session.run(None, {session.get_inputs()[0].name: np.zeros(shape, np.float32)})
                if any(out.shape[0] != shape[0] or not np.isfinite(out).all() for out in outputs):
                    raise RuntimeError(f"Invalid ONNX outputs: {model_id}, shape={shape}")
                
            del session
            digest = file_sha256(path)
            filename = f"{source.stem}.{digest[:12]}.onnx"
            shutil.copy2(path, args.output_dir / filename)
            exported[model_id] = {
                "filename": filename, "sha256": digest,
                "source_sha256": record["sha256"], "task": record["task"],
                "classes": record["classes"], "tested_input_shapes": shapes
            }
        print("Exported and graph-tested:", model_id, flush=True)
    
    manifest = {
        "schema_version": 1, "source_release_id": runtime.manifest["release_id"],
        "export_settings": settings, "models": exported,
        "versions": {name: version(name) for name in ("torch", "ultralytics", "onnx", "onnxruntime")},
        "validation": "Graph/finite-output checks passed; run check_onnx.py for real-image conversion parity."
    }
    
    write_json(args.output_dir / "manifest.json", manifest)
    print("ONNX bundle:", args.output_dir.resolve())

if __name__ == "__main__":
    main()