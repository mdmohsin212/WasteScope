from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from threading import RLock
import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

def check_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Expected an OpenCV BGR uint8 image.")

    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("Expected a non-empty HxWx3 BGR image.")

    return image


@dataclass(frozen=True, slots=True)
class Detections:
    data: np.ndarray
    image_shape: tuple[int, int]
    
    def records(self, names):
        return [
            {
                "class_id": int(row[5]),
                "class_name": names[int(row[5])],
                "confidence": float(row[4]),
                "xyxy": row[:4].tolist()
            }
            for row in self.data
        ]


class Runtime:
    def __init__(self, root=ROOT, *, models_dir=None, device="auto"):
        self.root = Path(ROOT)
        self.models_dir = (
            Path(models_dir) if models_dir
            else self.root / "models/weights"
        )
        self.manifest = json.loads((self.root / "models/manifest.json").read_text(encoding="utf-8-sig"))
        
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif isinstance(device, int) or str(device).isdigit():
            device = f"cuda:{device}"
        
        self.device = torch.device(device)
        self._models = {}
        self._lock = RLock()
    
    def config(self, task):
        path = self.root / self.manifest["configs"][task]["path"]
        return yaml.safe_load(path.read_text(encoding="utf-8-sig"))

    def names(self, model_id):
        return list(self.manifest["models"][model_id]["classes"])

    def _load(self, model_id):
        if model_id not in self._models:
            record = self.manifest["models"][model_id]
            path = self.models_dir / record["filename"]
            digest = hashlib.sha256()
            
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)

            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Checkpoint SHA256 mismatch: {path}")

            model = YOLO(str(path))
            names = [model.names[key] for key in sorted(model.names)]
            
            if (model.task != record["task"] or names != record["classes"]):
                raise ValueError(f"{model_id}: checkpoint task/class order mismatch")

            self._models[model_id] = model

        return self._models[model_id]
    

    def infer(self, model_id, images, settings, *, batch=1):
        for image in (images if isinstance(images, list) else [images]):
            check_image(image)
        
        options = {"classes": None, **settings}
        options["half"] = bool(options.get("half", False) and self.device.type == "cuda")
        
        with self._lock:
            return self._load(model_id).predict(
                images,
                **options,
                device=self.device,
                batch=batch,
                save=False,
                verbose=False,
                stream=False
            )