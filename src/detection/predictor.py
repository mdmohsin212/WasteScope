import numpy as np
from ..pipeline.runtime import Detections

class Detector:
    def __init__(self, runtime):
        self.runtime = runtime
        self.config = runtime.config("detection")
        self.model_id = self.config["model"]
        self.names = runtime.names(self.model_id)
    
    def raw(self, image):
        result = self.runtime.infer(
            self.model_id,
            image,
            self.config["inference"]
        )[0]
        
        data = result.boxes.data.cpu().numpy().astype(np.float32, copy=True)
        return Detections(data, tuple(image.shape[:2]))

    
    def filter(self, raw):
        policy = self.config["operating_point"]
        class_conf = policy.get("class_confidences", {})
        
        thresholds = np.array([
            class_conf.get(name, policy["default_confidence"])
            for name in self.names
        ])
        
        keep = (raw.data[:, 4] > thresholds[raw.data[:, 5].astype(int)])
        return Detections(raw.data[keep].copy(), raw.image_shape)

    def predict(self, image):
        return self.filter(self.raw(image))