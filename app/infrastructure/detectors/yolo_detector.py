from typing import List, Dict, Any
from app.core.interfaces.detector_interface import IDetector
from ultralytics import YOLO
from PIL import Image

class YoloDetector(IDetector):
    def __init__(self):
        self.model = None

    def load_model(self, model_weights_path: str) -> None:
        self.model = YOLO(model_weights_path)

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        results = self.model(image, verbose=False)[0]
        detections = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf.cpu())
            cls_id = int(box.cls.cpu())
            class_name = results.names[cls_id]

            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": xyxy.tolist(),
            })
        return detections