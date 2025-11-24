from torchvision import transforms
from typing import List, Dict, Any
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from app.core.interfaces import IDetector


class FasterRCNNDetector(IDetector):
    def __init__(self, architecture: str, classes: List[str] = None):
        self._ARCHITECTURE_MAP = {
            "fasterrcnn_resnet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
            "fasterrcnn_resnet50_fpn_v2": torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
            "fasterrcnn_mobilenet_v3_large_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
            "fasterrcnn_mobilenet_v3_large_320_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
        }

        if architecture not in self._ARCHITECTURE_MAP:
            raise ValueError(
                f"Unsupported Faster R-CNN architecture: {architecture}. "
                f"Available options: {list(self._ARCHITECTURE_MAP.keys())}"
            )

        self.architecture = architecture
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["__background__"] + (classes or [])

    def load_model(self, model_weights_path: str) -> None:
        state_dict = torch.load(model_weights_path, map_location=self.device)

        num_classes = self._determine_num_classes(state_dict)

        model_constructor = self._ARCHITECTURE_MAP[self.architecture]

        model = model_constructor(weights=None, weights_backbone=None)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model = model

    def predict(self, image: Image.Image, conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_tensor = transform(image).to(self.device)

        with torch.no_grad():
            predictions = self.model([img_tensor])[0]

        detections = []
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score < conf_threshold:
                continue

            x1, y1, x2, y2 = box
            class_name = self.classes[label] if self.classes else str(label)
            if class_name == "__background__":
                continue

            detections.append({
                "class": class_name,
                "confidence": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        return detections

    @staticmethod
    def _determine_num_classes(state_dict: dict) -> int:
        if "roi_heads.box_predictor.cls_score.weight" in state_dict:
            return state_dict["roi_heads.box_predictor.cls_score.weight"].shape[0]

        elif "roi_heads.box_predictor.bbox_pred.weight" in state_dict:
            bbox_weight_shape = state_dict["roi_heads.box_predictor.bbox_pred.weight"].shape[0]
            return bbox_weight_shape // 4

        raise ValueError("Unable to determine number of classes from model weights.")