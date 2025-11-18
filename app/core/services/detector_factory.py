from app.core.interfaces import IDetector
from app.infrastructure.detectors.yolo_detector import YoloDetector
from app.infrastructure.detectors.faster_rcnn_detector import FasterRCNNDetector

class DetectorFactory:
    @staticmethod
    def create(architecture: str) -> IDetector:
        arch = architecture.lower().strip()

        if arch in ["yolo"]:
            return YoloDetector()
        elif arch in ["faster_rcnn", "fasterrcnn", "faster_rcnn", "faster_rcnn"]:
            return FasterRCNNDetector()
        else:
            raise ValueError(f"Unsupported model architecture: {architecture}")