from app.core.interfaces.detector_factory_interface import IDetectorFactory
from app.core.interfaces import IDetector


class DetectionService:
    def __init__(self, detector_factory: IDetectorFactory, architecture: str, model_path: str):
        self.detector: IDetector = detector_factory.create(architecture,model_path=model_path)

    def detect_objects(self, image):
        return self.detector.predict(image)