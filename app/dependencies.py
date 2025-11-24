from app.core.interfaces import IDetectorFactory
from app.infrastructure.factories.detector_factory import DetectorFactory

def get_detector_factory() -> IDetectorFactory:
    return DetectorFactory()