from app.core.interfaces.detector_factory_interface import IDetectorFactory
from app.infrastructure.detectors.yolo_detector import YoloDetector
from app.infrastructure.detectors.fasterrcnn_detector import FasterRCNNDetector
from app.infrastructure.models_config import FASTERRCNN_ARCHITECTURES, ARCHITECTURE_ALIASES, SUPPORTED_ARCHITECTURES
from app.core.interfaces.detector_interface import IDetector


class DetectorFactory(IDetectorFactory):
    def __init__(self):
        self._architecture_map = FASTERRCNN_ARCHITECTURES
        self._aliases = ARCHITECTURE_ALIASES
        self._supported = SUPPORTED_ARCHITECTURES

    def create(self, architecture: str, **kwargs) -> IDetector:
        arch = architecture.lower().strip()
        canonical_name = self._resolve_alias(arch)

        if self._is_yolo(canonical_name):
            return self._create_yolo()

        if canonical_name in self._architecture_map:
            return self._create_fasterrcnn(canonical_name, **kwargs)

        raise ValueError(
            f"Unsupported architecture: '{architecture}'. "
            f"Supported: {', '.join(self._supported)}"
        )

    def _resolve_alias(self, arch: str) -> str:
        return self._aliases.get(arch, arch)

    @staticmethod
    def _is_yolo(arch: str) -> bool:
        return "yolo" in arch

    @staticmethod
    def _create_yolo() -> IDetector:
        return YoloDetector()

    def _create_fasterrcnn(self, arch_key: str, **kwargs) -> IDetector:
        name = self._architecture_map[arch_key]
        return FasterRCNNDetector(architecture=name, **kwargs)