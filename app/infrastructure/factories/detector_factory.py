from typing import Optional, List

from app.core.interfaces.detector_factory_interface import IDetectorFactory
from app.infrastructure.detectors.yolo_detector import YoloDetector
from app.infrastructure.detectors.fasterrcnn_detector import FasterRCNNDetector
from app.infrastructure.models_config import FASTERRCNN_ARCHITECTURES, ARCHITECTURE_ALIASES, SUPPORTED_ARCHITECTURES, FASTERRCNN_ALIASES
from app.core.interfaces.detector_interface import IDetector


class DetectorFactory(IDetectorFactory):
    def __init__(self):
        self._architecture_map = FASTERRCNN_ARCHITECTURES
        self._aliases = ARCHITECTURE_ALIASES
        self._supported = SUPPORTED_ARCHITECTURES
        self._fasterrcnn_aliases = FASTERRCNN_ALIASES

    def create(self, architecture: str, architecture_profile: Optional[str] = None, classes: Optional[List[str]] = None) -> IDetector:
        arch = architecture.lower().strip()
        profile_key = architecture_profile.lower() if architecture_profile else "default"
        true_arch = self._resolve_arch_alias(arch)
        true_profile = self._resolve_profile_alias(profile_key)

        if self._is_yolo(true_arch):
            return self._create_yolo()

        if true_profile in self._architecture_map:
            return self._create_fasterrcnn(true_profile, classes)

        raise ValueError(
            f"Unsupported architecture: '{architecture}'. "
            f"Supported: {', '.join(self._supported)}"
        )

    def _resolve_arch_alias(self, arch: str) -> str:
        return self._fasterrcnn_aliases.get(arch, arch)

    def _resolve_profile_alias(self, profile: str) -> str:
        return self._aliases.get(profile, profile)

    @staticmethod
    def _is_yolo(arch: str) -> bool:
        return "yolo" in arch

    @staticmethod
    def _create_yolo() -> IDetector:
        return YoloDetector()

    def _create_fasterrcnn(self, profile: str, classes: Optional[List[str]] = None) -> IDetector:
        arch_profile = self._architecture_map[profile]
        return FasterRCNNDetector(architecture=arch_profile, classes=classes)