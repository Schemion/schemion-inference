from abc import ABC, abstractmethod
from typing import Optional, List

from app.core.interfaces import IDetector

class IDetectorFactory(ABC):
    @abstractmethod
    def create(self, architecture: str, architecture_profile: Optional[str] = None, classes: Optional[List[str]] = None) -> IDetector:
        ...