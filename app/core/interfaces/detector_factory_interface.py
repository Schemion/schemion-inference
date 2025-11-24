from abc import ABC, abstractmethod
from app.core.interfaces import IDetector

class IDetectorFactory(ABC):
    @abstractmethod
    def create(self, architecture: str, **kwargs) -> IDetector:
        ...