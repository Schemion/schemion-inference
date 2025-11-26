from abc import ABC, abstractmethod
from typing import Any

class IImageLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> Any:
        ...
