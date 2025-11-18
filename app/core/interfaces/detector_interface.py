from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Dict, Any

class IDetector(ABC):
    @abstractmethod
    def load_model(self, model_weights_path: str) -> None:
        ...

    @abstractmethod
    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        ...