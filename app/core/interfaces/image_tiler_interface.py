from abc import ABC, abstractmethod
from typing import List

from PIL import Image

from app.core.entities.tile import Tile


class IImageTilerInterface(ABC):
    @abstractmethod
    def tile(self, image: Image.Image) -> List[Tile]:
        ...

    @abstractmethod
    def shift_predictions(self, predictions, offset_x, offset_y):
        ...

    @abstractmethod
    def merge_predictions(self, all_shifted_predictions):
        ...


