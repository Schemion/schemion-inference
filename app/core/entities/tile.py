from dataclasses import dataclass
from PIL import Image


@dataclass
class Tile:
    image: Image.Image
    x: int
    y: int
    width: int
    height: int