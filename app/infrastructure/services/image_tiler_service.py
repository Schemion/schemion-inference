from typing import List

from PIL import Image

from app.core.entities.tile import Tile
from app.core.interfaces.image_tiler_interface import IImageTilerInterface


class ImageTilerService(IImageTilerInterface):
    def __init__(self, tile_size: int = 640, overlap: int = 100):
        self.tile_size = tile_size
        self.overlap = overlap

    def tile(self, image: Image.Image) -> List[Tile]:
        tiles = []
        width, height = image.size

        if width < self.tile_size and height < self.tile_size:
            return [Tile(image, 0,0, width, height)]

        step = max(1, self.tile_size - self.overlap)

        y = 0
        while y < height:
            x = 0
            while x < width:
                x_end = min(x + self.tile_size, width)
                y_end = min(y + self.tile_size, height)

                if x_end - x < self.tile_size:
                    x = max(0, width - self.tile_size)
                    x_end = width

                if y_end - y < self.tile_size:
                    y = max(0, height - self.tile_size)
                    y_end = height

                tile_img = image.crop((x, y, x_end, y_end))
                tiles.append(Tile(tile_img, x, y, x_end - x, y_end - y))
                x += step
            y += step

        return tiles

    def shift_predictions(self, predictions, offset_x, offset_y):
        shifted_predictions = []
        for prediction in predictions:
            new_prediction = prediction.copy()

            if "bbox" in new_prediction:
                x1, y1, x2, y2 = new_prediction["bbox"]
                new_prediction["bbox"] = [
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y
                ]
            # даже туду не буду писать, но если будут масочные модели, но потом добавить для маски, тут типа пока пусто
            shifted_predictions.append(new_prediction)

        return shifted_predictions

    def merge_predictions(self, predictions: List[dict]):
        merged_predictions = []
        for prediction in predictions:
            merged_predictions.extend(prediction)

        return merged_predictions
