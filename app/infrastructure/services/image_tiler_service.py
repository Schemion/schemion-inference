from typing import List

from PIL import Image

from app.core.entities.tile import Tile
from app.core.interfaces.image_tiler_interface import IImageTilerInterface


class ImageTilerService(IImageTilerInterface):
    def __init__(self, tile_size: int = 640, overlap: int = 100):
        self.tile_size = tile_size
        self.overlap = overlap

    def tile(self, image: Image.Image):
        width, height = image.size

        if width <= self.tile_size and height <= self.tile_size:
            yield Tile(image, 0, 0, width, height)
            return

        step = max(1, self.tile_size - self.overlap)

        x_positions = []
        x = 0
        while x < width:
            if x + self.tile_size > width:
                x = max(0, width - self.tile_size)
            x_positions.append(x)
            if x + self.tile_size >= width:
                break
            x += step

        y_positions = []
        y = 0
        while y < height:
            if y + self.tile_size > height:
                y = max(0, height - self.tile_size)
            y_positions.append(y)
            if y + self.tile_size >= height:
                break
            y += step

        for y in y_positions:
            for x in x_positions:
                x_end = min(x + self.tile_size, width)
                y_end = min(y + self.tile_size, height)

                tile_img = image.crop((x, y, x_end, y_end))
                yield Tile(tile_img, x, y, x_end - x, y_end - y)

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
