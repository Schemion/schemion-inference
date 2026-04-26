from io import BytesIO
from unittest.mock import MagicMock

from PIL import Image

from app.infrastructure.services.image_loader_service import ImageLoader


def test_load_downloads_image_bytes_and_converts_to_rgb():
    storage = MagicMock()

    image = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    storage.download_file_to_bytes.return_value = buffer.getvalue()

    loader = ImageLoader(storage=storage, bucket="schemas-images")

    loaded = loader.load("input/test.png")

    storage.download_file_to_bytes.assert_called_once_with("input/test.png", "schemas-images")
    assert loaded.mode == "RGB"
    assert loaded.size == (10, 10)
