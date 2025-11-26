from PIL import Image
from io import BytesIO
from app.core.interfaces import IImageLoader
from app.core.interfaces.storage_interface import IStorageRepository

class ImageLoader(IImageLoader):
    def __init__(self, storage: IStorageRepository, bucket: str):
        self.storage = storage
        self.bucket = bucket

    def load(self, path: str):
        bytes_data = self.storage.download_file_to_bytes(path, self.bucket)
        return Image.open(BytesIO(bytes_data)).convert("RGB")
