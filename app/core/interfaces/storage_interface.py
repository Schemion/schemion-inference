from typing import Protocol

class IStorageRepository(Protocol):
    def upload_file(self, file_data: bytes, filename: str, content_type: str, bucket: str) -> str:
        ...

    def delete_file(self, object_name: str, bucket: str) -> None:
        ...

    def get_file_url(self, object_name: str, bucket: str) -> str:
        ...