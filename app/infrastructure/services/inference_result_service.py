import json
from app.core.interfaces import IInferenceResult
from app.core.interfaces.storage_interface import IStorageRepository

class InferenceResultService(IInferenceResult):
    def __init__(self, storage: IStorageRepository, bucket: str):
        self.storage = storage
        self.bucket = bucket

    def save(self, result_data: dict, filename: str) -> str:
        json_bytes = json.dumps(result_data, ensure_ascii=False, indent=2).encode("utf-8")

        return self.storage.upload_file(
            file_data=json_bytes,
            filename=filename,
            content_type="application/json",
            bucket=self.bucket
        )
