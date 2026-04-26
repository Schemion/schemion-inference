import json
from unittest.mock import MagicMock

from app.infrastructure.services.inference_result_service import InferenceResultService


def test_save_serializes_json_and_uploads_to_storage():
    storage = MagicMock()
    storage.upload_file.return_value = "inference-results/file.json"
    service = InferenceResultService(storage=storage, bucket="inference-results")

    payload = {"label": "man", "score": 0.95}
    result = service.save(payload, filename="result.json")

    assert result == "inference-results/file.json"
    storage.upload_file.assert_called_once()

    kwargs = storage.upload_file.call_args.kwargs
    assert kwargs["filename"] == "result.json"
    assert kwargs["content_type"] == "application/json"
    assert kwargs["bucket"] == "inference-results"

    decoded = json.loads(kwargs["file_data"].decode("utf-8"))
    assert decoded == payload
