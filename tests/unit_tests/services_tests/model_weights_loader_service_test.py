import os
from pathlib import Path
from unittest.mock import MagicMock

from app.infrastructure.services.model_weights_loader_service import ModelWeightsLoader


def test_load_creates_temp_file_with_source_extension_and_downloads_weights():
    storage = MagicMock()
    loader = ModelWeightsLoader(storage=storage, bucket="models")

    path = loader.load("models/yolo.pt")

    try:
        assert path.endswith(".pt")
        storage.download_file_to_path.assert_called_once_with(
            object_name="models/yolo.pt",
            bucket="models",
            local_path=path,
        )
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_delete_removes_file_if_exists(tmp_path):
    storage = MagicMock()
    loader = ModelWeightsLoader(storage=storage, bucket="models")
    file_path = tmp_path / "weights.pth"
    file_path.write_text("weights")

    loader.delete(str(file_path))

    assert not file_path.exists()


def test_delete_ignores_missing_file(tmp_path):
    storage = MagicMock()
    loader = ModelWeightsLoader(storage=storage, bucket="models")
    missing_path = tmp_path / "missing.pth"

    loader.delete(str(missing_path))

    assert not Path(missing_path).exists()
