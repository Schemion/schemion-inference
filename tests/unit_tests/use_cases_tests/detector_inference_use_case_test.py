from io import BytesIO
from unittest.mock import MagicMock, patch
from uuid import UUID

from PIL import Image

from app.core.use_cases import DetectorInferenceUseCase


def test_detector_inference_use_case():
    storage = MagicMock()
    task_repo = MagicMock()
    model_repo = MagicMock()
    detector_factory = MagicMock()

    use_case = DetectorInferenceUseCase(
        storage=storage,
        task_repo=task_repo,
        model_repo=model_repo,
        detector_factory=detector_factory,


    )

    task_id = "12345678-1234-5678-1234-567812345678"
    model_id = "87654321-4321-8765-4321-876543214321"
    input_path = "input/test_image.jpg"

    message = {
        "task_id": task_id,
        "model_id": model_id,
        "input_path": input_path,
    }

    task = MagicMock()
    task.id = task_id
    task_repo.get_by_id.return_value = task

    model_entity = MagicMock()
    model_entity.architecture = "yolo"
    model_entity.architecture_profile = "v8"
    model_entity.minio_model_path = "yolo_v8.pt"
    model_repo.get_by_id.return_value = model_entity

    image = Image.new("RGB", (640, 480))
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    storage.download_file_to_bytes.return_value = image_bytes.getvalue()

    detector = MagicMock()
    detector.predict.return_value = [{"label": "person", "confidence": 0.9, "bbox": [10, 20, 100, 150]}]
    detector_factory.create.return_value = detector

    storage.upload_file.return_value = "inference-results/inference_12345678-1234-5678-1234-567812345678.json"

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        temp_file = MagicMock()
        temp_file.name = "/tmp/yolo.pt"
        mock_temp.return_value = temp_file

        use_case.execute(message)

    task_repo.get_by_id.assert_called_once_with(UUID(task_id))
    model_repo.get_by_id.assert_called_once_with(UUID(model_id))
    storage.download_file_to_bytes.assert_called_once_with(input_path, "schemas-images")
    storage.download_file_to_path.assert_called_once_with(
        object_name="yolo_v8.pt",
        bucket="models",
        local_path="/tmp/yolo.pt",
    )
    detector_factory.create.assert_called_once_with(
        architecture="yolo", architecture_profile="v8", classes=[]
    )
    detector.load_model.assert_called_once_with("/tmp/yolo.pt")
    detector.predict.assert_called_once()
    storage.upload_file.assert_called_once()
    assert task.output_path == "inference-results/inference_12345678-1234-5678-1234-567812345678.json"
    task_repo.update.assert_called()