from unittest.mock import MagicMock
from uuid import UUID

from PIL import Image

from app.core.enums import TaskStatus
from app.core.use_cases import DetectorInferenceUseCase


def test_detector_inference_use_case():
    storage = MagicMock()
    task_repo = MagicMock()
    model_repo = MagicMock()
    image_loader = MagicMock()
    weights_loader = MagicMock()
    result_repo = MagicMock()
    detector_factory = MagicMock()
    image_tiler = MagicMock()

    use_case = DetectorInferenceUseCase(
        storage=storage,
        model_repo=model_repo,
        detector_factory=detector_factory,
        image_loader=image_loader,
        weights_loader=weights_loader,
        result_repo=result_repo,
        image_tiler=image_tiler,
    )

    task_id = "12345678-1234-5678-1234-567812345678"
    model_id = "87654321-4321-8765-4321-876543214321"
    input_path = "input/test_image.jpg"

    message = {
        "task_id": task_id,
        "model_id": model_id,
        "input_path": input_path,
    }

    model_entity = MagicMock()
    model_entity.architecture = "yolo"
    model_entity.architecture_profile = "default"
    model_entity.minio_model_path = "yolo.pt"
    model_entity.classes = []
    model_repo.get_by_id.return_value = model_entity

    image = Image.new("RGB", (640, 480))
    image_loader.load.return_value = image

    weights_path = "/tmp/yolo.pt"
    weights_loader.load.return_value = weights_path

    tile_mock = MagicMock()
    tile_mock.image = image
    tile_mock.x = 0
    tile_mock.y = 0
    image_tiler.tile.return_value = [tile_mock]

    detector = MagicMock()
    detector.predict.return_value = [{"label": "person", "confidence": 0.9, "bbox": [10, 20, 100, 150]}]
    detector_factory.create.return_value = detector

    image_tiler.merge_predictions.return_value = [{"label": "person", "confidence": 0.9, "bbox": [10, 20, 100, 150]}]

    result_path = f"inference-results/inference_{task_id}.json"
    result_repo.save.return_value = result_path

    status_update = use_case.execute(message)

    task_repo.assert_not_called()
    model_repo.get_by_id.assert_called_once_with(UUID(model_id))

    image_loader.load.assert_called_once_with(input_path)
    weights_loader.load.assert_called_once_with("yolo.pt")

    detector_factory.create.assert_called_once_with(
        architecture="yolo",
        architecture_profile="default",
        classes=[],
    )
    detector.load_model.assert_called_once_with(weights_path)
    detector.predict.assert_called_once()

    result_repo.save.assert_called_once()
    args, kwargs = result_repo.save.call_args
    result_data = args[0]
    assert result_data["task_id"] == task_id
    assert result_data["model_id"] == model_id
    assert result_data["model_arch"] == "yolo"
    assert len(result_data["predictions"]) == 1
    assert result_data["image_width"] == 640
    assert result_data["image_height"] == 480

    assert status_update["task_id"] == task_id
    assert status_update["task_type"] == "inference"
    assert status_update["status"] == TaskStatus.succeeded.value
    assert status_update["output_path"] == result_path
    assert status_update["error_msg"] is None

    weights_loader.delete.assert_called_once_with(weights_path)
