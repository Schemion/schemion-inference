from unittest.mock import MagicMock
from uuid import UUID

from PIL import Image

from app.core.enums import TaskStatus
from app.core.use_cases import DetectorInferenceUseCase
from app.infrastructure.services.image_tiler_service import ImageTilerService


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
        "confidence": 0.42,
        "max_image_size": 1024,
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
    detector.predict.assert_called_once_with(tile_mock.image, confidence=0.42)

    result_repo.save.assert_called_once()
    args, kwargs = result_repo.save.call_args
    result_data = args[0]
    assert result_data["task_id"] == task_id
    assert result_data["model_id"] == model_id
    assert result_data["model_arch"] == "yolo"
    assert len(result_data["predictions"]) == 1
    assert result_data["image_width"] == 640
    assert result_data["image_height"] == 480
    assert result_data["original_image_width"] == 640
    assert result_data["original_image_height"] == 480
    assert result_data["processed_image_width"] == 640
    assert result_data["processed_image_height"] == 480
    assert result_data["resize_scale"] == 1.0
    assert result_data["confidence"] == 0.42
    assert result_data["max_image_size"] == 1024

    assert status_update["task_id"] == task_id
    assert status_update["task_type"] == "inference"
    assert status_update["status"] == TaskStatus.succeeded.value
    assert status_update["output_path"] == result_path
    assert status_update["error_msg"] is None

    weights_loader.delete.assert_called_once_with(weights_path)


def test_detector_inference_use_case_uses_architecture_default_confidence():
    def _run_for_architecture(architecture: str) -> float:
        storage = MagicMock()
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

        model_entity = MagicMock()
        model_entity.architecture = architecture
        model_entity.architecture_profile = "default"
        model_entity.minio_model_path = "model.pt"
        model_entity.classes = []
        model_repo.get_by_id.return_value = model_entity

        image = Image.new("RGB", (320, 240))
        image_loader.load.return_value = image
        weights_loader.load.return_value = "/tmp/model.pt"

        tile_mock = MagicMock()
        tile_mock.image = image
        tile_mock.x = 0
        tile_mock.y = 0
        image_tiler.tile.return_value = [tile_mock]
        image_tiler.merge_predictions.return_value = []

        detector = MagicMock()
        detector.predict.return_value = []
        detector_factory.create.return_value = detector
        result_repo.save.return_value = "inference/result.json"

        use_case.execute(
            {
                "task_id": "12345678-1234-5678-1234-567812345678",
                "model_id": "87654321-4321-8765-4321-876543214321",
                "input_path": "input/test_image.jpg",
            }
        )

        return detector.predict.call_args.kwargs["confidence"]

    assert _run_for_architecture("yolo") == 0.25
    assert _run_for_architecture("faster_rcnn") == 0.5


def test_detector_inference_use_case_resizes_max_side_and_scales_bboxes_to_original_coordinates():
    storage = MagicMock()
    model_repo = MagicMock()
    image_loader = MagicMock()
    weights_loader = MagicMock()
    result_repo = MagicMock()
    detector_factory = MagicMock()

    use_case = DetectorInferenceUseCase(
        storage=storage,
        model_repo=model_repo,
        detector_factory=detector_factory,
        image_loader=image_loader,
        weights_loader=weights_loader,
        result_repo=result_repo,
        image_tiler=ImageTilerService(),
    )

    task_id = "12345678-1234-5678-1234-567812345678"
    model_id = "87654321-4321-8765-4321-876543214321"
    model_entity = MagicMock()
    model_entity.architecture = "yolo"
    model_entity.architecture_profile = "default"
    model_entity.minio_model_path = "yolo.pt"
    model_entity.classes = []
    model_repo.get_by_id.return_value = model_entity

    image_loader.load.return_value = Image.new("RGB", (1000, 500))
    weights_loader.load.return_value = "/tmp/yolo.pt"

    detector = MagicMock()

    def _predict(tile_image, confidence=None):
        assert tile_image.size == (500, 250)
        assert confidence == 0.25
        return [{"class": "part", "confidence": 0.9, "bbox": [10, 20, 110, 120]}]

    detector.predict.side_effect = _predict
    detector_factory.create.return_value = detector
    result_repo.save.return_value = "inference/result.json"

    status_update = use_case.execute(
        {
            "task_id": task_id,
            "model_id": model_id,
            "input_path": "input/test_image.jpg",
            "max_image_size": 500,
        }
    )

    assert status_update["status"] == TaskStatus.succeeded.value
    result_data = result_repo.save.call_args[0][0]
    assert result_data["original_image_width"] == 1000
    assert result_data["original_image_height"] == 500
    assert result_data["processed_image_width"] == 500
    assert result_data["processed_image_height"] == 250
    assert result_data["resize_scale"] == 0.5
    assert result_data["predictions"][0]["bbox"] == [20.0, 40.0, 220.0, 240.0]
