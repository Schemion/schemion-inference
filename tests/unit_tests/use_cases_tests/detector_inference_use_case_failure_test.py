from unittest.mock import MagicMock

from app.core.enums import TaskStatus
from app.core.use_cases import DetectorInferenceUseCase


def test_detector_inference_use_case_marks_task_failed_when_image_loading_crashes():
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
        task_repo=task_repo,
        model_repo=model_repo,
        detector_factory=detector_factory,
        image_loader=image_loader,
        weights_loader=weights_loader,
        result_repo=result_repo,
        image_tiler=image_tiler,
    )

    task = MagicMock()
    task_repo.get_by_id.return_value = task
    model_repo.get_by_id.return_value = MagicMock()
    image_loader.load.side_effect = RuntimeError("cannot read image")

    message = {
        "task_id": "12345678-1234-5678-1234-567812345678",
        "model_id": "87654321-4321-8765-4321-876543214321",
        "input_path": "input/broken.jpg",
    }

    use_case.execute(message)

    assert task.status == TaskStatus.failed.value
    assert task.error_msg == "cannot read image"
    assert task.updated_at is not None
    task_repo.update.assert_called_once_with(task)

    weights_loader.delete.assert_not_called()
    result_repo.save.assert_not_called()
