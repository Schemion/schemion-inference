from uuid import UUID
from datetime import datetime, timezone
import logging

from app.core.enums import TaskStatus
from app.core.interfaces import IDetectorFactory, IImageLoader, IModelWeightsLoader, IInferenceResult, IImageTilerInterface
from app.core.interfaces.storage_interface import IStorageRepository
from app.core.interfaces.model_interface import IModelRepository

logger = logging.getLogger(__name__)

class DetectorInferenceUseCase:
    def __init__(self,
             storage: IStorageRepository,
             image_loader: IImageLoader,
             weights_loader: IModelWeightsLoader,
             result_repo: IInferenceResult,
             model_repo: IModelRepository,
             detector_factory: IDetectorFactory,
             image_tiler: IImageTilerInterface
        ):
        self.image_loader = image_loader
        self.weights_loader = weights_loader
        self.result_repo = result_repo
        self.storage = storage
        self.model_repo = model_repo
        self.detector_factory = detector_factory
        self.image_tiler = image_tiler

    def execute(self, message: dict) -> dict:
        task_id_raw = str(message["task_id"])
        weights_file = None

        try:
            task_id = UUID(task_id_raw)
            model_id = UUID(message["model_id"])
            image_path = message["input_path"]

            logger.info(f"Inference task {task_id} started")

            model = self.model_repo.get_by_id(model_id)
            if not model:
                raise RuntimeError(f"Model {model_id} not found")

            logger.info(f"Task {task_id} - loading image")
            image = self.image_loader.load(image_path)

            logger.info(f"Task {task_id} - downloading weights to {model.minio_model_path}")

            weights_file = self.weights_loader.load(str(model.minio_model_path))

            logger.info(f"Task {task_id} - creating detector")

            detector = self.detector_factory.create(
                architecture=str(model.architecture),
                architecture_profile=str(model.architecture_profile),
                classes=model.classes or [],
            )
            detector.load_model(weights_file)


            all_shifted_predictions = []

            logger.info(f"Task {task_id} - running prediction")

            for tile in self.image_tiler.tile(image):
                logger.info(f"Running predict on tile at {tile.x},{tile.y} size={tile.image.width}x{tile.image.height}")
                predictions = detector.predict(tile.image)
                shifted = self.image_tiler.shift_predictions(predictions, tile.x, tile.y)
                all_shifted_predictions.append(shifted)

            logger.info(f"Task {task_id} - merging predictions")
            merged_predictions = self.image_tiler.merge_predictions(all_shifted_predictions)

            result = {
                "task_id": str(task_id),
                "model_id": str(model_id),
                "model_arch": model.architecture,
                "predictions": merged_predictions,
                "image_width": image.width,
                "image_height": image.height,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }

            object_path = self.result_repo.save(
                result,
                filename=f"inference_{task_id}.json"
            )

            logger.info(f"Inference success")

            return self._status_update(
                task_id=task_id_raw,
                status=TaskStatus.succeeded,
                output_path=object_path,
            )
        except Exception as exc:
            logger.exception(f"Task {task_id_raw} - inference failed: {exc}")
            return self._status_update(
                task_id=task_id_raw,
                status=TaskStatus.failed,
                error_msg=str(exc),
            )
        finally:
            if weights_file:
                self.weights_loader.delete(weights_file)

    @staticmethod
    def _status_update(
        task_id: str,
        status: TaskStatus,
        output_path: str | None = None,
        error_msg: str | None = None,
    ) -> dict:
        return {
            "task_id": task_id,
            "task_type": "inference",
            "status": status.value,
            "output_path": output_path,
            "error_msg": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
