from uuid import UUID
from datetime import datetime, timezone
import logging

from app.core.interfaces import IDetectorFactory, IImageLoader, IModelWeightsLoader, IInferenceResult, IImageTilerInterface
from app.core.interfaces.storage_interface import IStorageRepository
from app.core.interfaces.model_interface import IModelRepository
from app.core.interfaces.task_interface import ITaskRepository

logger = logging.getLogger(__name__)

class DetectorInferenceUseCase:
    def __init__(self,
             storage: IStorageRepository,
             image_loader: IImageLoader,
             weights_loader: IModelWeightsLoader,
             result_repo: IInferenceResult,
             task_repo: ITaskRepository,
             model_repo: IModelRepository,
             detector_factory: IDetectorFactory,
             image_tiler: IImageTilerInterface
        ):
        self.image_loader = image_loader
        self.weights_loader = weights_loader
        self.result_repo = result_repo
        self.storage = storage
        self.task_repo = task_repo
        self.model_repo = model_repo
        self.detector_factory = detector_factory,
        self.image_tiler = image_tiler

    def execute(self, message: dict) -> None:
        task_id = UUID(message["task_id"])
        model_id = UUID(message["model_id"])
        image_path = message["input_path"]

        logger.info(f"Inference task {task_id} started")

        task = self.task_repo.get_by_id(task_id)
        model = self.model_repo.get_by_id(model_id)

        logger.info(f"Task {task_id} - loading image")
        try:

            image = self.image_loader.load(image_path)

            logger.info(f"Task {task_id} - downloading weights to {model.minio_model_path}")

            weights_file = self.weights_loader.load(model.minio_model_path)

            logger.info(f"Task {task_id} - creating detector")

            detector = self.detector_factory.create(
                architecture=model.architecture,
                architecture_profile=model.architecture_profile,
                classes=model.classes or [],
            )
            detector.load_model(weights_file)

            logger.info(f"Task {task_id} - generating tiles")
            tiles = self.image_tiler.tile(image)

            all_shifted_predictions = []

            logger.info(f"Task {task_id} - running prediction")
            for tile in tiles:
                predictions = detector.predict(tile)
                shifted = self.image_tiler.shift_predictions(
                    predictions,
                    offset_x=tile.x,
                    offset_y=tile.y
                )

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

            task.output_path = object_path
            task.updated_at = datetime.now(timezone.utc)
            self.task_repo.update(task)

            logger.info(f"Inference success")

            self.weights_loader.delete(weights_file)
        except Exception as exc:
            logger.exception(f"Task {task_id} - inference failed: {exc}")
            task.error_msg = str(exc)
            task.updated_at = datetime.now(timezone.utc)
            self.task_repo.update(task)