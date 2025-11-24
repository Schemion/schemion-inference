from uuid import UUID
from datetime import datetime, timezone
from PIL import Image
from io import BytesIO
import json
import tempfile
import logging
import os

from app.core.interfaces import IDetectorFactory
from app.core.interfaces.storage_interface import IStorageRepository
from app.core.interfaces.model_interface import IModelRepository
from app.core.interfaces.task_interface import ITaskRepository

logger = logging.getLogger(__name__)

class InferenceUseCase:
    def __init__(self,storage: IStorageRepository, task_repo: ITaskRepository, model_repo: IModelRepository, detector_factory: IDetectorFactory,
            images_bucket: str = "schemas-images", # потом поменять так как изменил тему немного
            models_bucket: str = "models",
            results_bucket: str = "inference-results", # его пока нет но надо бы чтобы был
        ):
        self.storage = storage
        self.task_repo = task_repo
        self.model_repo = model_repo
        self.detector_factory = detector_factory
        self.images_bucket = images_bucket
        self.models_bucket = models_bucket
        self.results_bucket = results_bucket

    async def execute(self, message: dict) -> None:
        task_id = UUID(message["task_id"])
        model_id = UUID(message["model_id"])
        input_path = message["input_path"]

        logger.info(f"Inference task {task_id} started")

        task = self.task_repo.get_by_id(task_id)
        if not task:
            logger.error(f"Task {task_id} not found in database")
            return

        weights_path = None
        try:
            model_entity = self.model_repo.get_by_id(model_id)
            if not model_entity:
                raise ValueError(f"Model with id {model_id} not found")

            architecture = model_entity.architecture.lower().strip()

            classes = [] # TODO: надо обновить orm и добавить поля для классов, так как стоковые модели faster-rcnn не имеют зашитых в модель классов

            extension = ".pt" if "yolo" in architecture else ".pth"

            logger.info(f"Task {task_id} - loading image '{input_path}'")

            image_bytes = self.storage.download_file_to_bytes(input_path, self.images_bucket)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
            weights_path = tmp_file.name
            tmp_file.close()

            logger.info(f"Task {task_id} - downloading weights to {weights_path}")

            self.storage.download_file_to_path(
                object_name=model_entity.minio_model_path,
                bucket=self.models_bucket,
                local_path=weights_path,
            )

            logger.info(f"Task {task_id} - creating detector '{architecture}'")

            detector = self.detector_factory.create(architecture=architecture, classes=classes)
            detector.load_model(weights_path)

            logger.info(f"Task {task_id} - running prediction")
            predictions = detector.predict(image)

            result_data = {
                "task_id": str(task_id),
                "model_id": str(model_id),
                "model_arch": architecture,
                "predictions": predictions,
                "image_width": image.width,
                "image_height": image.height,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }

            result_json = json.dumps(result_data, ensure_ascii=False, indent=2)
            result_bytes = result_json.encode("utf-8")

            result_object_name = self.storage.upload_file(
                file_data=result_bytes,
                filename=f"inference_{task_id}.json",
                content_type="application/json",
                bucket=self.results_bucket,
            )
            task.output_path = result_object_name
            task.updated_at = datetime.now(timezone.utc)
            self.task_repo.update(task)

            logger.info(f"Inference success")

        except Exception as exc:
            logger.exception(f"Task {task_id} - inference failed: {exc}")
            task.error_msg = str(exc)
            task.updated_at = datetime.now(timezone.utc)
            self.task_repo.update(task)

        finally:
            if weights_path and os.path.exists(weights_path):
                try:
                    os.unlink(weights_path)
                except Exception as cleanup_exc:
                    logger.warning(f"Failed to delete temporary file {weights_path}: {cleanup_exc}")