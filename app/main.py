import asyncio
from app.core.use_cases import DetectorInferenceUseCase
from app.infrastructure.cloud_storage import MinioStorage
from app.infrastructure.persistence.repositories import ModelRepository, TaskRepository
from app.infrastructure.messaging import RabbitMQListener
from app.config import settings
from app.infrastructure.services import ImageLoader, InferenceResultService, ModelWeightsLoader
from app.infrastructure.services.image_tiler_service import ImageTilerService
from app.logger import setup_logger
from app.core.enums import QueueTypes
from app.database import SessionLocal
from app.dependencies import get_detector_factory


async def main():
    setup_logger()

    storage = MinioStorage(endpoint=settings.MINIO_ENDPOINT, access_key=settings.MINIO_ACCESS_KEY, secret_key=settings.MINIO_SECRET_KEY)

    db = SessionLocal()

    task_repository = TaskRepository(db)
    model_repository = ModelRepository(db)
    detector_factory = get_detector_factory()
    image_loader = ImageLoader(storage=storage, bucket=settings.MINIO_SCHEMAS_BUCKET)
    weights_loader = ModelWeightsLoader(storage=storage, bucket=settings.MINIO_MODELS_BUCKET)
    result_repo = InferenceResultService(storage=storage, bucket=settings.MINIO_INFERENCE_RESULTS_BUCKET)
    image_tiler = ImageTilerService()

    use_case = DetectorInferenceUseCase(
        storage=storage,
        task_repo=task_repository,
        model_repo=model_repository,
        detector_factory=detector_factory,
        image_loader=image_loader,
        weights_loader=weights_loader,
        result_repo=result_repo,
        image_tiler=image_tiler
    )

    listener = RabbitMQListener(
        queue_name=QueueTypes.inference_queue,
        callback=use_case.execute
    )

    await listener.start()

if __name__ == "__main__":
    asyncio.run(main())