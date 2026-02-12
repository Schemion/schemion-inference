from app.infrastructure.celery_app import celery_app
from app.core.use_cases import DetectorInferenceUseCase
from app.infrastructure.cloud_storage import MinioStorage
from app.infrastructure.persistence.repositories import ModelRepository, TaskRepository
from app.infrastructure.services import ImageLoader, InferenceResultService, ModelWeightsLoader
from app.infrastructure.services.image_tiler_service import ImageTilerService
from app.dependencies import get_detector_factory
from app.config import settings
from app.database import SessionLocal
from app.logger import setup_logger


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def process_inference_task(message: dict):
    setup_logger()

    storage = MinioStorage(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY
    )
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

    use_case.execute(message)
