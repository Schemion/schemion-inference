import json
import logging
import signal
import threading
import time

from bobber import BobberClient

from app.core.use_cases import DetectorInferenceUseCase
from app.core.enums import QueueTypes
from app.infrastructure.cloud_storage import MinioStorage
from app.infrastructure.persistence.repositories import ModelRepository, TaskRepository
from app.infrastructure.services import ImageLoader, InferenceResultService, ModelWeightsLoader
from app.infrastructure.services.image_tiler_service import ImageTilerService
from app.dependencies import get_detector_factory
from app.config import settings
from app.database import SessionLocal
from app.logger import setup_logger

logger = logging.getLogger(__name__)

def process_inference_task(message: dict):
    storage = MinioStorage(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY
    )
    db = SessionLocal()
    try:
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
    finally:
        db.close()


def _parse_message(payload: dict) -> dict | None:
    raw_value = payload.get("value")
    if raw_value is None:
        logger.error("Broker message missing 'value': %s", payload)
        return None
    if isinstance(raw_value, dict):
        return raw_value
    if not isinstance(raw_value, str):
        logger.error("Broker message value must be str or dict, got %s", type(raw_value))
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        logger.exception("Failed to decode broker message JSON: %s", raw_value)
        return None


def _on_broker_message(payload: dict) -> None:
    message = _parse_message(payload)
    if not message:
        return
    logger.info("Received inference task %s", message.get("task_id"))
    process_inference_task(message)


def main() -> None:
    setup_logger()

    client = BobberClient(host=settings.BOBBER_HOST, port=settings.BOBBER_PORT)
    if not client.healthcheck():
        raise ConnectionError("Bobber broker unavailable")

    stop_event = threading.Event()

    def _handle_signal(_sig, _frame):
        logger.info("Shutdown signal received")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    topic = QueueTypes.inference_queue.value
    client.subscribe(topic, _on_broker_message)
    logger.info("Listening to broker topic '%s'", topic)

    try:
        while not stop_event.is_set():
            time.sleep(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
