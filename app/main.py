import asyncio
from app.core.use_cases import InferenceUseCase
from app.infrastructure.cloud_storage import MinioStorage
from app.infrastructure.database.repositories.model_repository import ModelRepository
from app.infrastructure.database.repositories.task_repository import TaskRepository
from app.infrastructure.messaging import RabbitMQListener
from app.config import settings
from app.dependencies import get_db
from app.logger import setup_logger


async def main():
    setup_logger()

    storage = MinioStorage(endpoint=settings.MINIO_ENDPOINT, access_key=settings.MINIO_ACCESS_KEY, secret_key=settings.MINIO_SECRET_KEY)
    task_repository = TaskRepository(get_db())
    model_repository = ModelRepository(get_db())

    use_case = InferenceUseCase(storage, task_repository, model_repository)

    listener = RabbitMQListener(
        queue_name="inference",
        callback=use_case.execute
    )

    await listener.start()

if __name__ == "__main__":
    asyncio.run(main())