from typing import Optional, List
from uuid import UUID

from sqlalchemy.orm import Session

from app.infrastructure.database.models import Task as TaskModel
from app.core.entities.task import Task as TaskEntity
from app.core.interfaces import ITaskRepository
from app.infrastructure.mappers import OrmEntityMapper


class TaskRepository(ITaskRepository):
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, task_id: UUID) -> Optional[TaskEntity]:
        task = self.db.query(TaskModel).filter(task_id == TaskModel.id).first()
        return OrmEntityMapper.to_entity(task, TaskEntity)


    def update(self, task: TaskEntity) -> Optional[TaskEntity]:
        task = self.db.query(TaskModel).filter(task.id == TaskModel.id).first()
        if not task:
            return None

        mapped = OrmEntityMapper.to_model(task, TaskModel)
        for key, value in mapped.__dict__.items():
            if hasattr(task, key):
                setattr(task, key, value)

        self.db.commit()
        self.db.refresh(task)
        return OrmEntityMapper.to_entity(task, TaskEntity)
