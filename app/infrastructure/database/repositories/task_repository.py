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

    def get(self, task_id: UUID) -> Optional[TaskEntity]:
        model = self.db.query(TaskModel).filter(task_id == TaskModel.id).first()
        return OrmEntityMapper.to_entity(model, TaskEntity)

    def create(self, task: TaskEntity) -> TaskEntity:
        model = OrmEntityMapper.to_model(task, TaskModel)  # ENTITY → ORM MODEL
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return OrmEntityMapper.to_entity(model, TaskEntity)

    def update(self, task: TaskEntity) -> TaskEntity:
        model = self.db.query(TaskModel).filter(task.id == TaskModel.id).first()
        if not model:
            return None

        for key, value in task.__dict__.items():
            setattr(model, key, value)

        self.db.commit()
        self.db.refresh(model)
        return OrmEntityMapper.to_entity(model, TaskEntity)

    def list(self) -> List[TaskEntity]:
        models = self.db.query(TaskModel).all()
        return [OrmEntityMapper.to_entity(m, TaskEntity) for m in models]
