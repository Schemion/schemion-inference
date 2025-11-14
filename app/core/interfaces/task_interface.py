from abc import ABC, abstractmethod
from typing import Optional, List
from uuid import UUID
from app.core.entities import Task


class ITaskRepository(ABC):

    @abstractmethod
    def get(self, task_id: UUID) -> Optional[Task]: ...

    @abstractmethod
    def create(self, task: Task) -> Task: ...

    @abstractmethod
    def update(self, task: Task) -> Task: ...

    @abstractmethod
    def list(self) -> List[Task]: ...
