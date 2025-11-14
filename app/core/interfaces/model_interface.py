from abc import ABC, abstractmethod
from typing import Optional, List
from app.core.entities.model import Model


class IModelRepository(ABC):
    @abstractmethod
    def get_by_id(self, model_id) -> Optional[Model]:
        ...

    @abstractmethod
    def list(self) -> List[Model]:
        ...

    @abstractmethod
    def save(self, model: Model) -> Model:
        ...
