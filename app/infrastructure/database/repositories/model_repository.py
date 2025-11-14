from typing import Optional, List
from sqlalchemy.orm import Session

from app.core.entities.model import Model as ModelEntity
from app.core.interfaces.model_interface import IModelRepository
from app.infrastructure.database.models.model import Model
from app.infrastructure.mappers import OrmEntityMapper


class ModelRepository(IModelRepository):
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, model_id) -> Optional[ModelEntity]:
        orm_obj = self.db.query(Model).filter(model_id == Model.id).first()
        return OrmEntityMapper.to_entity(orm_obj, ModelEntity)

    def list(self) -> List[ModelEntity]:
        orm_objs = self.db.query(Model).all()
        return [OrmEntityMapper.to_entity(obj, ModelEntity) for obj in orm_objs]

    def save(self, entity: ModelEntity) -> ModelEntity:
        orm_obj = OrmEntityMapper.to_model(entity, Model)
        self.db.add(orm_obj)
        self.db.commit()
        self.db.refresh(orm_obj)
        return OrmEntityMapper.to_entity(orm_obj, ModelEntity)
