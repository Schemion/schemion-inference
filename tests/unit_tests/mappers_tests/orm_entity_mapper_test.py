from dataclasses import dataclass
from app.infrastructure import mappers


@dataclass
class MockEntity:
    id: int
    name: str
    value: int


class MockModel:
    id = None
    name = None
    value = None

    def __init__(self, id, name, value, ignored="pls ignore me"):
        self.id = id
        self.name = name
        self.value = value
        self.ignored_field = ignored

def test_to_entity_maps_only_matching_fields():
    model = MockModel(id=10, name="hello", value=5, ignored="skip me")

    entity = mappers.OrmEntityMapper.to_entity(model, MockEntity)

    assert isinstance(entity, MockEntity)
    assert entity.id == 10
    assert entity.name == "hello"
    assert entity.value == 5


def test_to_entity_ignores_extra_fields():
    model = MockModel(id=1, name="x", value=2)
    model.extra = "should_not_map"

    entity = mappers.OrmEntityMapper.to_entity(model, MockEntity)

    assert not hasattr(entity, "extra")


def test_to_entity_none_returns_none():
    assert mappers.OrmEntityMapper.to_entity(None, MockEntity) is None


def test_to_model_maps_only_existing_attributes():
    entity = MockEntity(id=123, name="test", value=777)

    model = mappers.OrmEntityMapper.to_model(entity, MockModel)

    assert isinstance(model, MockModel)
    assert model.id == 123
    assert model.name == "test"
    assert model.value == 777


def test_to_model_does_not_map_missing_fields():
    entity = MockEntity(id=1, name="a", value=2)

    model = mappers.OrmEntityMapper.to_model(entity, MockModel)

    assert not hasattr(model, "non_existing_field")