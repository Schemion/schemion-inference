from unittest.mock import MagicMock
from uuid import uuid4

from app.infrastructure.persistence.repositories.model_repository import ModelRepository


def test_get_by_id_queries_model_and_returns_first_match():
    db = MagicMock()
    query = db.query.return_value
    filtered = query.filter.return_value
    model = MagicMock()
    filtered.first.return_value = model

    repo = ModelRepository(db=db)
    model_id = uuid4()

    result = repo.get_by_id(model_id)

    assert result is model
    db.query.assert_called_once()
    query.filter.assert_called_once()
    filtered.first.assert_called_once_with()


def test_save_adds_commits_refreshes_and_returns_model():
    db = MagicMock()
    repo = ModelRepository(db=db)
    model = MagicMock()

    result = repo.save(model)

    assert result is model
    db.add.assert_called_once_with(model)
    db.commit.assert_called_once_with()
    db.refresh.assert_called_once_with(model)
