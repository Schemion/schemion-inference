from unittest.mock import MagicMock
from uuid import uuid4

from app.infrastructure.persistence.repositories.task_repository import TaskRepository


def test_get_by_id_queries_task_and_returns_first_match():
    db = MagicMock()
    query = db.query.return_value
    filtered = query.filter.return_value
    task = MagicMock()
    filtered.first.return_value = task

    repo = TaskRepository(db=db)
    task_id = uuid4()

    result = repo.get_by_id(task_id)

    assert result is task
    db.query.assert_called_once()
    query.filter.assert_called_once()
    filtered.first.assert_called_once_with()


def test_update_returns_none_if_task_not_found():
    db = MagicMock()
    query = db.query.return_value
    filtered = query.filter.return_value
    filtered.first.return_value = None

    repo = TaskRepository(db=db)
    task = MagicMock()
    task.id = uuid4()

    result = repo.update(task)

    assert result is None
    db.commit.assert_not_called()
    db.refresh.assert_not_called()


def test_update_commits_and_refreshes_when_task_exists():
    db = MagicMock()
    query = db.query.return_value
    filtered = query.filter.return_value
    task_model = MagicMock()
    filtered.first.return_value = task_model

    repo = TaskRepository(db=db)
    task = MagicMock()
    task.id = uuid4()

    result = repo.update(task)

    assert result is task_model
    db.commit.assert_called_once_with()
    db.refresh.assert_called_once_with(task_model)
