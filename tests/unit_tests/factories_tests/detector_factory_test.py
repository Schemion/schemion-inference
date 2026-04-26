from unittest.mock import MagicMock, patch

import pytest

from app.infrastructure.factories.detector_factory import DetectorFactory


def test_create_returns_yolo_detector_when_architecture_contains_yolo():
    factory = DetectorFactory()

    with patch.object(factory, "_create_yolo", return_value=MagicMock()) as create_yolo:
        detector = factory.create("YOLO26")

    create_yolo.assert_called_once_with()
    assert detector is create_yolo.return_value


def test_create_resolves_aliases_and_builds_fasterrcnn_with_classes():
    factory = DetectorFactory()

    with patch.object(factory, "_create_fasterrcnn", return_value=MagicMock()) as create_fasterrcnn:
        detector = factory.create(
            architecture="frcnn",
            architecture_profile="resnet50",
            classes=["cat", "dog"],
        )

    create_fasterrcnn.assert_called_once_with("resnet50_fpn", ["cat", "dog"])
    assert detector is create_fasterrcnn.return_value


def test_create_raises_for_unsupported_architecture():
    factory = DetectorFactory()

    with pytest.raises(ValueError, match="Unsupported architecture"):
        factory.create("unknown-model", architecture_profile="unknown-profile")
