"""Unit tests for yolo.py — YOLO object detection.

Tests Detection dataclass, is_available, and error handling.
Uses mocking to avoid loading YOLO model.
"""
from unittest.mock import patch
import pytest


class TestDetection:
    """Detection dataclass."""

    def test_detection_creation(self):
        from momento.yolo import Detection
        from PIL import Image
        img = Image.new("RGB", (64, 64))
        det = Detection(label="dog", confidence=0.95, bbox=[10, 20, 100, 200], cropped_image=img)
        assert det.label == "dog"
        assert det.confidence == 0.95
        assert det.bbox == [10, 20, 100, 200]

    def test_to_metadata(self):
        from momento.yolo import Detection
        from PIL import Image
        img = Image.new("RGB", (64, 64))
        det = Detection(label="cat", confidence=0.85, bbox=[5, 5, 50, 60], cropped_image=img)
        meta = det.to_metadata()
        assert meta["type"] == "yolo_object"
        assert meta["label"] == "cat"
        assert meta["confidence"] == 0.85


class TestYoloAvailable:
    """YOLO availability check."""

    def test_is_available_true_when_importable(self):
        with patch("momento.yolo._HAS_YOLO", True):
            from momento.yolo import is_available
            assert is_available() is True

    def test_is_available_false_when_not_importable(self):
        with patch("momento.yolo._HAS_YOLO", False):
            from momento.yolo import is_available
            assert is_available() is False


class TestDetectObjects:
    """detect_objects error handling."""

    def test_raises_if_not_available(self):
        with patch("momento.yolo._HAS_YOLO", False):
            from momento.yolo import detect_objects
            with pytest.raises(RuntimeError, match="ultralytics required"):
                detect_objects("/path/to/image.jpg")

    def test_raises_if_file_not_found(self):
        with patch("momento.yolo._HAS_YOLO", True):
            from momento.yolo import detect_objects
            with pytest.raises(FileNotFoundError, match="not found"):
                detect_objects("/nonexistent/image.jpg")


class TestDetectObjectsFromPil:
    """detect_objects_from_pil error handling."""

    def test_raises_if_not_available(self):
        with patch("momento.yolo._HAS_YOLO", False):
            from momento.yolo import detect_objects_from_pil
            from PIL import Image
            img = Image.new("RGB", (64, 64))
            with pytest.raises(RuntimeError, match="ultralytics required"):
                detect_objects_from_pil(img)