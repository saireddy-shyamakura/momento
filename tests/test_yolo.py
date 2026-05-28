"""Unit tests for yolo.py — YOLO object detection wrappers and data classes.

Covers:
- Detection dataclass and to_metadata
- is_available
- get_yolo_model (mocked)
- detect_objects (mocked) — bbox clipping, min_crop_size filter, empty results
- detect_objects_from_pil (mocked)
- Missing dependency handling
"""

from unittest.mock import patch, MagicMock
from PIL import Image
import pytest

from momento.yolo import (
    Detection,
    is_available,
    get_yolo_model,
    detect_objects,
    detect_objects_from_pil,
)


class TestDetectionDataclass:
    """Tests for the Detection dataclass."""

    def test_create_detection(self):
        img = Image.new("RGB", (32, 32))
        det = Detection(label="person", confidence=0.85, bbox=[10, 20, 50, 60], cropped_image=img)
        assert det.label == "person"
        assert det.confidence == 0.85
        assert det.bbox == [10, 20, 50, 60]
        assert det.cropped_image.size == (32, 32)

    def test_to_metadata(self):
        img = Image.new("RGB", (32, 32))
        det = Detection(label="dog", confidence=0.92, bbox=[5, 5, 40, 40], cropped_image=img)
        meta = det.to_metadata()
        assert meta["type"] == "yolo_object"
        assert meta["label"] == "dog"
        assert meta["confidence"] == 0.92
        assert meta["bbox_x1"] == 5.0
        assert meta["bbox_y1"] == 5.0
        assert meta["bbox_x2"] == 40.0
        assert meta["bbox_y2"] == 40.0

    def test_to_metadata_rounds_correctly(self):
        img = Image.new("RGB", (32, 32))
        det = Detection(label="car", confidence=0.87654, bbox=[10.123, 20.456, 50.789, 60.111], cropped_image=img)
        meta = det.to_metadata()
        assert meta["confidence"] == 0.8765  # round to 4 decimal places
        assert meta["bbox_x1"] == 10.1  # round to 1 decimal place
        assert meta["bbox_y1"] == 20.5
        assert meta["bbox_x2"] == 50.8
        assert meta["bbox_y2"] == 60.1


class TestIsAvailable:
    """Tests for is_available()."""

    def test_returns_true_when_yolo_importable(self):
        with patch("momento.yolo._HAS_YOLO", True):
            assert is_available() is True

    def test_returns_false_when_yolo_not_importable(self):
        with patch("momento.yolo._HAS_YOLO", False):
            assert is_available() is False


class TestGetYoloModel:
    """Tests for get_yolo_model."""

    def test_raises_when_ultralytics_missing(self):
        with patch("momento.yolo._HAS_YOLO", False):
            with pytest.raises(RuntimeError, match="ultralytics required"):
                get_yolo_model()

    def test_loads_model(self):
        mock_model = MagicMock()
        with patch("momento.yolo._HAS_YOLO", True), \
             patch("momento.yolo.YOLO") as mock_yolo:
            mock_yolo.return_value = mock_model
            model = get_yolo_model("yolov8n.pt")
            assert model is mock_model
            mock_yolo.assert_called_once_with("yolov8n.pt")

    def test_model_cached_after_first_load(self):
        with patch("momento.yolo._HAS_YOLO", True), \
             patch("momento.yolo.YOLO") as mock_yolo:
            mock_yolo.return_value = MagicMock()
            model1 = get_yolo_model("yolov8n.pt")
            model2 = get_yolo_model("yolov8n.pt")
            assert model1 is model2
            mock_yolo.assert_called_once()  # Only loaded once


class TestDetectObjects:
    """Tests for detect_objects (from file path)."""

    def test_raises_when_ultralytics_missing(self):
        with patch("momento.yolo._HAS_YOLO", False):
            with pytest.raises(RuntimeError, match="ultralytics required"):
                detect_objects("dummy.jpg")

    def test_raises_when_image_not_found(self):
        with patch("momento.yolo._HAS_YOLO", True):
            with pytest.raises(FileNotFoundError, match="not found"):
                detect_objects("/nonexistent/image.jpg")

    def test_returns_empty_when_no_detections(self, tmp_path):
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
        with patch("momento.yolo._HAS_YOLO", True), \
             patch("momento.yolo.get_yolo_model") as mock_get_model:
            model = MagicMock()
            result_mock = MagicMock()
            result_mock.boxes = []  # No detections
            model.return_value = [result_mock]
            mock_get_model.return_value = model
            detections = detect_objects(str(img_file))
            assert detections == []

    def test_parses_detections_correctly(self, tmp_path):
        img_file = tmp_path / "test.jpg"
        # Create a real image file
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(str(img_file))
        with patch("momento.yolo._HAS_YOLO", True), \
             patch("momento.yolo.get_yolo_model") as mock_get_model:
            model = MagicMock()
            box_mock = MagicMock()
            xyxy_mock = MagicMock()
            xyxy_mock.tolist.return_value = [10.0, 20.0, 50.0, 60.0]
            box_mock.xyxy = [xyxy_mock]
            box_mock.conf = [0.85]
            box_mock.cls = [0]
            result_mock = MagicMock()
            result_mock.boxes = [box_mock]
            result_mock.names = {0: "person"}
            model.return_value = [result_mock]
            mock_get_model.return_value = model
            detections = detect_objects(str(img_file))
            assert len(detections) == 1
            det = detections[0]
            assert det.label == "person"
            assert det.confidence == 0.85
            assert det.bbox == [10, 20, 50, 60]
            assert det.cropped_image.size == (40, 40)

    def test_bbox_clipped_to_image_boundaries(self, tmp_path):
        """Bbox coordinates outside the image should be clamped."""
        img_file = tmp_path / "test.jpg"
        img = Image.new("RGB", (50, 50), color="blue")
        img.save(str(img_file))
        with patch("momento.yolo._HAS_YOLO", True), \
             patch("momento.yolo.get_yolo_model") as mock_get_model:
            model = MagicMock()
            box_mock = MagicMock()
            xyxy_mock = MagicMock()
            xyxy_mock.tolist.return_value = [-10.0, -5.0, 100.0, 200.0]  # Outside image
            box_mock.xyxy = [xyxy_mock]
            box_mock.conf = [0.9]
            box_mock.cls = [0]
            result_mock = MagicMock()
            result_mock.boxes = [box_mock]
            result_mock.names = {0: "object"}
            model.return_value = [result_mock]
            mock_get_model.return_value = model
            detections = detect_objects(str(img_file))
            assert len(detections) == 1
            det = detections[0]
            # Bbox should be clamped to [0, 0, 50, 50]
            assert det.bbox == [0, 0, 50, 50]

    def test_min_crop_size_filter(self, tmp_path):
        """Detections with crops smaller than min_crop_size should be skipped."""
        img_file = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(str(img_file))
        with patch("momento.yolo._HAS_YOLO", True), \
             patch("momento.yolo.get_yolo_model") as mock_get_model:
            model = MagicMock()
            # Small bbox (20x20) — below default min_crop_size=32
            box_small = MagicMock()
            xyxy_small = MagicMock()
            xyxy_small.tolist.return_value = [10.0, 10.0, 30.0, 30.0]
            box_small.xyxy = [xyxy_small]
            box_small.conf = [0.9]
            box_small.cls = [0]
            # Large bbox (50x50) — above min_crop_size
            box_large = MagicMock()
            xyxy_large = MagicMock()
            xyxy_large.tolist.return_value = [10.0, 10.0, 60.0, 60.0]
            box_large.xyxy = [xyxy_large]
            box_large.conf = [0.8]
            box_large.cls = [1]
            result_mock = MagicMock()
            result_mock.boxes = [box_small, box_large]
            result_mock.names = {0: "small_obj", 1: "large_obj"}
            model.return_value = [result_mock]
            mock_get_model.return_value = model
            detections = detect_objects(str(img_file), min_crop_size=32)
            assert len(detections) == 1
            assert detections[0].label == "large_obj"


class TestDetectObjectsFromPil:
    """Tests for detect_objects_from_pil (from in-memory PIL image)."""

    def test_raises_when_ultralytics_missing(self):
        with patch("momento.yolo._HAS_YOLO", False):
            with pytest.raises(RuntimeError, match="ultralytics required"):
                detect_objects_from_pil(Image.new("RGB", (1, 1)))

    def test_parses_detections_from_pil(self):
        img = Image.new("RGB", (100, 100), color="red")
        with patch("momento.yolo._HAS_YOLO", True), \
             patch("momento.yolo.get_yolo_model") as mock_get_model:
            model = MagicMock()
            box_mock = MagicMock()
            xyxy_mock = MagicMock()
            xyxy_mock.tolist.return_value = [5.0, 5.0, 45.0, 45.0]
            box_mock.xyxy = [xyxy_mock]
            box_mock.conf = [0.95]
            box_mock.cls = [0]
            result_mock = MagicMock()
            result_mock.boxes = [box_mock]
            result_mock.names = {0: "cat"}
            model.return_value = [result_mock]
            mock_get_model.return_value = model
            detections = detect_objects_from_pil(img)
            assert len(detections) == 1
            assert detections[0].label == "cat"
            assert detections[0].confidence == 0.95

    def test_returns_empty_when_no_results(self):
        img = Image.new("RGB", (100, 100), color="red")
        with patch("momento.yolo._HAS_YOLO", True), \
             patch("momento.yolo.get_yolo_model") as mock_get_model:
            model = MagicMock()
            model.return_value = []  # No results
            mock_get_model.return_value = model
            detections = detect_objects_from_pil(img)
            assert detections == []