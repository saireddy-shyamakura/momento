"""Unit tests for ocr.py — EasyOCR text extraction wrappers.

Covers:
- is_available
- get_reader (mocked)
- extract_text (mocked) — text concatenation, confidence filtering, empty result
- extract_text_from_pil (mocked)
- Missing dependency handling
- File not found edge case
"""

from unittest.mock import patch, MagicMock
from PIL import Image
import pytest

from momento.ocr import (
    is_available,
    get_reader,
    extract_text,
    extract_text_from_pil,
)


class TestIsAvailable:
    """Tests for is_available()."""

    def test_returns_true_when_easyocr_importable(self):
        with patch("momento.ocr._HAS_OCR", True):
            assert is_available() is True

    def test_returns_false_when_easyocr_not_importable(self):
        with patch("momento.ocr._HAS_OCR", False):
            assert is_available() is False


class TestGetReader:
    """Tests for get_reader."""

    def test_raises_when_ocr_missing(self):
        with patch("momento.ocr._HAS_OCR", False):
            with pytest.raises(RuntimeError, match="easyocr required"):
                get_reader()

    def test_creates_reader_with_default_languages(self):
        with patch("momento.ocr._HAS_OCR", True), \
             patch("momento.ocr.easyocr.Reader") as mock_reader_class, \
             patch("momento.ocr.device_manager") as mock_device:
            mock_device.device = "cpu"
            mock_reader_class.return_value = MagicMock()
            reader = get_reader()
            mock_reader_class.assert_called_once_with(["en"], gpu=False)
            assert reader is not None

    def test_creates_reader_with_custom_languages(self):
        with patch("momento.ocr._HAS_OCR", True), \
             patch("momento.ocr.easyocr.Reader") as mock_reader_class, \
             patch("momento.ocr.device_manager") as mock_device:
            mock_device.device = "cpu"
            mock_reader_class.return_value = MagicMock()
            reader = get_reader(languages=["en", "fr"])
            mock_reader_class.assert_called_once_with(["en", "fr"], gpu=False)

    def test_reader_is_cached(self):
        with patch("momento.ocr._HAS_OCR", True), \
             patch("momento.ocr.easyocr.Reader") as mock_reader_class, \
             patch("momento.ocr.device_manager") as mock_device:
            mock_device.device = "cpu"
            mock_reader_class.return_value = MagicMock()
            r1 = get_reader()
            r2 = get_reader()
            assert r1 is r2
            mock_reader_class.assert_called_once()

    def test_enables_gpu_when_cuda_available(self):
        with patch("momento.ocr._HAS_OCR", True), \
             patch("momento.ocr.easyocr.Reader") as mock_reader_class, \
             patch("momento.ocr.device_manager") as mock_device:
            mock_device.device = "cuda"
            mock_reader_class.return_value = MagicMock()
            get_reader()
            mock_reader_class.assert_called_once_with(["en"], gpu=True)


class TestExtractText:
    """Tests for extract_text from file path."""

    def test_raises_when_ocr_missing(self):
        with patch("momento.ocr._HAS_OCR", False):
            with pytest.raises(RuntimeError, match="easyocr required"):
                extract_text("dummy.jpg")

    def test_raises_when_image_not_found(self):
        with patch("momento.ocr._HAS_OCR", True):
            with pytest.raises(FileNotFoundError, match="not found"):
                extract_text("/nonexistent/image.jpg")

    def test_returns_empty_string_when_no_text(self, tmp_path):
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
        with patch("momento.ocr._HAS_OCR", True), \
             patch("momento.ocr.get_reader") as mock_get_reader:
            reader = MagicMock()
            reader.readtext.return_value = []  # No text found
            mock_get_reader.return_value = reader
            result = extract_text(str(img_file))
            assert result == ""

    def test_concatenates_text_regions(self, tmp_path):
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
        with patch("momento.ocr._HAS_OCR", True), \
             patch("momento.ocr.get_reader") as mock_get_reader:
            reader = MagicMock()
            # EasyOCR returns list of (bbox, text, confidence)
            reader.readtext.return_value = [
                ([[0, 0], [10, 0], [10, 5], [0, 5]], "Hello", 0.95),
                ([[20, 0], [30, 0], [30, 5], [20, 5]], "World", 0.88),
            ]
            mock_get_reader.return_value = reader
            result = extract_text(str(img_file))
            assert result == "Hello World"

    def test_filters_by_confidence(self, tmp_path):
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
        with patch("momento.ocr._HAS_OCR", True), \
             patch("momento.ocr.get_reader") as mock_get_reader:
            reader = MagicMock()
            reader.readtext.return_value = [
                ([[0, 0], [10, 0], [10, 5], [0, 5]], "High", 0.95),
                ([[20, 0], [30, 0], [30, 5], [20, 5]], "Low", 0.20),  # Below default threshold
            ]
            mock_get_reader.return_value = reader
            result = extract_text(str(img_file), min_confidence=0.3)
            assert result == "High"

    def test_filters_empty_text(self, tmp_path):
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
        with patch("momento.ocr._HAS_OCR", True), \
             patch("momento.ocr.get_reader") as mock_get_reader:
            reader = MagicMock()
            reader.readtext.return_value = [
                ([[0, 0], [10, 0], [10, 5], [0, 5]], "   ", 0.9),  # Whitespace only
                ([[20, 0], [30, 0], [30, 5], [20, 5]], "Valid", 0.9),
            ]
            mock_get_reader.return_value = reader
            result = extract_text(str(img_file))
            assert result == "Valid"


class TestExtractTextFromPil:
    """Tests for extract_text_from_pil from in-memory PIL image."""

    def test_raises_when_ocr_missing(self):
        with patch("momento.ocr._HAS_OCR", False):
            with pytest.raises(RuntimeError, match="easyocr required"):
                extract_text_from_pil(Image.new("RGB", (1, 1)))

    def test_returns_empty_string_when_no_text(self):
        img = Image.new("RGB", (100, 100), color="white")
        with patch("momento.ocr._HAS_OCR", True), \
             patch("momento.ocr.get_reader") as mock_get_reader:
            reader = MagicMock()
            reader.readtext.return_value = []
            mock_get_reader.return_value = reader
            result = extract_text_from_pil(img)
            assert result == ""

    def test_concatenates_text_regions(self):
        img = Image.new("RGB", (100, 100), color="white")
        with patch("momento.ocr._HAS_OCR", True), \
             patch("momento.ocr.get_reader") as mock_get_reader:
            reader = MagicMock()
            reader.readtext.return_value = [
                ([[0, 0], [10, 0], [10, 5], [0, 5]], "Hello", 0.95),
                ([[20, 0], [30, 0], [30, 5], [20, 5]], "World", 0.88),
            ]
            mock_get_reader.return_value = reader
            result = extract_text_from_pil(img)
            assert result == "Hello World"

    def test_filters_by_confidence(self):
        img = Image.new("RGB", (100, 100), color="white")
        with patch("momento.ocr._HAS_OCR", True), \
             patch("momento.ocr.get_reader") as mock_get_reader:
            reader = MagicMock()
            reader.readtext.return_value = [
                ([[0, 0], [10, 0], [10, 5], [0, 5]], "Keep", 0.95),
                ([[20, 0], [30, 0], [30, 5], [20, 5]], "Discard", 0.15),
            ]
            mock_get_reader.return_value = reader
            result = extract_text_from_pil(img, min_confidence=0.3)
            assert result == "Keep"

    def test_converts_image_to_rgb(self):
        """OCR should convert grayscale images to RGB."""
        img = Image.new("L", (100, 100), color=128)  # Grayscale
        with patch("momento.ocr._HAS_OCR", True), \
             patch("momento.ocr.get_reader") as mock_get_reader:
            reader = MagicMock()
            reader.readtext.return_value = [
                ([[0, 0], [10, 0], [10, 5], [0, 5]], "Text", 0.9),
            ]
            mock_get_reader.return_value = reader
            result = extract_text_from_pil(img)
            assert result == "Text"