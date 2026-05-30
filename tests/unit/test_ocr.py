"""Unit tests for ocr.py — OCR text extraction.

Tests is_available, extract_text error handling,
and extract_text_from_pil delegation.
"""
import pytest
from unittest.mock import patch


class TestOcrAvailable:
    """OCR availability check."""

    def test_is_available_true_when_importable(self):
        with patch("momento.ocr._HAS_OCR", True):
            from momento.ocr import is_available
            assert is_available() is True

    def test_is_available_false_when_not_importable(self):
        with patch("momento.ocr._HAS_OCR", False):
            from momento.ocr import is_available
            assert is_available() is False


class TestExtractText:
    """extract_text error handling."""

    def test_raises_if_not_available(self):
        with patch("momento.ocr._HAS_OCR", False):
            from momento.ocr import extract_text
            with pytest.raises(RuntimeError, match="easyocr required"):
                extract_text("/path/to/image.jpg")

    def test_raises_if_file_not_found(self):
        with patch("momento.ocr._HAS_OCR", True):
            from momento.ocr import extract_text
            with pytest.raises(FileNotFoundError, match="not found"):
                extract_text("/nonexistent/image.jpg")


class TestExtractTextFromPil:
    """extract_text_from_pil error handling."""

    def test_raises_if_not_available(self):
        with patch("momento.ocr._HAS_OCR", False):
            from momento.ocr import extract_text_from_pil
            from PIL import Image
            img = Image.new("RGB", (64, 64))
            with pytest.raises(RuntimeError, match="easyocr required"):
                extract_text_from_pil(img)