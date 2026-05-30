"""Unit tests for features.py — feature extraction compatibility layer.

Tests the delegation to ClipBackend and backward-compatible functions.
Uses mocking to avoid loading CLIP model.
"""
from unittest.mock import MagicMock, patch
import pytest


def _reset_backend():
    """Reset the global _backend singleton between tests."""
    import momento.features
    momento.features._backend = None


class TestGetBackend:
    """Backend singleton creation."""

    def setup_method(self):
        _reset_backend()

    def test_get_backend_creates_singleton(self):
        with patch("momento.features.ClipBackend") as mock_cls:
            mock_cls.return_value = MagicMock()
            from momento.features import _get_backend
            b1 = _get_backend()
            b2 = _get_backend()
            assert b1 is b2
            # Should only construct once
            mock_cls.assert_called_once()


class TestExtractFunctions:
    """Feature extraction functions delegate to ClipBackend."""

    def setup_method(self):
        _reset_backend()

    def test_extract_image_features(self):
        with patch("momento.features._get_backend") as mock_get:
            mock_backend = MagicMock()
            mock_backend.embed_image.return_value = MagicMock()
            mock_get.return_value = mock_backend

            from momento.features import extract_image_features
            extract_image_features("/path/to/image.jpg")
            mock_backend.embed_image.assert_called_once_with("/path/to/image.jpg")

    def test_extract_text_features(self):
        with patch("momento.features._get_backend") as mock_get:
            mock_backend = MagicMock()
            mock_backend.embed_text.return_value = MagicMock()
            mock_get.return_value = mock_backend

            from momento.features import extract_text_features
            extract_text_features("a cat")
            mock_backend.embed_text.assert_called_once_with("a cat")

    def test_extract_image_features_batch(self):
        with patch("momento.features._get_backend") as mock_get:
            mock_backend = MagicMock()
            mock_backend.embed_images_batch.return_value = ([], [])
            mock_get.return_value = mock_backend

            from momento.features import extract_image_features_batch
            extract_image_features_batch(["/a.jpg", "/b.jpg"], batch_size=16)
            mock_backend.embed_images_batch.assert_called_once_with(["/a.jpg", "/b.jpg"], 16)

    def test_clear_model_cache(self):
        """clear_model_cache sets _backend to None (doesn't need backend to exist)."""
        _reset_backend()
        from momento.features import clear_model_cache
        # Should not raise even if _backend is None
        clear_model_cache()