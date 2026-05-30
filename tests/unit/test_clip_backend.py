"""Unit tests for ClipBackend in embedding/clip_backend.py.

Tests class structure and inheritance.
Full validation tests require the clip package to be installed.
"""
from unittest.mock import MagicMock, patch


class TestClipBackendProperties:
    """ClipBackend class structure and properties."""

    def test_clip_backend_is_subclass(self):
        from momento.embedding.clip_backend import ClipBackend
        from momento.embedding.base import EmbeddingBackend
        assert issubclass(ClipBackend, EmbeddingBackend)

    def test_clip_backend_importable(self):
        from momento.embedding.clip_backend import ClipBackend
        assert ClipBackend is not None

    def test_clip_backend_has_required_methods(self):
        from momento.embedding.clip_backend import ClipBackend
        methods = ["embed_image", "embed_image_pil", "embed_text",
                   "embed_images_batch", "embed_pil_batch", "clear_cache"]
        for method in methods:
            assert hasattr(ClipBackend, method)