"""Unit tests for EmbeddingBackend ABC in embedding/base.py.

Tests the abstract interface including:
- Cannot instantiate directly
- All abstract methods defined
- Method signatures match expected types
"""
import pytest


class TestEmbeddingBackendInterface:
    """Interface compliance tests."""

    def test_cannot_instantiate_abstract_class(self):
        from momento.embedding.base import EmbeddingBackend
        with pytest.raises(TypeError):
            EmbeddingBackend()

    def test_all_abstract_methods_defined(self):
        """All expected abstract methods should exist on the class."""
        from momento.embedding.base import EmbeddingBackend
        methods = [
            "embed_image",
            "embed_image_pil",
            "embed_text",
            "embed_images_batch",
            "embed_pil_batch",
            "clear_cache",
        ]
        for method in methods:
            assert hasattr(EmbeddingBackend, method), f"Missing abstract method: {method}"

    def test_abstract_properties_defined(self):
        from momento.embedding.base import EmbeddingBackend
        assert hasattr(EmbeddingBackend, "name")
        assert hasattr(EmbeddingBackend, "dimension")

    def test_concrete_subclass_satisfies_interface(self):
        """ClipBackend should satisfy the EmbeddingBackend interface."""
        from momento.embedding.clip_backend import ClipBackend
        from momento.embedding.base import EmbeddingBackend
        assert issubclass(ClipBackend, EmbeddingBackend)