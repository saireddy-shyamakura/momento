"""Unit tests for indexer.py — Indexing orchestrator.

Tests Indexer initialization, IndexingStats, and feature indexing methods.
Uses mocking to avoid real file processing.
"""
from unittest.mock import MagicMock, patch
import pytest


class TestIndexingStats:
    """IndexingStats dataclass."""

    def test_default_values(self):
        from momento.indexer import IndexingStats
        stats = IndexingStats()
        assert stats.images_added == 0
        assert stats.videos_added == 0
        assert stats.objects_added == 0
        assert stats.ocr_added == 0
        assert stats.total_vectors == 0
        assert stats.duration_secs == 0.0
        assert stats.errors == []

    def test_add_error(self):
        from momento.indexer import IndexingStats
        stats = IndexingStats()
        stats.add_error("something failed")
        assert stats.has_errors() is True
        assert len(stats.errors) == 1

    def test_no_errors_by_default(self):
        from momento.indexer import IndexingStats
        stats = IndexingStats()
        assert stats.has_errors() is False


class TestIndexerInit:
    """Indexer initialization."""

    def test_initialization(self):
        from momento.indexer import Indexer
        mock_index = MagicMock()
        indexer = Indexer(mock_index)
        assert indexer.index is mock_index
        assert indexer.stats.images_added == 0


class TestIndexerRun:
    """Indexer run with mocked dependencies."""

    def test_index_all_features_images_only(self):
        from momento.indexer import Indexer
        mock_index = MagicMock()

        with patch("momento.indexer._check_memory"), \
             patch("momento.indexer.IndexingCheckpoint.load", return_value=None), \
             patch("momento.indexer.ENABLE_VIDEO_INDEXING", False), \
             patch("momento.indexer.ENABLE_YOLO", False), \
             patch("momento.indexer.ENABLE_OCR", False), \
             patch("momento.indexer.add_images_multi") as mock_add_images, \
             patch("momento.indexer.is_shutdown_requested", return_value=False), \
             patch("momento.indexer.IndexingCheckpoint.clear"):

            mock_add_images.return_value = 5
            mock_index.get_vector_count.return_value = 30

            indexer = Indexer(mock_index)
            stats = indexer.index_all_features("/test/folder")
            assert stats.images_added == 5

    def test_index_all_features_with_shutdown(self):
        from momento.indexer import Indexer
        mock_index = MagicMock()

        with patch("momento.indexer._check_memory"), \
             patch("momento.indexer.IndexingCheckpoint.load", return_value=None), \
             patch("momento.indexer.is_shutdown_requested", return_value=True), \
             patch("momento.indexer.IndexingCheckpoint.clear"):

            indexer = Indexer(mock_index)
            stats = indexer.index_all_features("/test/folder")
            assert stats.duration_secs >= 0


class TestIndexerCheckMemory:
    """Memory check helper."""

    def test_check_memory(self):
        from momento.indexer import _check_memory
        with patch("momento.indexer.psutil") as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 4 * 1024 * 1024 * 1024
            mock_psutil.virtual_memory.return_value = mock_mem
            assert _check_memory() is True

    def test_check_memory_low(self):
        from momento.indexer import _check_memory
        with patch("momento.indexer.psutil") as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 1 * 1024 * 1024 * 1024
            mock_psutil.virtual_memory.return_value = mock_mem
            assert _check_memory() is False