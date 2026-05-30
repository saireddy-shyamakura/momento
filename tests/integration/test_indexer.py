"""Unit tests for indexer.py — Indexer orchestrator, IndexingStats, IndexingCheckpoint.

Covers:
- IndexingStats dataclass
- IndexingCheckpoint save/load/clear
- Indexer initialization
- index_all_features — feature ordering, checkpoint/resume, shutdown handling
- _index_images / _index_videos / _index_objects / _index_ocr with mocked ingest
- _check_memory
- Feature isolation (one feature failure doesn't stop others)
"""

import os
import json
import time
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import asdict

from momento.indexer import (
    Indexer,
    IndexingStats,
    IndexingCheckpoint,
    FeatureName,
    _check_memory,
    CHECKPOINT_FILE,
)


class TestIndexingStats:
    """Tests for IndexingStats dataclass."""

    def test_default_values(self):
        stats = IndexingStats()
        assert stats.images_added == 0
        assert stats.videos_added == 0
        assert stats.objects_added == 0
        assert stats.ocr_added == 0
        assert stats.total_vectors == 0
        assert stats.duration_secs == 0.0
        assert stats.errors == []
        assert stats.has_errors() is False

    def test_add_error(self):
        stats = IndexingStats()
        stats.add_error("something failed")
        assert stats.has_errors() is True
        assert len(stats.errors) == 1
        assert stats.errors[0] == "something failed"


class TestIndexingCheckpoint:
    """Tests for IndexingCheckpoint persistence."""

    def test_save_and_load(self, tmp_path):
        cp = IndexingCheckpoint(
            folder=str(tmp_path / "photos"),
            features_completed=["IMAGES", "VIDEOS"],
            current_feature="OBJECTS",
            started_at=time.time(),
        )
        # Monkey-patch the checkpoint path to use tmp_path
        with patch("momento.indexer.CHECKPOINT_FILE", str(tmp_path / "checkpoint.json")):
            cp.save()
            loaded = IndexingCheckpoint.load(str(tmp_path / "photos"))
            assert loaded is not None
            assert loaded.folder == cp.folder
            assert loaded.features_completed == ["IMAGES", "VIDEOS"]
            assert loaded.current_feature == "OBJECTS"

    def test_load_returns_none_when_no_file(self, tmp_path):
        with patch("momento.indexer.CHECKPOINT_FILE", str(tmp_path / "checkpoint.json")):
            result = IndexingCheckpoint.load(str(tmp_path / "photos"))
            assert result is None

    def test_load_returns_none_on_folder_mismatch(self, tmp_path):
        cp = IndexingCheckpoint(folder="some_folder")
        with patch("momento.indexer.CHECKPOINT_FILE", str(tmp_path / "checkpoint.json")):
            cp.save()
            result = IndexingCheckpoint.load("different_folder")
            assert result is None

    def test_load_handles_corrupted_file(self, tmp_path):
        cp_file = tmp_path / "checkpoint.json"
        cp_file.write_text("not valid json")
        with patch("momento.indexer.CHECKPOINT_FILE", str(cp_file)):
            result = IndexingCheckpoint.load("any_folder")
            assert result is None

    def test_clear_deletes_file(self, tmp_path):
        cp = IndexingCheckpoint(folder=str(tmp_path))
        with patch("momento.indexer.CHECKPOINT_FILE", str(tmp_path / "checkpoint.json")):
            cp.save()
            assert os.path.exists(str(tmp_path / "checkpoint.json"))
            IndexingCheckpoint.clear()
            assert not os.path.exists(str(tmp_path / "checkpoint.json"))

    def test_clear_no_file_does_not_raise(self, tmp_path):
        with patch("momento.indexer.CHECKPOINT_FILE", str(tmp_path / "nonexistent.json")):
            IndexingCheckpoint.clear()  # Should not raise


class TestCheckMemory:
    """Tests for _check_memory."""

    def test_returns_true_when_memory_available(self):
        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.available = 8 * 1024 * 1024 * 1024  # 8 GB
            assert _check_memory() is True

    def test_returns_false_when_memory_low(self):
        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.available = 1 * 1024 * 1024 * 1024  # 1 GB
            assert _check_memory() is False

    def test_handles_psutil_not_available(self):
        with patch("momento.indexer.psutil", None) or \
             patch("momento.indexer.psutil.virtual_memory", side_effect=Exception):
            # If psutil is unavailable, _check_memory should handle gracefully
            assert _check_memory() is True


class TestIndexer:
    """Tests for the Indexer class."""

    def test_init(self):
        mock_index = MagicMock()
        indexer = Indexer(mock_index)
        assert indexer.index is mock_index
        assert indexer.stats is not None

    def test_index_all_features_calls_all_features(self):
        mock_index = MagicMock()
        mock_index.get_vector_count.return_value = 42
        
        with patch("momento.indexer._check_memory", return_value=True), \
             patch("momento.indexer.IndexingCheckpoint.load", return_value=None), \
             patch("momento.indexer.IndexingCheckpoint.clear"), \
             patch.object(Indexer, "_index_images") as mock_img, \
             patch.object(Indexer, "_index_videos") as mock_vid, \
             patch.object(Indexer, "_index_objects") as mock_obj, \
             patch.object(Indexer, "_index_ocr") as mock_ocr:
            
            indexer = Indexer(mock_index)
            stats = indexer.index_all_features("/test/folder")
            
            mock_img.assert_called_once_with("/test/folder")
            mock_vid.assert_called_once_with("/test/folder")
            mock_obj.assert_called_once_with("/test/folder")
            mock_ocr.assert_called_once_with("/test/folder")
            
            assert stats.total_vectors == 42
            assert stats.duration_secs > 0

    def test_index_all_features_resumes_from_checkpoint(self):
        mock_index = MagicMock()
        
        checkpoint = IndexingCheckpoint(
            folder="/test/folder",
            features_completed=["IMAGES", "VIDEOS"],
        )
        
        with patch("momento.indexer._check_memory", return_value=True), \
             patch("momento.indexer.IndexingCheckpoint.load", return_value=checkpoint), \
             patch("momento.indexer.IndexingCheckpoint.clear"), \
             patch.object(Indexer, "_index_images") as mock_img, \
             patch.object(Indexer, "_index_videos") as mock_vid, \
             patch.object(Indexer, "_index_objects") as mock_obj, \
             patch.object(Indexer, "_index_ocr") as mock_ocr:
            
            indexer = Indexer(mock_index)
            indexer.index_all_features("/test/folder")
            
            # IMAGES and VIDEOS should be skipped
            mock_img.assert_not_called()
            mock_vid.assert_not_called()
            # OBJECTS and OCR should still run
            mock_obj.assert_called_once_with("/test/folder")
            mock_ocr.assert_called_once_with("/test/folder")

    def test_index_all_features_isolates_feature_failures(self):
        """One feature failure should not stop other features."""
        mock_index = MagicMock()
        
        with patch("momento.indexer._check_memory", return_value=True), \
             patch("momento.indexer.IndexingCheckpoint.load", return_value=None), \
             patch("momento.indexer.IndexingCheckpoint.clear"), \
             patch.object(Indexer, "_index_images", side_effect=Exception("Image crash")), \
             patch.object(Indexer, "_index_videos") as mock_vid, \
             patch.object(Indexer, "_index_objects") as mock_obj, \
             patch.object(Indexer, "_index_ocr") as mock_ocr:
            
            indexer = Indexer(mock_index)
            stats = indexer.index_all_features("/test/folder")
            
            # Other features should still run after image failure
            mock_vid.assert_called_once()
            mock_obj.assert_called_once()
            mock_ocr.assert_called_once()
            # The error should be captured
            assert stats.has_errors() is True

    def test_index_images_disabled_when_multi_embed(self):
        """When ENABLE_MULTI_EMBED is True, _index_images should call add_images_multi."""
        mock_index = MagicMock()
        
        with patch("momento.indexer.ENABLE_MULTI_EMBED", True), \
             patch("momento.indexer.add_images_multi", return_value=5) as mock_multi, \
             patch("momento.indexer._check_memory", return_value=True), \
             patch("momento.indexer.IndexingCheckpoint.load", return_value=None), \
             patch("momento.indexer.IndexingCheckpoint.clear"), \
             patch.object(Indexer, "_index_videos"), \
             patch.object(Indexer, "_index_objects"), \
             patch.object(Indexer, "_index_ocr"):
            
            indexer = Indexer(mock_index)
            indexer.index_all_features("/test/folder")
            
            mock_multi.assert_called_once_with("/test/folder", mock_index)

    def test_index_images_disabled_when_not_multi_embed(self):
        """When ENABLE_MULTI_EMBED is False, _index_images should call add_images."""
        mock_index = MagicMock()
        
        with patch("momento.indexer.ENABLE_MULTI_EMBED", False), \
             patch("momento.indexer.add_images", return_value=3) as mock_img, \
             patch("momento.indexer._check_memory", return_value=True), \
             patch("momento.indexer.IndexingCheckpoint.load", return_value=None), \
             patch("momento.indexer.IndexingCheckpoint.clear"), \
             patch.object(Indexer, "_index_videos"), \
             patch.object(Indexer, "_index_objects"), \
             patch.object(Indexer, "_index_ocr"):
            
            indexer = Indexer(mock_index)
            indexer.index_all_features("/test/folder")
            
            mock_img.assert_called_once_with("/test/folder", mock_index)

    def test_index_videos_skipped_when_disabled(self):
        """When ENABLE_VIDEO_INDEXING is False, _index_videos should not call add_videos."""
        mock_index = MagicMock()
        indexer = Indexer(mock_index)
        with patch("momento.indexer.ENABLE_VIDEO_INDEXING", False), \
             patch("momento.indexer.add_videos") as mock_add_vid:
            indexer._index_videos("/test/folder")
            mock_add_vid.assert_not_called()

    def test_index_objects_skipped_when_disabled(self):
        """When ENABLE_YOLO is False, _index_objects should not call add_objects."""
        mock_index = MagicMock()
        indexer = Indexer(mock_index)
        with patch("momento.indexer.ENABLE_YOLO", False), \
             patch("momento.indexer.add_objects") as mock_add_obj:
            indexer._index_objects("/test/folder")
            mock_add_obj.assert_not_called()

    def test_index_ocr_skipped_when_disabled(self):
        """When ENABLE_OCR is False, _index_ocr should not call add_ocr."""
        mock_index = MagicMock()
        indexer = Indexer(mock_index)
        with patch("momento.indexer.ENABLE_OCR", False), \
             patch("momento.indexer.add_ocr") as mock_add_ocr:
            indexer._index_ocr("/test/folder")
            mock_add_ocr.assert_not_called()

    def test_index_videos_propagates_errors(self):
        """Errors in _index_videos should be captured in stats."""
        mock_index = MagicMock()
        indexer = Indexer(mock_index)
        with patch("momento.indexer.ENABLE_VIDEO_INDEXING", True), \
             patch("momento.indexer.add_videos", side_effect=RuntimeError("video error")):
            indexer._index_videos("/test/folder")
        assert indexer.stats.has_errors() is True

    def test_index_objects_propagates_errors(self):
        """Errors in _index_objects should be captured in stats."""
        mock_index = MagicMock()
        indexer = Indexer(mock_index)
        with patch("momento.indexer.ENABLE_YOLO", True), \
             patch("momento.indexer.add_objects", side_effect=RuntimeError("yolo error")):
            indexer._index_objects("/test/folder")
        assert indexer.stats.has_errors() is True

    def test_index_ocr_propagates_errors(self):
        """Errors in _index_ocr should be captured in stats."""
        mock_index = MagicMock()
        indexer = Indexer(mock_index)
        with patch("momento.indexer.ENABLE_OCR", True), \
             patch("momento.indexer.add_ocr", side_effect=RuntimeError("ocr error")):
            indexer._index_ocr("/test/folder")
        assert indexer.stats.has_errors() is True


class TestFeatureOrder:
    """Test feature ordering and shutdown handling."""

    def test_features_in_correct_order(self):
        mock_index = MagicMock()
        mock_index.get_vector_count.return_value = 0
        
        call_order = []
        
        def track_images(folder):
            call_order.append("images")

        def track_videos(folder):
            call_order.append("videos")

        def track_objects(folder):
            call_order.append("objects")

        def track_ocr(folder):
            call_order.append("ocr")

        with patch("momento.indexer._check_memory", return_value=True), \
             patch("momento.indexer.IndexingCheckpoint.load", return_value=None), \
             patch("momento.indexer.IndexingCheckpoint.clear"), \
             patch.object(Indexer, "_index_images", side_effect=track_images), \
             patch.object(Indexer, "_index_videos", side_effect=track_videos), \
             patch.object(Indexer, "_index_objects", side_effect=track_objects), \
             patch.object(Indexer, "_index_ocr", side_effect=track_ocr):
            
            indexer = Indexer(mock_index)
            indexer.index_all_features("/test/folder")
            
            assert call_order == ["images", "videos", "objects", "ocr"]

    def test_shutdown_stops_after_current_feature(self):
        mock_index = MagicMock()
        
        def shutdown_after_images(folder):
            from momento.shutdown import _shutdown_requested
            # Simulate shutdown requested after images
            import momento.shutdown
            momento.shutdown._shutdown_requested = True

        with patch("momento.indexer._check_memory", return_value=True), \
             patch("momento.indexer.IndexingCheckpoint.load", return_value=None), \
             patch("momento.indexer.IndexingCheckpoint.clear"), \
             patch.object(Indexer, "_index_images", side_effect=shutdown_after_images), \
             patch.object(Indexer, "_index_videos") as mock_vid, \
             patch.object(Indexer, "_index_objects") as mock_obj, \
             patch.object(Indexer, "_index_ocr") as mock_ocr, \
             patch("momento.indexer.is_shutdown_requested", side_effect=[False, False, True, True, True]):
            
            indexer = Indexer(mock_index)
            indexer.index_all_features("/test/folder")
            
            # Videos should not be called if shutdown was requested after images
            # Note: the checkpoint system checks before each feature
            mock_vid.assert_not_called()
            mock_obj.assert_not_called()
            mock_ocr.assert_not_called()
