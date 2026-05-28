"""Unit tests for index_utils.py — index management utilities.

Covers:
- get_or_create_index
- verify_index — stale entry detection and removal
- get_index_stats — per-type breakdown
- reset_index
- get_index_health
"""

from unittest.mock import patch, MagicMock
import pytest

from momento.index_utils import (
    get_or_create_index,
    verify_index,
    get_index_stats,
    reset_index,
    get_index_health,
)
from momento.config import COMPOSITE_SEP


class TestGetOrCreateIndex:
    """Tests for get_or_create_index."""

    def test_returns_index_instance(self):
        with patch("momento.index_utils.Index") as mock_index_class:
            mock_instance = MagicMock()
            mock_instance.get_vector_count.return_value = 42
            mock_index_class.return_value = mock_instance
            result = get_or_create_index()
            assert result is mock_instance

    def test_propagates_index_error(self):
        with patch("momento.index_utils.Index", side_effect=RuntimeError("DB corrupted")):
            with pytest.raises(RuntimeError, match="DB corrupted"):
                get_or_create_index()


class TestVerifyIndex:
    """Tests for verify_index."""

    def test_clean_index_returns_zero_stale(self):
        mock_index = MagicMock()
        mock_index.get_all_paths.return_value = ["/img/a.jpg", "/img/b.jpg"]
        with patch("os.path.exists", return_value=True):
            result = verify_index(mock_index)
            assert result["stale_count"] == 0
            assert result["is_clean"] is True
            mock_index.delete_paths.assert_not_called()

    def test_removes_stale_entries(self):
        mock_index = MagicMock()
        mock_index.get_all_paths.return_value = ["/img/exists.jpg", "/img/missing.jpg"]
        # First exists, second does not
        exists_side_effect = lambda p: p == "/img/exists.jpg"
        with patch("os.path.exists", side_effect=exists_side_effect):
            result = verify_index(mock_index)
            assert result["stale_count"] == 1
            assert result["is_clean"] is False
            mock_index.delete_paths.assert_called_once_with(["/img/missing.jpg"])

    def test_handles_composite_ids(self):
        mock_index = MagicMock()
        composite_id = f"/img/photo.jpg{COMPOSITE_SEP}orig"
        mock_index.get_all_paths.return_value = [composite_id]
        with patch("os.path.exists", return_value=True):
            result = verify_index(mock_index)
            assert result["stale_count"] == 0
            assert result["is_clean"] is True

    def test_composite_id_with_missing_base(self):
        mock_index = MagicMock()
        composite_id = f"/img/missing.jpg{COMPOSITE_SEP}yolo_person_10_20_50_60"
        mock_index.get_all_paths.return_value = [composite_id]
        with patch("os.path.exists", return_value=False):
            result = verify_index(mock_index)
            assert result["stale_count"] == 1
            assert result["is_clean"] is False

    def test_propagates_errors(self):
        mock_index = MagicMock()
        mock_index.get_all_paths.side_effect = RuntimeError("DB error")
        with pytest.raises(RuntimeError, match="DB error"):
            verify_index(mock_index)

    def test_reports_total_entries(self):
        mock_index = MagicMock()
        mock_index.get_all_paths.return_value = ["/img/a.jpg", "/img/b.jpg"]
        with patch("os.path.exists", return_value=True):
            result = verify_index(mock_index)
            assert result["total_entries"] == 2


class TestGetIndexStats:
    """Tests for get_index_stats."""

    def test_returns_correct_structure(self):
        mock_index = MagicMock()
        mock_index.get_vector_count.return_value = 100
        mock_index.get_all_paths.return_value = [
            f"/img/photo.jpg{COMPOSITE_SEP}orig",
            f"/img/photo.jpg{COMPOSITE_SEP}flip",
            f"/video/clip.mp4{COMPOSITE_SEP}frame_0000",
            f"/img/photo.jpg{COMPOSITE_SEP}yolo_person_10_20_50_60",
            f"/img/photo.jpg{COMPOSITE_SEP}ocr",
        ]
        stats = get_index_stats(mock_index)
        assert stats["total_vectors"] == 100
        assert stats["total_entries"] == 5
        assert stats["estimated_images"] == 2  # orig + flip
        assert stats["estimated_videos"] == 1  # frame_
        assert stats["estimated_objects"] == 1  # yolo_
        assert stats["estimated_ocr"] == 1  # ocr
        assert "db_path" in stats

    def test_propagates_errors(self):
        mock_index = MagicMock()
        mock_index.get_vector_count.side_effect = RuntimeError("DB error")
        with pytest.raises(RuntimeError, match="DB error"):
            get_index_stats(mock_index)


class TestResetIndex:
    """Tests for reset_index."""

    def test_reset_calls_delete_all(self):
        mock_index = MagicMock()
        result = reset_index(mock_index)
        assert result is True
        mock_index.delete_all.assert_called_once()

    def test_propagates_errors(self):
        mock_index = MagicMock()
        mock_index.delete_all.side_effect = RuntimeError("DB error")
        with pytest.raises(RuntimeError, match="DB error"):
            reset_index(mock_index)


class TestGetIndexHealth:
    """Tests for get_index_health."""

    def test_healthy_index(self):
        mock_index = MagicMock()
        mock_index.get_vector_count.return_value = 50
        mock_index.get_all_paths.return_value = ["/img/a.jpg"]
        with patch("os.path.exists", return_value=True):
            health = get_index_health(mock_index)
            assert health["status"] == "healthy"
            assert health["vectors"] == 50
            assert health["stale_count"] == 0

    def test_warning_on_stale_entries(self):
        mock_index = MagicMock()
        mock_index.get_vector_count.return_value = 50
        mock_index.get_all_paths.return_value = ["/img/missing.jpg"]
        with patch("os.path.exists", return_value=False):
            health = get_index_health(mock_index)
            assert health["status"] == "warning"
            assert health["stale_count"] == 1

    def test_error_on_db_failure(self):
        mock_index = MagicMock()
        mock_index.get_vector_count.side_effect = RuntimeError("DB error")
        health = get_index_health(mock_index)
        assert health["status"] == "error"
        assert "error" in health