"""Unit tests for app_controller.py — main application orchestrator."""

from unittest.mock import patch, MagicMock

import pytest

from momento.app_controller import AppController
from momento.shutdown import is_shutdown_requested


class TestAppControllerInit:
    """Tests for AppController initialization."""

    def test_init_creates_state(self):
        controller = AppController()
        assert controller.state is not None
        assert controller.state.index is None
        assert controller.state.current_folder is None


class TestIndexVerification:
    """Tests for _verify_index method."""

    def test_verify_clean_index_returns_zero(self):
        controller = AppController()
        mock_index = MagicMock()
        mock_index.get_all_paths.return_value = []
        result = controller._verify_index(mock_index)
        assert result == 0

    def test_verify_removes_stale_entries(self, tmp_path):
        controller = AppController()
        mock_index = MagicMock()
        existing = str(tmp_path / "exists.jpg")
        nonexistent = str(tmp_path / "nonexistent.jpg")
        # Create one file so it exists
        existing = str(tmp_path / "exists.jpg")
        with open(existing, 'wb') as f:
            f.write(b"dummy")
        nonexistent = str(tmp_path / "nonexistent.jpg")
        # Note: nonexistent is not created

        from momento.config import COMPOSITE_SEP
        mock_index.get_all_paths.return_value = [existing, nonexistent]
        controller._verify_index(mock_index)
        mock_index.delete_paths.assert_called_once_with([nonexistent])

    def test_verify_no_remove_when_all_exist(self, tmp_path):
        controller = AppController()
        mock_index = MagicMock()
        p1 = str(tmp_path / "a.jpg")
        p2 = str(tmp_path / "b.jpg")
        with open(p1, 'wb') as f:
            f.write(b"a")
        with open(p2, 'wb') as f:
            f.write(b"b")
        mock_index.get_all_paths.return_value = [p1, p2]
        result = controller._verify_index(mock_index)
        assert result == 0
        mock_index.delete_paths.assert_not_called()


class TestIsShutdownRequested:
    """Tests for the global shutdown flag."""

    def test_default_is_false(self):
        assert is_shutdown_requested() is False

    def test_importable(self):
        from momento.shutdown import is_shutdown_requested
        assert callable(is_shutdown_requested)
