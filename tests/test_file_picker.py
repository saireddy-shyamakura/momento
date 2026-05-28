"""Unit tests for file_picker.py — folder and file selection UI."""

import os
from unittest.mock import patch, MagicMock

import pytest

from momento.file_picker import FilePicker


class TestPreviewIndexableFiles:
    """Tests for preview_indexable_files method."""

    def test_preview_counts_images_correctly(self, tmp_path):
        # Create test files
        (tmp_path / "img1.jpg").write_bytes(b"dummy")
        (tmp_path / "img2.png").write_bytes(b"dummy")
        (tmp_path / "img3.webp").write_bytes(b"dummy")
        (tmp_path / "text.txt").write_bytes(b"dummy")
        (tmp_path / "movie.mp4").write_bytes(b"dummy")
        picker = FilePicker()
        preview = picker.preview_indexable_files(str(tmp_path))
        assert preview["image_count"] == 3
        assert preview["video_count"] == 1
        assert preview["total_count"] == 4

    def test_preview_empty_folder(self, tmp_path):
        picker = FilePicker()
        preview = picker.preview_indexable_files(str(tmp_path))
        assert preview["total_count"] == 0

    def test_preview_respects_supported_extensions(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"dummy")
        (tmp_path / "b.gif").write_bytes(b"dummy")  # not supported
        (tmp_path / "c.mov").write_bytes(b"dummy")
        picker = FilePicker()
        preview = picker.preview_indexable_files(str(tmp_path))
        assert preview["image_count"] == 1  # only .jpg
        assert preview["video_count"] == 1  # .mov

    def test_preview_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.jpg").write_bytes(b"dummy")
        (sub / "sub.png").write_bytes(b"dummy")
        picker = FilePicker()
        preview = picker.preview_indexable_files(str(tmp_path))
        assert preview["total_count"] == 2

    def test_preview_ignores_symlinks(self, tmp_path):
        target = tmp_path / "real_target.jpg"
        target.write_bytes(b"dummy")
        link = tmp_path / "link.jpg"
        try:
            os.symlink(str(target), str(link))
        except OSError:
            pytest.skip("Cannot create symlink on this system")
        picker = FilePicker()
        preview = picker.preview_indexable_files(str(tmp_path))
        # followlinks=False by default in os.walk
        assert preview["total_count"] == 1

    def test_preview_nonexistent_folder_returns_zeros(self):
        picker = FilePicker()
        preview = picker.preview_indexable_files("/nonexistent/path")
        assert preview["total_count"] == 0


class TestConfirmFolderDiskSpace:
    """Tests for disk space estimation."""

    def test_estimate_space_needed_basic(self, tmp_path):
        (tmp_path / "img.jpg").write_bytes(b"dummy")
        picker = FilePicker()
        preview = picker.preview_indexable_files(str(tmp_path))
        space = picker._estimate_space_needed(preview)
        assert space["estimated_vectors"] > 0
        assert space["needed_gb"] > 0

    def test_estimate_free_gb_exists(self, tmp_path):
        (tmp_path / "img.jpg").write_bytes(b"dummy")
        picker = FilePicker()
        preview = picker.preview_indexable_files(str(tmp_path))
        space = picker._estimate_space_needed(preview)
        assert "free_gb" in space
        # On most systems this will be > 0
        assert space["free_gb"] > 0


class TestSelectImageFromFolder:
    """Tests for select_image_from_folder method."""

    def test_select_returns_empty_for_empty_folder(self, tmp_path):
        picker = FilePicker()
        result = picker.select_image_from_folder(str(tmp_path))
        assert result == ""

    def test_select_lists_images(self, tmp_path):
        (tmp_path / "test.jpg").write_bytes(b"dummy")
        picker = FilePicker()
        with patch("builtins.input", return_value="1"):
            result = picker.select_image_from_folder(str(tmp_path))
        assert "test.jpg" in result