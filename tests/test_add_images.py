"""Unit tests for add_images() skip logic, extension filtering, and empty folder.

Validates: Requirements 4.1, 4.2, 4.3, 4.4
"""

import os
import numpy as np
import pytest
from unittest.mock import patch

from add_images import add_images
from index import Index


# ── Helpers ──────────────────────────────────────────────────────────

def _make_index(tmp_path) -> Index:
    """Return a fresh Index backed by a temporary ChromaDB directory."""
    return Index(db_path=str(tmp_path / "chroma"))


def _dummy_vectors(paths):
    """Return (paths, list_of_dummy_vectors) matching the signature of extract_image_features_batch."""
    vectors = [np.random.randn(512).astype(np.float32) for _ in paths]
    return paths, vectors


def _create_image_file(directory, name: str) -> str:
    """Create a minimal valid file with the given name and return its absolute path."""
    path = directory / name
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)  # minimal JPEG-like header
    return str(path.resolve())


# ── Test: all images already indexed → returns 0 ─────────────────────

class TestAllAlreadyIndexed:
    """Requirement 4.3 — images already in the index are skipped."""

    def test_all_indexed_returns_zero(self, tmp_path):
        """When every image in the folder is already indexed, add_images returns 0."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        # Create two image files
        path_a = _create_image_file(img_dir, "a.jpg")
        path_b = _create_image_file(img_dir, "b.png")

        idx = _make_index(tmp_path)

        # Pre-populate the index with dummy vectors for both images
        dummy_vecs = [np.random.randn(512).astype(np.float32) for _ in range(2)]
        idx.add_vectors([path_a, path_b], dummy_vecs)

        with patch("add_images.extract_image_features_batch") as mock_extract:
            result = add_images(str(img_dir), idx)

        assert result == 0
        mock_extract.assert_not_called()

    def test_all_indexed_does_not_call_feature_extraction(self, tmp_path):
        """Feature extraction must not be called when all images are already indexed."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        path = _create_image_file(img_dir, "photo.jpg")
        idx = _make_index(tmp_path)
        idx.add_vectors([path], [np.random.randn(512).astype(np.float32)])

        with patch("add_images.extract_image_features_batch") as mock_extract:
            add_images(str(img_dir), idx)

        mock_extract.assert_not_called()


# ── Test: unsupported extensions only → returns 0, no extraction ─────

class TestUnsupportedExtensionsOnly:
    """Requirement 4.2 — files with unsupported extensions are filtered out."""

    def test_unsupported_extensions_returns_zero(self, tmp_path):
        """Folder containing only .gif and .txt files → add_images returns 0."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        # Create non-image files
        (img_dir / "animation.gif").write_bytes(b"GIF89a" + b"\x00" * 10)
        (img_dir / "notes.txt").write_text("not an image")

        idx = _make_index(tmp_path)

        with patch("add_images.extract_image_features_batch") as mock_extract:
            result = add_images(str(img_dir), idx)

        assert result == 0
        mock_extract.assert_not_called()

    def test_unsupported_extensions_no_feature_extraction(self, tmp_path):
        """Feature extraction must never be called for unsupported file types."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        (img_dir / "doc.txt").write_text("text file")
        (img_dir / "anim.gif").write_bytes(b"GIF89a" + b"\x00" * 10)

        idx = _make_index(tmp_path)

        with patch("add_images.extract_image_features_batch") as mock_extract:
            add_images(str(img_dir), idx)

        mock_extract.assert_not_called()


# ── Test: mix of new and already-indexed images ───────────────────────

class TestMixedNewAndIndexed:
    """Requirement 4.1, 4.3 — only new images are processed and counted."""

    def test_mixed_folder_returns_count_of_new_only(self, tmp_path):
        """Only the 2 new images should be processed; the 1 already-indexed is skipped."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        already_indexed = _create_image_file(img_dir, "old.jpg")
        new_a = _create_image_file(img_dir, "new_a.jpg")
        new_b = _create_image_file(img_dir, "new_b.png")

        idx = _make_index(tmp_path)
        # Pre-index only the "old" image
        idx.add_vectors([already_indexed], [np.random.randn(512).astype(np.float32)])

        with patch("add_images.extract_image_features_batch") as mock_extract:
            mock_extract.side_effect = lambda paths, **kwargs: _dummy_vectors(paths)
            result = add_images(str(img_dir), idx)

        assert result == 2

    def test_mixed_folder_only_processes_new_paths(self, tmp_path):
        """extract_image_features_batch must only receive the paths not yet indexed."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        already_indexed = _create_image_file(img_dir, "old.jpg")
        new_img = _create_image_file(img_dir, "new.jpg")

        idx = _make_index(tmp_path)
        idx.add_vectors([already_indexed], [np.random.randn(512).astype(np.float32)])

        captured_paths = []

        def capture_extract(paths, **kwargs):
            captured_paths.extend(paths)
            return _dummy_vectors(paths)

        with patch("add_images.extract_image_features_batch", side_effect=capture_extract):
            add_images(str(img_dir), idx)

        assert already_indexed not in captured_paths
        assert new_img in captured_paths

    def test_mixed_folder_with_unsupported_files(self, tmp_path):
        """Unsupported files are ignored; only new supported images are counted."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        already_indexed = _create_image_file(img_dir, "old.jpg")
        new_img = _create_image_file(img_dir, "new.jpg")
        (img_dir / "ignore.txt").write_text("not an image")
        (img_dir / "ignore.gif").write_bytes(b"GIF89a" + b"\x00" * 10)

        idx = _make_index(tmp_path)
        idx.add_vectors([already_indexed], [np.random.randn(512).astype(np.float32)])

        with patch("add_images.extract_image_features_batch") as mock_extract:
            mock_extract.side_effect = lambda paths, **kwargs: _dummy_vectors(paths)
            result = add_images(str(img_dir), idx)

        assert result == 1


# ── Test: empty folder → returns 0 without exception ─────────────────

class TestEmptyFolder:
    """Requirement 4.4 — empty folder is handled gracefully."""

    def test_empty_folder_returns_zero(self, tmp_path):
        """An empty folder should return 0 without raising any exception."""
        img_dir = tmp_path / "empty"
        img_dir.mkdir()

        idx = _make_index(tmp_path)

        result = add_images(str(img_dir), idx)

        assert result == 0

    def test_empty_folder_does_not_raise(self, tmp_path):
        """add_images() on an empty folder must not raise any exception."""
        img_dir = tmp_path / "empty"
        img_dir.mkdir()

        idx = _make_index(tmp_path)

        try:
            add_images(str(img_dir), idx)
        except Exception as exc:
            pytest.fail(f"add_images() raised an unexpected exception on empty folder: {exc}")

    def test_empty_folder_no_feature_extraction(self, tmp_path):
        """Feature extraction must not be called for an empty folder."""
        img_dir = tmp_path / "empty"
        img_dir.mkdir()

        idx = _make_index(tmp_path)

        with patch("add_images.extract_image_features_batch") as mock_extract:
            add_images(str(img_dir), idx)

        mock_extract.assert_not_called()
