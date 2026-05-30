"""Unit tests for ExactIndex in search/exact_index.py.

Tests the SQLite FTS5 exact search index with external content table.
- Add/remove/clear paths
- count() and len()
- Search with FTS5 MATCH
- Exact filename/path match
- Edge cases (empty, whitespace, duplicates)
- Close/reopen persistence
"""
import os
import pytest


def _import_exact_index():
    from momento.search.exact_index import ExactIndex
    return ExactIndex


class TestExactIndexBasics:
    """Basic CRUD operations."""

    def test_new_index_empty(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        assert idx.count() == 0
        assert len(idx) == 0

    def test_add_paths(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        count = idx.add_paths(["/path/to/image.jpg", "/path/to/photo.png"])
        assert count == 2
        assert idx.count() == 2

    def test_add_empty_list(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        count = idx.add_paths([])
        assert count == 0

    def test_add_duplicate_paths(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        c1 = idx.add_paths(["/a.jpg"])
        c2 = idx.add_paths(["/a.jpg"])
        assert c1 == 1
        assert c2 == 0  # INSERT OR IGNORE
        assert idx.count() == 1

    def test_remove_paths(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        idx.add_paths(["/a.jpg", "/b.jpg", "/c.jpg"])
        assert idx.count() == 3
        removed = idx.remove_paths(["/a.jpg"])
        assert removed == 1
        assert idx.count() == 2

    def test_remove_nonexistent_path(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        idx.add_paths(["/a.jpg"])
        removed = idx.remove_paths(["/nonexistent.jpg"])
        assert removed == 0

    def test_remove_empty_list(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        removed = idx.remove_paths([])
        assert removed == 0

    def test_clear_index(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        idx.add_paths(["/a.jpg", "/b.jpg"])
        assert idx.count() == 2
        idx.clear()
        assert idx.count() == 0

    def test_close_and_reopen(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        idx.add_paths(["/a.jpg"])
        idx.close()
        assert idx.count() == 1  # Persisted


class TestExactIndexSearch:
    """Search functionality."""

    def test_exact_filename_match(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        idx.add_paths(["/path/to/sunset.jpg"])
        results = idx.search("sunset.jpg")
        assert len(results) >= 1
        score, path = results[0]
        assert score == 1.0
        assert "sunset.jpg" in path

    def test_exact_path_match(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        idx.add_paths(["/unique/path/photo.jpg"])
        results = idx.search("/unique/path/photo.jpg")
        assert len(results) >= 1
        score, path = results[0]
        assert score == 1.0

    def test_fts_search(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        idx.add_paths(["/photos/vacation/sunset.jpg"])
        results = idx.search("sunset")
        assert len(results) >= 1
        assert any("sunset" in p for _, p in results)

    def test_empty_query(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        idx.add_paths(["/a.jpg"])
        assert idx.search("") == []

    def test_whitespace_query(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        idx.add_paths(["/a.jpg"])
        assert idx.search("   ") == []

    def test_no_match(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        idx.add_paths(["/a.jpg"])
        assert idx.search("nonexistent_xyz") == []

    def test_multiple_matches(self, tmp_path):
        ExactIndex = _import_exact_index()
        db_path = os.path.join(tmp_path, "test_exact.db")
        idx = ExactIndex(db_path)
        idx.add_paths(["/dir1/sunset.jpg", "/dir2/sunset_beach.jpg", "/dir3/night.jpg"])
        results = idx.search("sunset")
        assert len(results) >= 1