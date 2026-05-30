"""Unit tests for MetadataStore in storage/metadata_store.py.

Tests the SQLite-backed metadata store including:
- Add/update/get single file
- Batch add multiple entries
- Filter by extension, size, date, OCR text
- get_all_paths, count, remove, clear
- JSON serialization for objects/custom fields
- Edge cases (empty batch, missing path)
"""
import os
import json
import pytest


def _import_metadata_store():
    from momento.storage.metadata_store import MetadataStore
    return MetadataStore


class TestMetadataStoreBasics:
    """Basic CRUD operations."""

    def test_add_and_get_file(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file("/path/to/image.jpg", file_size=1024)
        meta = store.get_file("/path/to/image.jpg")

        assert meta is not None
        assert meta["filename"] == "image.jpg"
        assert meta["ext"] == ".jpg"
        assert meta["file_size"] == 1024
        assert meta["objects"] == []
        assert meta["custom"] == {}

    def test_get_nonexistent_file(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        meta = store.get_file("/nonexistent.jpg")
        assert meta is None

    def test_add_file_with_ocr_and_objects(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file(
            "/path/to/photo.jpg",
            file_size=2048,
            ocr_text="Hello World",
            objects=["person", "dog"],
            custom={"rating": 5},
        )
        meta = store.get_file("/path/to/photo.jpg")

        assert meta["ocr_text"] == "Hello World"
        assert meta["objects"] == ["person", "dog"]
        assert meta["custom"] == {"rating": 5}

    def test_update_existing_file(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file("/path/to/image.jpg", file_size=100)
        store.add_file("/path/to/image.jpg", file_size=200)  # Updated
        meta = store.get_file("/path/to/image.jpg")
        assert meta["file_size"] == 200

    def test_remove_path(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file("/a.jpg")
        assert store.remove_path("/a.jpg") is True
        assert store.get_file("/a.jpg") is None

    def test_remove_nonexistent_path(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        assert store.remove_path("/nonexistent.jpg") is False

    def test_clear_store(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file("/a.jpg")
        store.add_file("/b.jpg")
        assert store.count() == 2
        store.clear()
        assert store.count() == 0

    def test_count(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        assert store.count() == 0
        store.add_file("/a.jpg")
        assert store.count() == 1
        store.add_file("/b.jpg")
        assert store.count() == 2


class TestMetadataStoreBatch:
    """Batch operations."""

    def test_batch_add(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        entries = [
            {"path": "/a.jpg", "file_size": 100},
            {"path": "/b.jpg", "file_size": 200, "ocr_text": "hello"},
            {"path": "/c.jpg", "file_size": 300, "objects": ["cat"]},
        ]
        count = store.batch_add(entries)
        assert count == 3
        assert store.count() == 3

    def test_batch_add_empty(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        count = store.batch_add([])
        assert count == 0

    def test_batch_add_missing_path_skipped(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        entries = [
            {"path": "/a.jpg"},
            {"file_size": 200},  # Missing path → skipped
            {"path": "/b.jpg"},
        ]
        count = store.batch_add(entries)
        assert count == 2
        assert store.count() == 2


class TestMetadataStoreFilter:
    """Filter operations."""

    def test_filter_by_extension(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file("/a.jpg")
        store.add_file("/b.png")
        store.add_file("/c.jpg")
        results = store.filter(ext=".jpg")
        assert len(results) == 2
        for r in results:
            assert r["ext"] == ".jpg"

    def test_filter_by_min_size(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file("/small.jpg", file_size=100)
        store.add_file("/medium.jpg", file_size=500)
        store.add_file("/large.jpg", file_size=1000)
        results = store.filter(min_size=500)
        assert len(results) == 2

    def test_filter_by_max_size(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file("/small.jpg", file_size=100)
        store.add_file("/large.jpg", file_size=1000)
        results = store.filter(max_size=500)
        assert len(results) == 1

    def test_filter_by_size_range(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file("/a.jpg", file_size=100)
        store.add_file("/b.jpg", file_size=500)
        store.add_file("/c.jpg", file_size=1000)
        results = store.filter(min_size=200, max_size=800)
        assert len(results) == 1
        assert results[0]["filename"] == "b.jpg"

    def test_filter_by_ocr_contains(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file("/a.jpg", ocr_text="Hello World")
        store.add_file("/b.jpg", ocr_text="Goodbye")
        results = store.filter(ocr_contains="Hello")
        assert len(results) == 1
        assert results[0]["filename"] == "a.jpg"

    def test_filter_multiple_criteria(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file("/a.jpg", file_size=100, ocr_text="cat")
        store.add_file("/b.jpg", file_size=500, ocr_text="dog")
        store.add_file("/c.jpg", file_size=100, ocr_text="cat")
        results = store.filter(ext=".jpg", min_size=50, max_size=200, ocr_contains="cat")
        assert len(results) == 2

    def test_filter_no_criteria_returns_all(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file("/a.jpg")
        store.add_file("/b.jpg")
        results = store.filter()
        assert len(results) == 2


class TestMetadataStoreGetAll:
    """get_all_paths and related operations."""

    def test_get_all_paths(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file("/b.jpg")
        store.add_file("/a.jpg")
        store.add_file("/c.jpg")
        paths = store.get_all_paths()
        assert paths == ["/a.jpg", "/b.jpg", "/c.jpg"]  # Sorted

    def test_get_all_paths_empty(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        assert store.get_all_paths() == []

    def test_objects_json_parsing(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file("/a.jpg", objects=["person", "car", "dog"])
        meta = store.get_file("/a.jpg")
        assert meta["objects"] == ["person", "car", "dog"]

    def test_custom_json_parsing(self, tmp_path):
        MetadataStore = _import_metadata_store()
        db_path = os.path.join(tmp_path, "test_meta.db")
        store = MetadataStore(db_path)

        store.add_file("/a.jpg", custom={"width": 1920, "height": 1080})
        meta = store.get_file("/a.jpg")
        assert meta["custom"]["width"] == 1920
        assert meta["custom"]["height"] == 1080