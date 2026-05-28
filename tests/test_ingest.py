import numpy as np
import pytest
from unittest.mock import patch

from momento.ingest import add_objects
from momento.index import Index
from momento.config import COMPOSITE_SEP


def _make_index(tmp_path) -> Index:
    return Index(db_path=str(tmp_path / "chroma_db"))


def _create_image_file(directory, name: str) -> str:
    path = directory / name
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    return str(path.resolve())


def _dummy_object_embeddings():
    vec = np.random.randn(512).astype(np.float32)
    return [
        ({"label": "person", "bbox": [10, 20, 50, 60]}, vec),
        ({"label": "cat", "bbox": [5, 5, 40, 40]}, vec),
    ]


class TestAddObjectsIdempotency:
    """Regression tests for YOLO ingestion idempotency and stable object IDs."""

    @patch("momento.ingest.extract_object_embeddings")
    def test_add_objects_uses_stable_ids_across_runs(self, mock_extract, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        image_path = _create_image_file(img_dir, "sample.jpg")
        mock_extract.return_value = _dummy_object_embeddings()

        idx = _make_index(tmp_path)
        first_count = add_objects(str(img_dir), idx)
        assert first_count == 2
        assert idx.get_vector_count() == 2

        second_count = add_objects(str(img_dir), idx)
        assert second_count == 2
        assert idx.get_vector_count() == 2

        expected_ids = {
            f"{image_path}{COMPOSITE_SEP}yolo_person_10_20_50_60",
            f"{image_path}{COMPOSITE_SEP}yolo_cat_5_5_40_40",
        }
        assert idx.get_existing_ids(list(expected_ids)) == expected_ids
