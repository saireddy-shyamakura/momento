"""Unit tests for recall_search() in retrieval/recall.py.

Tests the first-stage fast recall layer including:
- Empty/unbuilt index handling
- Recall multiplier logic (3x top_k)
- Threshold filtering
- Metadata where clause passthrough
- Edge cases (threshold 0.0, 1.0)
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


def _import_recall():
    from momento.retrieval.recall import recall_search
    return recall_search


def _make_mock_index(is_built=True, vector_count=100, search_results=None):
    """Create a mock Index with configurable behavior."""
    mock = MagicMock()
    mock.is_built.return_value = is_built
    mock.get_vector_count.return_value = vector_count
    if search_results is not None:
        mock.search_with_metadata.return_value = search_results
    return mock


_DUMMY_QUERY = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
_TOP_K = 10


class TestRecallUnbuiltIndex:
    """When index is not built, recall_search returns []."""

    def test_returns_empty_when_index_not_built(self):
        recall_search = _import_recall()
        mock_index = _make_mock_index(is_built=False)
        results = recall_search(_DUMMY_QUERY, mock_index, top_k=_TOP_K)
        assert results == []

    def test_does_not_call_search_when_not_built(self):
        recall_search = _import_recall()
        mock_index = _make_mock_index(is_built=False)
        recall_search(_DUMMY_QUERY, mock_index, top_k=_TOP_K)
        mock_index.search_with_metadata.assert_not_called()

    def test_no_exception_when_index_not_built(self):
        recall_search = _import_recall()
        mock_index = _make_mock_index(is_built=False)
        try:
            result = recall_search(_DUMMY_QUERY, mock_index, top_k=_TOP_K)
        except Exception as exc:
            pytest.fail(f"recall_search raised unexpected exception: {exc}")
        assert result == []


class TestRecallEmptyIndex:
    """When index has 0 vectors, recall_search returns []."""

    def test_returns_empty_when_no_vectors(self):
        recall_search = _import_recall()
        mock_index = _make_mock_index(vector_count=0)
        results = recall_search(_DUMMY_QUERY, mock_index, top_k=_TOP_K)
        assert results == []

    def test_does_not_call_search_when_zero_vectors(self):
        recall_search = _import_recall()
        mock_index = _make_mock_index(vector_count=0)
        recall_search(_DUMMY_QUERY, mock_index, top_k=_TOP_K)
        mock_index.search_with_metadata.assert_not_called()


class TestRecallMultiplier:
    """Recall stage requests top_k * RECALL_MULTIPLIER candidates."""

    def test_recall_multiplier_applied(self):
        recall_search = _import_recall()
        mock_index = _make_mock_index(
            search_results=[(0.9, "id1", {}), (0.8, "id2", {}), (0.7, "id3", {})]
        )
        # top_k=10 => recall_k = min(30, 100) = 30
        recall_search(_DUMMY_QUERY, mock_index, top_k=10)
        mock_index.search_with_metadata.assert_called_once()
        call_kwargs = mock_index.search_with_metadata.call_args[1]
        assert call_kwargs["top_k"] == 30

    def test_recall_k_capped_by_vector_count(self):
        recall_search = _import_recall()
        mock_index = _make_mock_index(vector_count=5, search_results=[])
        # top_k=10 => recall_k = min(30, 5) = 5
        recall_search(_DUMMY_QUERY, mock_index, top_k=10)
        mock_index.search_with_metadata.assert_called_once()
        call_kwargs = mock_index.search_with_metadata.call_args[1]
        assert call_kwargs["top_k"] == 5

    def test_top_k_single_candidate(self):
        recall_search = _import_recall()
        mock_index = _make_mock_index(
            vector_count=10,
            search_results=[(0.9, "id1", {})],
        )
        # top_k=1 => recall_k = min(3, 10) = 3
        recall_search(_DUMMY_QUERY, mock_index, top_k=1)
        call_kwargs = mock_index.search_with_metadata.call_args[1]
        assert call_kwargs["top_k"] == 3


class TestRecallThreshold:
    """Results below threshold are filtered out."""

    def test_filters_results_below_threshold(self):
        recall_search = _import_recall()
        raw = [
            (0.9, "id1", {"path": "/a.jpg"}),
            (0.6, "id2", {"path": "/b.jpg"}),
            (0.3, "id3", {"path": "/c.jpg"}),
            (0.1, "id4", {"path": "/d.jpg"}),
        ]
        mock_index = _make_mock_index(search_results=raw)
        results = recall_search(_DUMMY_QUERY, mock_index, top_k=10, threshold=0.5)
        assert len(results) == 2
        for score, eid, meta in results:
            assert score >= 0.5

    def test_threshold_zero_passes_all(self):
        recall_search = _import_recall()
        raw = [(0.0, "id1", {}), (0.5, "id2", {}), (1.0, "id3", {})]
        mock_index = _make_mock_index(search_results=raw)
        results = recall_search(_DUMMY_QUERY, mock_index, top_k=10, threshold=0.0)
        assert len(results) == 3

    def test_threshold_one_passes_perfect_only(self):
        recall_search = _import_recall()
        raw = [(1.0, "id1", {}), (0.99, "id2", {}), (0.5, "id3", {})]
        mock_index = _make_mock_index(search_results=raw)
        results = recall_search(_DUMMY_QUERY, mock_index, top_k=10, threshold=1.0)
        assert len(results) == 1
        assert results[0][1] == "id1"

    def test_no_results_above_threshold(self):
        recall_search = _import_recall()
        raw = [(0.3, "id1", {}), (0.2, "id2", {})]
        mock_index = _make_mock_index(search_results=raw)
        results = recall_search(_DUMMY_QUERY, mock_index, top_k=10, threshold=0.5)
        assert results == []


class TestRecallWhereClause:
    """Metadata filter (where) is passed through to the index."""

    def test_where_passthrough(self):
        recall_search = _import_recall()
        mock_index = _make_mock_index(search_results=[])
        where_filter = {"ext": {"$eq": ".jpg"}}
        recall_search(_DUMMY_QUERY, mock_index, top_k=10, where=where_filter)
        call_kwargs = mock_index.search_with_metadata.call_args[1]
        assert call_kwargs["where"] == where_filter

    def test_where_none_by_default(self):
        recall_search = _import_recall()
        mock_index = _make_mock_index(search_results=[])
        recall_search(_DUMMY_QUERY, mock_index, top_k=10)
        call_kwargs = mock_index.search_with_metadata.call_args[1]
        assert call_kwargs["where"] is None

    def test_where_empty_dict(self):
        recall_search = _import_recall()
        mock_index = _make_mock_index(search_results=[])
        recall_search(_DUMMY_QUERY, mock_index, top_k=10, where={})
        call_kwargs = mock_index.search_with_metadata.call_args[1]
        assert call_kwargs["where"] == {}


class TestRecallResultsPassThrough:
    """Results pass through from the index (recall does not re-sort)."""

    def test_results_preserved_from_index(self):
        recall_search = _import_recall()
        raw = [
            (0.9, "id1", {}),
            (0.6, "id2", {}),
            (0.3, "id3", {}),
        ]
        mock_index = _make_mock_index(search_results=raw)
        results = recall_search(_DUMMY_QUERY, mock_index, top_k=10)
        assert results == raw
