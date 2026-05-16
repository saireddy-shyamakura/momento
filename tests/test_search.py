"""Example-based unit tests for the _search() threshold filtering logic in search.py.

Validates: Requirements 2.1, 2.2, 2.3
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Lazy import of _search to avoid pulling in torch / chromadb at module level.
# ---------------------------------------------------------------------------

def _import_search():
    """Import _search without triggering torch or chromadb at module level."""
    import search as _search_module
    return _search_module._search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_index(results, is_built=True):
    """
    Return a MagicMock that mimics the Index interface used by _search().

    Args:
        results: List of (score, path) tuples that index.search() will return.
        is_built: Value returned by index.is_built().
    """
    mock = MagicMock()
    mock.is_built.return_value = is_built
    mock.search.return_value = results
    return mock


# A dummy query vector — _search() passes it straight through to index.search(),
# so its actual value does not matter for these unit tests.
_DUMMY_QUERY = object()
_TOP_K = 10
_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Requirement 2.1 — only results >= threshold are returned
# ---------------------------------------------------------------------------

class TestThresholdFiltering:
    """Requirement 2.1 — _search() returns only results whose score >= threshold."""

    def test_returns_only_results_above_threshold(self):
        """Mixed scores: only those >= threshold should be returned."""
        _search = _import_search()

        raw = [
            (0.9, "/img/a.jpg"),
            (0.7, "/img/b.jpg"),
            (0.5, "/img/c.jpg"),   # exactly at threshold — should be included
            (0.49, "/img/d.jpg"),  # just below — should be excluded
            (0.2, "/img/e.jpg"),
        ]
        mock_index = _make_mock_index(raw)

        results = _search(_DUMMY_QUERY, _TOP_K, mock_index, threshold=_THRESHOLD)

        assert results == [
            (0.9, "/img/a.jpg"),
            (0.7, "/img/b.jpg"),
            (0.5, "/img/c.jpg"),
        ]

    def test_all_results_above_threshold_are_returned(self):
        """When every score is above threshold, all results are returned."""
        _search = _import_search()

        raw = [
            (0.95, "/img/x.jpg"),
            (0.80, "/img/y.jpg"),
            (0.60, "/img/z.jpg"),
        ]
        mock_index = _make_mock_index(raw)

        results = _search(_DUMMY_QUERY, _TOP_K, mock_index, threshold=0.5)

        assert results == raw

    def test_exact_threshold_score_is_included(self):
        """A result whose score equals the threshold exactly must be included."""
        _search = _import_search()

        threshold = 0.75
        raw = [(threshold, "/img/exact.jpg")]
        mock_index = _make_mock_index(raw)

        results = _search(_DUMMY_QUERY, _TOP_K, mock_index, threshold=threshold)

        assert results == [(threshold, "/img/exact.jpg")]

    def test_score_just_below_threshold_is_excluded(self):
        """A result whose score is just below the threshold must be excluded."""
        _search = _import_search()

        threshold = 0.75
        raw = [(0.7499, "/img/close.jpg")]
        mock_index = _make_mock_index(raw)

        results = _search(_DUMMY_QUERY, _TOP_K, mock_index, threshold=threshold)

        assert results == []


# ---------------------------------------------------------------------------
# Requirement 2.2 — empty list when no results are above threshold
# ---------------------------------------------------------------------------

class TestAllBelowThreshold:
    """Requirement 2.2 — _search() returns [] when no score meets the threshold."""

    def test_empty_list_when_all_scores_below_threshold(self):
        """All scores below threshold → empty list returned."""
        _search = _import_search()

        raw = [
            (0.1, "/img/a.jpg"),
            (0.3, "/img/b.jpg"),
            (0.49, "/img/c.jpg"),
        ]
        mock_index = _make_mock_index(raw)

        results = _search(_DUMMY_QUERY, _TOP_K, mock_index, threshold=_THRESHOLD)

        assert results == []

    def test_empty_list_when_index_returns_no_results(self):
        """Index.search() returns [] → _search() also returns []."""
        _search = _import_search()

        mock_index = _make_mock_index([])

        results = _search(_DUMMY_QUERY, _TOP_K, mock_index, threshold=_THRESHOLD)

        assert results == []

    def test_threshold_of_one_excludes_all_but_perfect_match(self):
        """With threshold=1.0, only a perfect score of 1.0 passes."""
        _search = _import_search()

        raw = [
            (1.0, "/img/perfect.jpg"),
            (0.99, "/img/almost.jpg"),
        ]
        mock_index = _make_mock_index(raw)

        results = _search(_DUMMY_QUERY, _TOP_K, mock_index, threshold=1.0)

        assert results == [(1.0, "/img/perfect.jpg")]

    def test_threshold_of_zero_passes_all_results(self):
        """With threshold=0.0, every result (including score 0.0) passes."""
        _search = _import_search()

        raw = [
            (0.0, "/img/zero.jpg"),
            (0.5, "/img/mid.jpg"),
            (1.0, "/img/top.jpg"),
        ]
        mock_index = _make_mock_index(raw)

        results = _search(_DUMMY_QUERY, _TOP_K, mock_index, threshold=0.0)

        assert results == raw


# ---------------------------------------------------------------------------
# Requirement 2.3 — empty list when Index.is_built() returns False
# ---------------------------------------------------------------------------

class TestIndexNotBuilt:
    """Requirement 2.3 — _search() returns [] without exception when index is empty."""

    def test_returns_empty_list_when_index_not_built(self):
        """is_built() == False → _search() returns [] immediately."""
        _search = _import_search()

        mock_index = _make_mock_index([], is_built=False)

        results = _search(_DUMMY_QUERY, _TOP_K, mock_index, threshold=_THRESHOLD)

        assert results == []

    def test_does_not_call_search_when_index_not_built(self):
        """When is_built() is False, index.search() must never be called."""
        _search = _import_search()

        mock_index = _make_mock_index([], is_built=False)

        _search(_DUMMY_QUERY, _TOP_K, mock_index, threshold=_THRESHOLD)

        mock_index.search.assert_not_called()

    def test_no_exception_when_index_not_built(self):
        """Calling _search() on an unbuilt index must not raise any exception."""
        _search = _import_search()

        mock_index = _make_mock_index([], is_built=False)

        try:
            result = _search(_DUMMY_QUERY, _TOP_K, mock_index, threshold=_THRESHOLD)
        except Exception as exc:
            pytest.fail(f"_search() raised an unexpected exception: {exc}")

        assert result == []

    def test_returns_empty_list_regardless_of_would_be_results(self):
        """Even if index.search() would return results, is_built()=False wins."""
        _search = _import_search()

        # Provide results that would pass the threshold — but is_built is False
        mock_index = _make_mock_index(
            [(0.9, "/img/a.jpg"), (0.8, "/img/b.jpg")],
            is_built=False,
        )

        results = _search(_DUMMY_QUERY, _TOP_K, mock_index, threshold=_THRESHOLD)

        assert results == []
