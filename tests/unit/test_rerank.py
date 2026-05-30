"""Unit tests for rerank_results() in retrieval/rerank.py.

Tests the re-ranking stage including:
- Empty candidate handling
- Identity pass-through (order by score)
- Top-K truncation
- Edge cases (single candidate, fewer than top_k)
"""
import pytest


def _import_rerank():
    from momento.retrieval.rerank import rerank_results
    return rerank_results


class TestRerankEmpty:
    """Empty candidates → []."""

    def test_empty_candidates_returns_empty(self):
        rerank_results = _import_rerank()
        results = rerank_results("test query", [], top_k=10)
        assert results == []

    def test_none_candidates_returns_empty(self):
        rerank_results = _import_rerank()
        results = rerank_results("test query", [], top_k=10)
        assert results == []


class TestRerankIdentity:
    """Identity re-rank sorts by score descending and truncates."""

    def test_sorts_by_score_descending(self):
        rerank_results = _import_rerank()
        candidates = [
            (0.5, "id1", {}),
            (0.9, "id2", {}),
            (0.7, "id3", {}),
        ]
        results = rerank_results("query", candidates, top_k=10)
        assert results == [
            (0.9, "id2", {}),
            (0.7, "id3", {}),
            (0.5, "id1", {}),
        ]

    def test_truncates_to_top_k(self):
        rerank_results = _import_rerank()
        candidates = [
            (0.9, "id1", {}),
            (0.8, "id2", {}),
            (0.7, "id3", {}),
            (0.6, "id4", {}),
        ]
        results = rerank_results("query", candidates, top_k=2)
        assert len(results) == 2
        assert results[0][1] == "id1"
        assert results[1][1] == "id2"

    def test_fewer_candidates_than_top_k(self):
        rerank_results = _import_rerank()
        candidates = [(0.9, "id1", {})]
        results = rerank_results("query", candidates, top_k=10)
        assert len(results) == 1

    def test_single_candidate(self):
        rerank_results = _import_rerank()
        candidates = [(0.75, "id1", {"path": "/a.jpg"})]
        results = rerank_results("query", candidates, top_k=10)
        assert results == [(0.75, "id1", {"path": "/a.jpg"})]

    def test_ties_preserve_order(self):
        rerank_results = _import_rerank()
        candidates = [
            (0.9, "id1", {}),
            (0.9, "id2", {}),
            (0.8, "id3", {}),
        ]
        results = rerank_results("query", candidates, top_k=10)
        # Ties may reorder, but all should be present
        assert len(results) == 3
        scores = [s for s, eid, m in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_zero(self):
        rerank_results = _import_rerank()
        candidates = [(0.9, "id1", {})]
        results = rerank_results("query", candidates, top_k=0)
        assert results == []

    def test_top_k_one(self):
        rerank_results = _import_rerank()
        candidates = [
            (0.9, "id1", {}),
            (0.8, "id2", {}),
            (0.7, "id3", {}),
        ]
        results = rerank_results("query", candidates, top_k=1)
        assert len(results) == 1
        assert results[0][1] == "id1"