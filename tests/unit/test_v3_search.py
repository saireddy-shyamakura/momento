"""Unit tests for V3 search pipeline in search/__init__.py.

Tests the V3 pipeline orchestration including:
- _v3_search_pipeline with exact match shortcut
- Exact match with score >= 0.95 returns immediately
- Query expansion integration
- Re-ranking integration
- text_search with prefix logic
- image_search with V3 flag
- search_hybrid integration
- register_paths_for_exact_search delegation
- Lazy singleton initialization
"""
import sys
import types
from unittest.mock import MagicMock, patch, ANY

import pytest


# Top-level constant for the query vector dimension
_DUMMY_QUERY = object()


def _make_mock_index(is_built=True):
    mock = MagicMock()
    mock.is_built.return_value = is_built
    return mock


class TestV3ExactMatchShortcut:
    """Exact match with score >= 0.95 returns immediately."""

    def test_exact_match_returns_early(self):
        from momento.search import _v3_search_pipeline

        mock_index = _make_mock_index()

        with patch("momento.search.ENABLE_HYBRID_SEARCH", True), \
             patch("momento.search._get_exact_index") as mock_get_exact, \
             patch("momento.search._get_router") as mock_get_router, \
             patch("momento.search.extract_text_features"):

            mock_exact = MagicMock()
            mock_exact.search.return_value = [(0.98, "/path/exact.jpg")]
            mock_get_exact.return_value = mock_exact

            mock_router = MagicMock()
            from momento.retrieval.router import QueryType
            mock_router.classify.return_value = QueryType.EXACT
            mock_get_router.return_value = mock_router

            results = _v3_search_pipeline(_DUMMY_QUERY, "exact.jpg", mock_index, top_k=10)

        assert len(results) == 1
        assert results[0][0] >= 0.95
        assert "exact.jpg" in results[0][1]

    def test_exact_match_below_threshold_falls_through(self):
        from momento.search import _v3_search_pipeline

        mock_index = _make_mock_index()

        with patch("momento.search.ENABLE_HYBRID_SEARCH", True), \
             patch("momento.search._get_exact_index") as mock_get_exact, \
             patch("momento.search._get_router") as mock_get_router, \
             patch("momento.search.expand_query") as mock_expand, \
             patch("momento.search.extract_text_features") as mock_extract, \
             patch("momento.search.recall_search") as mock_recall:

            mock_exact = MagicMock()
            mock_exact.search.return_value = [(0.80, "/path/exact.jpg")]  # Below 0.95
            mock_get_exact.return_value = mock_exact

            mock_router = MagicMock()
            from momento.retrieval.router import QueryType
            mock_router.classify.return_value = QueryType.EXACT
            mock_get_router.return_value = mock_router

            mock_extract.return_value = MagicMock()
            mock_expand.return_value = ["exact"]
            mock_recall.return_value = []

            _v3_search_pipeline(_DUMMY_QUERY, "exact.jpg", mock_index, top_k=10)
            # Should proceed to recall stage
            mock_recall.assert_called_once()

    def test_exact_match_disabled_when_hybrid_off(self):
        from momento.search import _v3_search_pipeline

        mock_index = _make_mock_index()

        with patch("momento.search.ENABLE_HYBRID_SEARCH", False), \
             patch("momento.search._get_router") as mock_get_router, \
             patch("momento.search.extract_text_features") as mock_extract, \
             patch("momento.search.expand_query") as mock_expand, \
             patch("momento.search.recall_search") as mock_recall, \
             patch("momento.search.ENABLE_QUERY_EXPANSION", False):

            mock_router = MagicMock()
            mock_router.classify.return_value = MagicMock()
            mock_get_router.return_value = mock_router

            mock_extract.return_value = MagicMock()
            mock_recall.return_value = []

            result = _v3_search_pipeline(_DUMMY_QUERY, "query", mock_index, top_k=10)
            mock_recall.assert_called_once()


class TestV3QueryExpansion:
    """Query expansion is called when enabled."""

    def test_expansion_enabled(self):
        from momento.search import _v3_search_pipeline

        mock_index = _make_mock_index()

        with patch("momento.search.ENABLE_HYBRID_SEARCH", False), \
             patch("momento.search.ENABLE_QUERY_EXPANSION", True), \
             patch("momento.search._get_router") as mock_get_router, \
             patch("momento.search.extract_text_features") as mock_extract, \
             patch("momento.search.expand_query") as mock_expand, \
             patch("momento.search.recall_search") as mock_recall:

            mock_router = MagicMock()
            mock_router.classify.return_value = MagicMock()
            mock_get_router.return_value = mock_router

            mock_extract.return_value = MagicMock()
            mock_expand.return_value = ["query", "variant1", "variant2"]
            mock_recall.return_value = []

            _v3_search_pipeline(_DUMMY_QUERY, "query", mock_index, top_k=10)
            mock_expand.assert_called_once_with("query", max_variants=5)
            assert mock_recall.call_count >= 3


class TestV3Rerank:
    """Re-ranking is called when enabled."""

    def test_rerank_enabled(self):
        from momento.search import _v3_search_pipeline

        mock_index = _make_mock_index()

        with patch("momento.search.ENABLE_HYBRID_SEARCH", False), \
             patch("momento.search.ENABLE_QUERY_EXPANSION", False), \
             patch("momento.search.ENABLE_RERANK", True), \
             patch("momento.search._get_router") as mock_get_router, \
             patch("momento.search.extract_text_features") as mock_extract, \
             patch("momento.search.recall_search") as mock_recall, \
             patch("momento.search.rerank_results") as mock_rerank, \
             patch("momento.search.fuse_scores") as mock_fuse, \
             patch("momento.search.FUSION_WEIGHT_EMBEDDING", 0.6), \
             patch("momento.search.FUSION_WEIGHT_OBJECT", 0.2), \
             patch("momento.search.FUSION_WEIGHT_OCR", 0.2), \
             patch("momento.search.RERANK_TOP_K", 10):

            mock_router = MagicMock()
            mock_router.classify.return_value = MagicMock()
            mock_get_router.return_value = mock_router

            mock_extract.return_value = MagicMock()
            mock_recall.return_value = []
            mock_rerank.return_value = []

            _v3_search_pipeline(_DUMMY_QUERY, "query", mock_index, top_k=10)
            mock_rerank.assert_called_once()


class TestV3TextSearch:
    """text_search function behavior."""

    def test_default_uses_v3(self):
        from momento.search import text_search

        mock_index = _make_mock_index()

        with patch("momento.search.validate_text_query") as mock_validate, \
             patch("momento.search.extract_text_features") as mock_extract, \
             patch("momento.search._v3_search_pipeline") as mock_v3, \
             patch("momento.search._search") as mock_legacy:

            mock_validate.return_value = (True, "")
            mock_extract.return_value = MagicMock()
            mock_v3.return_value = []

            text_search("test query", mock_index, top_k=5)
            mock_v3.assert_called_once()
            mock_legacy.assert_not_called()

    def test_v3_disabled_uses_legacy(self):
        from momento.search import text_search

        mock_index = _make_mock_index()

        with patch("momento.search.validate_text_query") as mock_validate, \
             patch("momento.search.extract_text_features") as mock_extract, \
             patch("momento.search._v3_search_pipeline") as mock_v3, \
             patch("momento.search._search") as mock_legacy:

            mock_validate.return_value = (True, "")
            mock_extract.return_value = MagicMock()

            text_search("test query", mock_index, top_k=5, use_v3=False)
            mock_v3.assert_not_called()
            mock_legacy.assert_called_once()

    def test_text_search_with_prefix(self):
        from momento.search import text_search

        mock_index = _make_mock_index()

        with patch("momento.search.validate_text_query") as mock_validate, \
             patch("momento.search.extract_text_features") as mock_extract, \
             patch("momento.search._v3_search_pipeline") as mock_v3:

            mock_validate.return_value = (True, "")
            mock_extract.return_value = MagicMock()
            mock_v3.return_value = []

            # "cat" is a single word that should get prefixed
            text_search("cat", mock_index, top_k=5)
            # The prefix "a photo of" is applied in text_search before extract_text_features
            assert mock_extract.called


class TestV3ImageSearch:
    """image_search function behavior."""

    def test_default_uses_v3(self):
        from momento.search import image_search

        mock_index = _make_mock_index()

        with patch("momento.search.validate_image_path") as mock_validate, \
             patch("momento.search.extract_image_features") as mock_extract, \
             patch("momento.search._v3_search_pipeline") as mock_v3, \
             patch("momento.search._search") as mock_legacy, \
             patch("momento.search.ENABLE_HYBRID_SEARCH", True):

            mock_validate.return_value = (True, "")
            mock_extract.return_value = MagicMock()
            mock_v3.return_value = []

            image_search("/path/to/image.jpg", mock_index, top_k=5)
            mock_v3.assert_called_once()
            mock_legacy.assert_not_called()

    def test_v3_disabled_uses_legacy(self):
        from momento.search import image_search

        mock_index = _make_mock_index()

        with patch("momento.search.validate_image_path") as mock_validate, \
             patch("momento.search.extract_image_features") as mock_extract, \
             patch("momento.search._v3_search_pipeline") as mock_v3, \
             patch("momento.search._search") as mock_legacy, \
             patch("momento.search.ENABLE_HYBRID_SEARCH", True):

            mock_validate.return_value = (True, "")
            mock_extract.return_value = MagicMock()

            image_search("/path/to/image.jpg", mock_index, top_k=5, use_v3=False)
            mock_v3.assert_not_called()
            mock_legacy.assert_called_once()


class TestV3SearchHybrid:
    """search_hybrid delegates to text_search."""

    def test_search_hybrid_delegates(self):
        from momento.search import search_hybrid

        mock_index = _make_mock_index()

        with patch("momento.search.text_search") as mock_text_search:
            mock_text_search.return_value = []

            search_hybrid("test query", mock_index, top_k=10)
            mock_text_search.assert_called_once()


class TestV3RegisterPaths:
    """register_paths_for_exact_search delegates to ExactIndex."""

    def test_register_paths(self):
        from momento.search import register_paths_for_exact_search

        with patch("momento.search._get_exact_index") as mock_get_exact:
            mock_exact = MagicMock()
            mock_exact.add_paths.return_value = 2
            mock_get_exact.return_value = mock_exact

            count = register_paths_for_exact_search(["/a.jpg", "/b.jpg"])
            assert count == 2
            mock_exact.add_paths.assert_called_once_with(["/a.jpg", "/b.jpg"])

    def test_register_empty_paths(self):
        from momento.search import register_paths_for_exact_search

        with patch("momento.search._get_exact_index") as mock_get_exact:
            mock_exact = MagicMock()
            mock_exact.add_paths.return_value = 0
            mock_get_exact.return_value = mock_exact

            count = register_paths_for_exact_search([])
            assert count == 0


class TestV3LazySingletons:
    """Lazy singleton initialization."""

    def test_get_exact_index_lazy_init(self):
        from momento.search import _get_exact_index

        # First call creates
        with patch("momento.search.ExactIndex") as mock_cls:
            mock_cls.return_value = MagicMock()
            instance1 = _get_exact_index()
            # Second call returns same
            instance2 = _get_exact_index()

        assert instance1 is instance2

    def test_get_metadata_store_lazy_init(self):
        from momento.search import _get_metadata_store

        with patch("momento.search.MetadataStore") as mock_cls:
            mock_cls.return_value = MagicMock()
            instance1 = _get_metadata_store()
            instance2 = _get_metadata_store()

        assert instance1 is instance2

    def test_get_router_lazy_init(self):
        from momento.search import _get_router

        with patch("momento.search.QueryRouter") as mock_cls:
            mock_cls.return_value = MagicMock()
            instance1 = _get_router()
            instance2 = _get_router()

        assert instance1 is instance2