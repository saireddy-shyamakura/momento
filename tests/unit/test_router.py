"""Unit tests for QueryRouter and QueryType in retrieval/router.py.

Tests the rule-based query classification:
- File extensions → EXACT
- Path prefixes → EXACT
- Short queries (1-2 words) → HYBRID
- Long/descriptive queries → SEMANTIC
- Edge cases (empty, whitespace, mixed case)
"""
import pytest


def _import_router():
    from momento.retrieval.router import QueryRouter, QueryType
    return QueryRouter, QueryType


class TestRouterExactMatch:
    """Queries that look like filenames/paths → EXACT."""

    def test_file_extension_jpg(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("photo.jpg") == QueryType.EXACT

    def test_file_extension_png(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("image.png") == QueryType.EXACT

    def test_file_extension_webp(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("pic.webp") == QueryType.EXACT

    def test_file_extension_mp4(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("video.mp4") == QueryType.EXACT

    def test_file_extension_uppercase(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("IMAGE.JPG") == QueryType.EXACT

    def test_path_prefix_slash(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("/home/user/photo.jpg") == QueryType.EXACT

    def test_path_prefix_tilde(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("~/Pictures/vacation") == QueryType.EXACT

    def test_path_prefix_dot_slash(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("./images/photo.png") == QueryType.EXACT

    def test_path_prefix_dot_dot_slash(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("../data/image.jpg") == QueryType.EXACT

    def test_file_without_extension_not_exact(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        # Just a word, not a path → HYBRID or SEMANTIC
        assert router.classify("sunset") != QueryType.EXACT


class TestRouterHybrid:
    """Short queries (1-2 words) → HYBRID."""

    def test_single_word(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("dog") == QueryType.HYBRID

    def test_two_words(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("red car") == QueryType.HYBRID

    def test_two_words_with_punctuation(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("sunset, beach") == QueryType.HYBRID

    def test_single_word_with_numbers(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("photo123") == QueryType.HYBRID


class TestRouterSemantic:
    """Natural language / multi-word queries → SEMANTIC."""

    def test_three_words(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("dog in park") == QueryType.SEMANTIC

    def test_full_sentence(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("a photo of a dog playing in the park") == QueryType.SEMANTIC

    def test_question(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("what is this image about") == QueryType.SEMANTIC

    def test_descriptive_query(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("people walking on a beach at sunset") == QueryType.SEMANTIC


class TestRouterEdgeCases:
    """Edge cases for query classification."""

    def test_empty_string(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        # Empty string has 0 words, which is <= 2 → HYBRID
        assert router.classify("") == QueryType.HYBRID

    def test_whitespace_only(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        # Whitespace has 0 words, which is <= 2 → HYBRID
        assert router.classify("   ") == QueryType.HYBRID

    def test_special_characters(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.classify("cat & dog") == QueryType.SEMANTIC

    def test_extension_in_middle_of_string(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        # "sunset.jpg" ends with .jpg → EXACT
        assert router.classify("sunset.jpg") == QueryType.EXACT

    def test_extension_not_at_end(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        # ".jpg" is not at end (query is "jpg image") → should be HYBRID (2 words)
        assert router.classify("jpg image") == QueryType.HYBRID


class TestRouterRouteString:
    """route() returns human-readable string."""

    def test_semantic_route_string(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.route("a dog playing fetch") == "vector_search"

    def test_exact_route_string(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.route("photo.jpg") == "exact_lookup"

    def test_hybrid_route_string(self):
        QueryRouter, QueryType = _import_router()
        router = QueryRouter()
        assert router.route("dog") == "hybrid_search"