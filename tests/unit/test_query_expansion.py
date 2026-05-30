"""Unit tests for expand_query() in retrieval/query_expansion.py.

Tests the query expansion logic including:
- Empty query → []
- Synonym injection for known keywords
- "photo of" variant for single unknown words
- max_variants limit
- No duplicate variants
- Case-insensitive matching
- Multi-keyword queries
"""
import pytest


def _import_expand():
    from momento.retrieval.query_expansion import expand_query
    return expand_query


class TestQueryExpansionEmpty:
    """Empty/invalid queries."""

    def test_empty_string(self):
        expand_query = _import_expand()
        assert expand_query("") == []

    def test_whitespace_only(self):
        expand_query = _import_expand()
        assert expand_query("   ") == []


class TestQueryExpansionSynonyms:
    """Known keywords generate synonym variants."""

    def test_single_keyword_expands(self):
        expand_query = _import_expand()
        results = expand_query("dog")
        assert "dog" in results
        assert len(results) > 1

    def test_synonym_variants_present(self):
        expand_query = _import_expand()
        results = expand_query("dog")
        # At minimum should include "dog" and at least one synonym variant
        assert len(results) >= 2

    def test_known_synonym_cat(self):
        expand_query = _import_expand()
        results = expand_query("cat")
        assert "cat" in results
        assert any("kitten" in r or "feline" in r for r in results)

    def test_case_insensitive_matching(self):
        expand_query = _import_expand()
        results = expand_query("DOG")
        assert any("dog" in r.lower() for r in results)


class TestQueryExpansionPhotoOf:
    """Single unknown words get 'photo of' variant."""

    def test_single_unknown_word_gets_photo_variant(self):
        expand_query = _import_expand()
        # "sunset" is not in the synonym map
        results = expand_query("sunset")
        assert "sunset" in results
        assert "a photo of sunset" in results

    def test_known_word_with_limited_max_variants_gets_no_photo_variant(self):
        expand_query = _import_expand()
        # "dog" IS in synonym map — with max_variants=4, only synonyms fit
        results = expand_query("dog", max_variants=4)
        assert "a photo of dog" not in results
        assert len(results) == 4


class TestQueryExpansionMultiWord:
    """Multi-word queries with keyword matching."""

    def test_multi_word_with_keyword(self):
        expand_query = _import_expand()
        results = expand_query("dog in park")
        assert "dog in park" in results

    def test_multi_word_without_known_keywords(self):
        expand_query = _import_expand()
        results = expand_query("funny moment")
        assert "funny moment" in results
        # No known keywords, no photo variant (multi-word)
        assert len(results) == 1

    def test_keyword_in_middle_of_query(self):
        expand_query = _import_expand()
        results = expand_query("a cat in the garden")
        assert "a cat in the garden" in results


class TestQueryExpansionLimits:
    """max_variants limits the output."""

    def test_max_variants_limit(self):
        expand_query = _import_expand()
        results = expand_query("dog", max_variants=3)
        assert len(results) <= 3

    def test_max_variants_one(self):
        expand_query = _import_expand()
        results = expand_query("dog", max_variants=1)
        assert results == ["dog"]

    def test_max_variants_ten(self):
        expand_query = _import_expand()
        results = expand_query("dog", max_variants=10)
        assert len(results) <= 10
        assert len(results) >= 2


class TestQueryExpansionNoDuplicates:
    """No duplicate variants in the result."""

    def test_no_duplicate_variants(self):
        expand_query = _import_expand()
        results = expand_query("dog", max_variants=10)
        assert len(results) == len(set(results))

    def test_original_always_present(self):
        expand_query = _import_expand()
        results = expand_query("car")
        assert "car" in results


class TestQueryExpansionMultipleKeywords:
    """Multiple keyword matches in a query."""

    def test_two_known_keywords(self):
        expand_query = _import_expand()
        results = expand_query("dog car", max_variants=5)
        assert "dog car" in results
        # Should match at least one keyword
        assert len(results) >= 2