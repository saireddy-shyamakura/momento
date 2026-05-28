"""Example-based unit tests for the Index class in index.py.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

import os
import numpy as np
import pytest

from momento.index import Index


# ── Helpers ──────────────────────────────────────────────────────────

def _make_index(tmp_path) -> Index:
    """Return a fresh Index backed by a temporary ChromaDB directory."""
    return Index(db_path=str(tmp_path / "chroma"))


def _random_vector(dim: int = 512) -> np.ndarray:
    """Return a random unit-normalised float32 vector of the given dimension."""
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _abs_path(tmp_path, name: str) -> str:
    """Return an absolute path string under tmp_path (file need not exist)."""
    return str(tmp_path / name)


# ── add_vectors → get_existing_ids round trip ────────────────────────

class TestAddVectorsGetExistingIds:
    """Requirement 3.1 — stored paths are retrievable via get_existing_ids."""

    def test_all_added_paths_returned(self, tmp_path):
        idx = _make_index(tmp_path)
        paths = [_abs_path(tmp_path, f"img_{i}.jpg") for i in range(5)]
        vectors = [_random_vector() for _ in paths]

        idx.add_vectors(paths, vectors)

        existing = idx.get_existing_ids(paths)
        assert existing == set(paths)

    def test_empty_paths_returns_empty_set(self, tmp_path):
        idx = _make_index(tmp_path)
        assert idx.get_existing_ids([]) == set()

    def test_partial_overlap(self, tmp_path):
        idx = _make_index(tmp_path)
        paths = [_abs_path(tmp_path, f"img_{i}.jpg") for i in range(4)]
        vectors = [_random_vector() for _ in paths]
        idx.add_vectors(paths[:2], vectors[:2])

        existing = idx.get_existing_ids(paths)
        assert existing == set(paths[:2])


# ── item_exists ───────────────────────────────────────────────────────

class TestItemExists:
    """Requirements 3.2, 3.3 — item_exists returns correct bool."""

    def test_returns_true_for_added_path(self, tmp_path):
        idx = _make_index(tmp_path)
        # item_exists calls os.path.abspath internally, so use an absolute path
        path = str((tmp_path / "photo.jpg").resolve())
        idx.add_vectors([path], [_random_vector()])

        assert idx.item_exists(path) is True

    def test_returns_false_for_unknown_path(self, tmp_path):
        idx = _make_index(tmp_path)
        unknown = str((tmp_path / "never_added.jpg").resolve())

        assert idx.item_exists(unknown) is False

    def test_returns_false_on_empty_index(self, tmp_path):
        idx = _make_index(tmp_path)
        assert idx.item_exists(str((tmp_path / "any.jpg").resolve())) is False


# ── search ────────────────────────────────────────────────────────────

class TestSearch:
    """Requirement 3.4 — search returns (score, path) tuples sorted descending."""

    def test_returns_tuples_on_nonempty_index(self, tmp_path):
        idx = _make_index(tmp_path)
        paths = [_abs_path(tmp_path, f"img_{i}.jpg") for i in range(3)]
        vectors = [_random_vector() for _ in paths]
        idx.add_vectors(paths, vectors)

        results = idx.search(vectors[0], top_k=3)

        assert len(results) > 0
        for score, path in results:
            assert isinstance(score, float)
            assert isinstance(path, str)

    def test_results_sorted_descending(self, tmp_path):
        idx = _make_index(tmp_path)
        paths = [_abs_path(tmp_path, f"img_{i}.jpg") for i in range(5)]
        vectors = [_random_vector() for _ in paths]
        idx.add_vectors(paths, vectors)

        results = idx.search(vectors[0], top_k=5)

        scores = [s for s, _ in results]
        assert scores == sorted(scores, reverse=True), (
            f"Search results not sorted descending: {scores}"
        )

    def test_returns_empty_list_on_empty_index(self, tmp_path):
        idx = _make_index(tmp_path)
        results = idx.search(_random_vector(), top_k=3)
        assert results == []

    def test_top_k_limits_results(self, tmp_path):
        idx = _make_index(tmp_path)
        paths = [_abs_path(tmp_path, f"img_{i}.jpg") for i in range(10)]
        vectors = [_random_vector() for _ in paths]
        idx.add_vectors(paths, vectors)

        results = idx.search(vectors[0], top_k=3)
        assert len(results) <= 3


# ── upsert idempotence ────────────────────────────────────────────────

class TestUpsertIdempotence:
    """Requirement 3.5 — adding the same path twice leaves count == 1."""

    def test_duplicate_add_leaves_count_one(self, tmp_path):
        idx = _make_index(tmp_path)
        path = _abs_path(tmp_path, "dup.jpg")
        vec = _random_vector()

        idx.add_vectors([path], [vec])
        idx.add_vectors([path], [vec])

        assert idx.get_vector_count() == 1

    def test_duplicate_add_with_different_vector_leaves_count_one(self, tmp_path):
        idx = _make_index(tmp_path)
        path = _abs_path(tmp_path, "dup.jpg")

        idx.add_vectors([path], [_random_vector()])
        idx.add_vectors([path], [_random_vector()])

        assert idx.get_vector_count() == 1


# ── get_vector_count ──────────────────────────────────────────────────

class TestGetVectorCount:
    """Requirement 3.6 — get_vector_count returns N after adding N unique vectors."""

    def test_count_zero_on_empty_index(self, tmp_path):
        idx = _make_index(tmp_path)
        assert idx.get_vector_count() == 0

    def test_count_equals_n_after_adding_n_vectors(self, tmp_path):
        idx = _make_index(tmp_path)
        n = 7
        paths = [_abs_path(tmp_path, f"img_{i}.jpg") for i in range(n)]
        vectors = [_random_vector() for _ in paths]

        idx.add_vectors(paths, vectors)

        assert idx.get_vector_count() == n

    def test_count_increments_correctly(self, tmp_path):
        idx = _make_index(tmp_path)
        for i in range(4):
            path = _abs_path(tmp_path, f"img_{i}.jpg")
            idx.add_vectors([path], [_random_vector()])
            assert idx.get_vector_count() == i + 1


# ── delete_all ────────────────────────────────────────────────────────

class TestDeleteAll:
    """delete_all() leaves get_vector_count() == 0."""

    def test_delete_all_empties_index(self, tmp_path):
        idx = _make_index(tmp_path)
        paths = [_abs_path(tmp_path, f"img_{i}.jpg") for i in range(5)]
        vectors = [_random_vector() for _ in paths]
        idx.add_vectors(paths, vectors)

        idx.delete_all()

        assert idx.get_vector_count() == 0

    def test_delete_all_on_empty_index_is_noop(self, tmp_path):
        idx = _make_index(tmp_path)
        idx.delete_all()  # should not raise
        assert idx.get_vector_count() == 0

    def test_can_add_after_delete_all(self, tmp_path):
        idx = _make_index(tmp_path)
        path = _abs_path(tmp_path, "img.jpg")
        idx.add_vectors([path], [_random_vector()])
        idx.delete_all()

        new_path = _abs_path(tmp_path, "new_img.jpg")
        idx.add_vectors([new_path], [_random_vector()])
        assert idx.get_vector_count() == 1


# ── get_all_paths ─────────────────────────────────────────────────────

class TestGetAllPaths:
    """get_all_paths() returns all stored paths."""

    def test_returns_all_added_paths(self, tmp_path):
        idx = _make_index(tmp_path)
        paths = [_abs_path(tmp_path, f"img_{i}.jpg") for i in range(6)]
        vectors = [_random_vector() for _ in paths]
        idx.add_vectors(paths, vectors)

        result = idx.get_all_paths()

        assert set(result) == set(paths)

    def test_returns_empty_list_on_empty_index(self, tmp_path):
        idx = _make_index(tmp_path)
        assert idx.get_all_paths() == []

    def test_returns_empty_list_after_delete_all(self, tmp_path):
        idx = _make_index(tmp_path)
        paths = [_abs_path(tmp_path, f"img_{i}.jpg") for i in range(3)]
        vectors = [_random_vector() for _ in paths]
        idx.add_vectors(paths, vectors)
        idx.delete_all()

        assert idx.get_all_paths() == []


# ── Property-Based Tests (Hypothesis) ────────────────────────────────

import tempfile
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
import numpy as np


# Dimension of CLIP ViT-B/16 embeddings (see config.py)
_DIM = 512


def _hyp_vector(dim: int = _DIM) -> np.ndarray:
    """Return a random unit-normalised float32 vector."""
    v = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        v[0] = 1.0
        norm = 1.0
    return v / norm


# ── Property 6: Index add → exists round trip ─────────────────────────
# Feature: momento-stable-release, Property 6: Index add → exists round trip
# Validates: Requirements 3.1, 3.2

@given(
    names=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"),
            min_size=1,
            max_size=20,
        ),
        min_size=1,
        max_size=10,
        unique=True,
    )
)
@settings(max_examples=100, deadline=None)
def test_property6_index_add_exists_round_trip(names):
    """
    For any set of unique path strings added via add_vectors(), a subsequent
    get_existing_ids() call with those same paths SHALL return all of them.

    Validates: Requirements 3.1, 3.2
    """
    # Feature: momento-stable-release, Property 6: Index add → exists round trip
    with tempfile.TemporaryDirectory() as tmp:
        idx = Index(db_path=os.path.join(tmp, "chroma"))
        paths = [os.path.join(tmp, f"{name}.jpg") for name in names]
        vectors = [_hyp_vector() for _ in paths]

        idx.add_vectors(paths, vectors)

        existing = idx.get_existing_ids(paths)
        assert existing == set(paths), (
            f"Expected all {len(paths)} paths to be found, got {len(existing)}"
        )


# ── Property 7: Index upsert idempotence ─────────────────────────────
# Feature: momento-stable-release, Property 7: Index upsert idempotence
# Validates: Requirements 3.5

@given(
    name=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=100, deadline=None)
def test_property7_index_upsert_idempotence(name):
    """
    Calling add_vectors() twice with the same path SHALL result in
    get_vector_count() returning 1 (no duplicates).

    Validates: Requirements 3.5
    """
    # Feature: momento-stable-release, Property 7: Index upsert idempotence
    with tempfile.TemporaryDirectory() as tmp:
        idx = Index(db_path=os.path.join(tmp, "chroma"))
        path = os.path.join(tmp, f"{name}.jpg")
        vec = _hyp_vector()

        idx.add_vectors([path], [vec])
        idx.add_vectors([path], [vec])

        count = idx.get_vector_count()
        assert count == 1, (
            f"Expected count == 1 after double-add of same path, got {count}"
        )


# ── Property 8: Index count invariant ────────────────────────────────
# Feature: momento-stable-release, Property 8: Index count invariant
# Validates: Requirements 3.6

@given(
    names=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"),
            min_size=1,
            max_size=20,
        ),
        min_size=1,
        max_size=15,
        unique=True,
    )
)
@settings(max_examples=100, deadline=None)
def test_property8_index_count_invariant(names):
    """
    After adding N unique paths via add_vectors(), get_vector_count() SHALL
    return N.

    Validates: Requirements 3.6
    """
    # Feature: momento-stable-release, Property 8: Index count invariant
    with tempfile.TemporaryDirectory() as tmp:
        idx = Index(db_path=os.path.join(tmp, "chroma"))
        paths = [os.path.join(tmp, f"{name}.jpg") for name in names]
        vectors = [_hyp_vector() for _ in paths]

        idx.add_vectors(paths, vectors)

        count = idx.get_vector_count()
        assert count == len(paths), (
            f"Expected count == {len(paths)}, got {count}"
        )


# ── Property 9: Search results sorted descending ─────────────────────
# Feature: momento-stable-release, Property 9: Search results sorted descending
# Validates: Requirements 3.4

@given(
    n=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=100, deadline=None)
def test_property9_search_results_sorted_descending(n):
    """
    For any non-empty Index and any query vector, the list returned by
    Index.search() SHALL be sorted in descending order by Similarity_Score.

    Validates: Requirements 3.4
    """
    # Feature: momento-stable-release, Property 9: Search results sorted descending
    with tempfile.TemporaryDirectory() as tmp:
        idx = Index(db_path=os.path.join(tmp, "chroma"))
        paths = [os.path.join(tmp, f"img_{i}.jpg") for i in range(n)]
        vectors = [_hyp_vector() for _ in paths]
        idx.add_vectors(paths, vectors)

        query = _hyp_vector()
        results = idx.search(query, top_k=n)

        assume(len(results) >= 2)

        scores = [score for score, _ in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Search results not sorted descending at index {i}: "
                f"{scores[i]} < {scores[i + 1]}"
            )
