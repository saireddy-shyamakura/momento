
"""
search — Hybrid search for Momento V3.

Combines:
- Exact search (SQLite FTS for filename/path matching)
- Semantic search (vector index)
- Hybrid mode (both)

Also re-exports legacy search functions for backward compatibility.
"""
from typing import List, Optional, Tuple
import numpy as np

from .exact_index import ExactIndex
from ..embedding import ClipBackend
from ..features import extract_image_features, extract_text_features
from ..index import Index
from ..config import (
    SIMILARITY_THRESHOLD,
    ENABLE_QUERY_EXPANSION,
    ENABLE_HYBRID_SEARCH,
    ENABLE_RERANK,
    RERANK_TOP_K,
    RECALL_MULTIPLIER,
    FUSION_WEIGHT_EMBEDDING,
    FUSION_WEIGHT_OBJECT,
    FUSION_WEIGHT_OCR,
)
from ..validation import validate_image_path, validate_text_query, ValidationError
from ..retrieval.recall import recall_search
from ..retrieval.rerank import rerank_results
from ..retrieval.fusion import fuse_scores, FusionWeights
from ..retrieval.router import QueryRouter, QueryType
from ..retrieval.query_expansion import expand_query
from ..storage.metadata_store import MetadataStore
from ..logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "ExactIndex",
    "image_search",
    "text_search",
    "search_hybrid",
    "register_paths_for_exact_search",
]

# Lazy-initialized global instances
_exact_index: Optional[ExactIndex] = None
_metadata_store: Optional[MetadataStore] = None
_router: Optional[QueryRouter] = None


def _get_exact_index() -> ExactIndex:
    global _exact_index
    if _exact_index is None:
        _exact_index = ExactIndex()
    return _exact_index


def _get_metadata_store() -> MetadataStore:
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = MetadataStore()
    return _metadata_store


def _get_router() -> QueryRouter:
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router


def _search(query_vector, top_k: int, index: Index, threshold: float = SIMILARITY_THRESHOLD,
            use_aggregation: bool = False,
            where: Optional[dict] = None) -> List[Tuple[float, str]]:
    """Internal search helper (legacy). Returns (score, path)."""
    if not index.is_built():
        return []

    if use_aggregation:
        results = index.search_aggregated(query_vector, top_k)
        return [(score, path) for score, path in results if score >= threshold]

    raw = index.search(query_vector, top_k, where=where)
    return [(score, entry_id) for score, entry_id in raw if score >= threshold]


def _v3_search_pipeline(
    query_vector: np.ndarray,
    query_text: str,
    index: Index,
    top_k: int = 10,
    threshold: float = SIMILARITY_THRESHOLD,
    use_aggregation: bool = False,
) -> List[Tuple[float, str]]:
    """V3 hybrid search pipeline.

    1. Exact index lookup (filename/path)
    2. Expanded query recall from vector index
    3. Optional re-ranking
    4. Optional cross-modal fusion
    5. Aggregate and return
    """
    router = _get_router()
    query_type = router.classify(query_text)

    # Step 1: Exact match (if hybrid search enabled)
    if ENABLE_HYBRID_SEARCH and query_type in (QueryType.EXACT, QueryType.HYBRID):
        exact = _get_exact_index()
        exact_results = exact.search(query_text, top_k=top_k)
        if exact_results and exact_results[0][0] >= 0.95:
            logger.debug(f"Exact match found for '{query_text}': {exact_results[0][1]}")
            return exact_results[:top_k]

    # Step 2: Query expansion (if enabled)
    query_variants = [query_text]
    if ENABLE_QUERY_EXPANSION:
        query_variants = expand_query(query_text, max_variants=5)

    # Step 3: Multi-query recall
    all_candidates: List[Tuple[float, str, dict]] = []
    seen_ids: set = set()

    for variant in query_variants:
        variant_vector = extract_text_features(variant).reshape(1, -1)
        candidates = recall_search(
            variant_vector,
            index,
            top_k=top_k,
            threshold=threshold * 0.8,
        )
        for score, eid, meta in candidates:
            if eid not in seen_ids:
                seen_ids.add(eid)
                all_candidates.append((score, eid, meta))

    logger.debug(f"Recall stage: {len(all_candidates)} unique candidates from "
                 f"{len(query_variants)} query variants")

    # Step 4: Re-ranking (if enabled)
    if ENABLE_RERANK:
        all_candidates = rerank_results(query_text, all_candidates, top_k=RERANK_TOP_K)

    # Step 5: Cross-modal fusion
    fusion_weights = FusionWeights(
        embedding=FUSION_WEIGHT_EMBEDDING,
        object_match=FUSION_WEIGHT_OBJECT,
        ocr_relevance=FUSION_WEIGHT_OCR,
    )
    embedding_scores = [(s, eid) for s, eid, _ in all_candidates]
    fused = fuse_scores(embedding_scores, weights=fusion_weights)

    # Apply threshold and trim
    results = [(s, p) for s, p in fused if s >= threshold]
    return results[:top_k]


def image_search(query_image_path: str, index: Index, top_k: int = 3,
                 threshold: float = SIMILARITY_THRESHOLD,
                 use_aggregation: bool = False,
                 use_v3: bool = True) -> List[Tuple[float, str]]:
    """Search for images similar to a query image.

    Uses V3 pipeline when use_v3=True (default), falling back to legacy.

    Args:
        query_image_path: Path to query image
        index: Index instance
        top_k: Number of top results to return
        threshold: Minimum similarity score
        use_aggregation: Aggregate multi-vector scores
        use_v3: Use V3 enhanced pipeline

    Returns:
        List of (similarity_score, image_path) tuples
    """
    is_valid, error_msg = validate_image_path(query_image_path)
    if not is_valid:
        raise ValidationError(f"Invalid query image: {error_msg}")

    query = extract_image_features(query_image_path).reshape(1, -1)

    if use_v3 and ENABLE_HYBRID_SEARCH:
        return _v3_search_pipeline(query, query_image_path, index, top_k, threshold, use_aggregation)

    return _search(query, top_k, index, threshold=threshold, use_aggregation=use_aggregation)


def _text_should_be_prefixed(text: str) -> bool:
    """Determine if a text query should be prefixed with 'a photo of'."""
    normalized = text.strip().lower()
    word_count = len(normalized.split())

    prefix_words = ("a ", "an ", "the ", "my ", "this ", "that ",
                    "some ", "any ", "these ", "those ", "what ",
                    "which ", "whose ", "photo ", "picture ", "image ")
    if normalized.startswith(prefix_words):
        return False

    if normalized[-1:] in (".", "!", "?"):
        return False

    verbs = {"is", "are", "was", "were", "be", "been", "being",
             "has", "have", "had", "do", "does", "did",
             "show", "find", "get", "give", "search", "look"}
    first_word = normalized.split()[0] if word_count >= 1 else ""
    if first_word in verbs:
        return False

    if word_count >= 6:
        return False

    return True


def text_search(text: str, index: Index, top_k: int = 3,
                threshold: float = SIMILARITY_THRESHOLD,
                use_aggregation: bool = False,
                use_v3: bool = True) -> List[Tuple[float, str]]:
    """Search for images matching a text description.

    Uses V3 pipeline when use_v3=True (default), falling back to legacy.

    Args:
        text: Text query
        index: Index instance
        top_k: Number of top results to return
        threshold: Minimum similarity score
        use_aggregation: Aggregate multi-vector scores
        use_v3: Use V3 enhanced pipeline

    Returns:
        List of (similarity_score, image_path) tuples
    """
    is_valid, error_msg = validate_text_query(text)
    if not is_valid:
        raise ValidationError(f"Invalid text query: {error_msg}")

    text = text.strip()

    # Only add prefix if query looks like a bare noun phrase (legacy behavior)
    if _text_should_be_prefixed(text):
        text = f"a photo of {text}"

    if use_v3:
        query = extract_text_features(text).reshape(1, -1)
        return _v3_search_pipeline(query, text, index, top_k, threshold, use_aggregation)

    query = extract_text_features(text).reshape(1, -1)
    return _search(query, top_k, index, threshold=threshold, use_aggregation=use_aggregation)


def search_hybrid(query: str, index: Index, top_k: int = 10,
                  threshold: float = SIMILARITY_THRESHOLD) -> List[Tuple[float, str]]:
    """V3 hybrid search combining exact + semantic + fusion.

    Args:
        query: Text query (can be filename, path, or semantic search)
        index: Index instance
        top_k: Number of results
        threshold: Minimum similarity score

    Returns:
        List of (score, path) tuples
    """
    return text_search(query, index, top_k=top_k, threshold=threshold, use_v3=True)


def register_paths_for_exact_search(paths: List[str]) -> int:
    """Register file paths in the exact search index.

    Should be called after indexing to enable exact filename lookups.

    Args:
        paths: List of absolute file paths.

    Returns:
        Number of paths indexed.
    """
    exact = _get_exact_index()
    return exact.add_paths(paths)