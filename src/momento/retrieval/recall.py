"""
recall.py — Fast CLIP recall layer (first-stage retrieval).

Given a query, returns top-K candidates from the vector index using
cosine similarity. This is the "recall" stage — we intentionally
return more candidates than needed and let the reranker narrow down.
"""
from typing import List, Optional, Tuple
import numpy as np

from ..index import Index
from ..logger import get_logger

logger = get_logger(__name__)

# Number of candidates to recall before reranking
_RECALL_MULTIPLIER = 3  # recall 3x the final top_k


def recall_search(
    query_vector: np.ndarray,
    index: Index,
    top_k: int = 10,
    threshold: float = 0.0,
    where: Optional[dict] = None,
) -> List[Tuple[float, str, dict]]:
    """First-stage fast recall from vector index.

    Recalls more candidates than requested (top_k * _RECALL_MULTIPLIER)
    to give the reranker a wider pool to choose from.

    Args:
        query_vector: Normalized query embedding (1, dim).
        index: Vector index instance.
        top_k: Desired number of final results.
        threshold: Minimum similarity score.
        where: Optional metadata filter.

    Returns:
        List of (score, entry_id, metadata) tuples, sorted descending by score.
    """
    if not index.is_built():
        logger.debug("Recall skipped — index is empty")
        return []

    # Recall more than needed so reranker has material to work with
    recall_k = min(top_k * _RECALL_MULTIPLIER, index.get_vector_count())
    if recall_k == 0:
        return []

    results = index.search_with_metadata(query_vector, top_k=recall_k, where=where)

    # Apply threshold
    if threshold > 0.0:
        results = [(s, eid, m) for s, eid, m in results if s >= threshold]

    logger.debug(f"Recall stage: {len(results)} candidates from top-{recall_k}")
    return results