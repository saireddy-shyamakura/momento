"""
rerank.py — Optional re-ranking stage for Momento V3.

After fast CLIP recall, a stronger model can re-rank the top candidates.
Currently a no-op placeholder; pluggable for future cross-encoder models.
"""
from typing import List, Optional, Tuple
import numpy as np

from ..logger import get_logger

logger = get_logger(__name__)


def rerank_results(
    query_text: str,
    candidates: List[Tuple[float, str, dict]],
    top_k: int = 10,
) -> List[Tuple[float, str, dict]]:
    """Re-rank candidate results using a stronger model.

    Currently returns candidates sorted by original score (identity
    re-rank). Intended to be replaced with a cross-encoder or
    Cohere/SentenceTransformer re-ranker.

    Args:
        query_text: Original text query.
        candidates: List of (score, entry_id, metadata) from recall stage.
        top_k: Number of results to return.

    Returns:
            Re-ranked list of (score, entry_id, metadata).
    """
    if not candidates:
        return []

    # Identity re-rank: just return top_k by original score
    sorted_candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

    result = sorted_candidates[:top_k]
    logger.debug(f"Rerank stage: {len(result)} results (identity pass-through)")
    return result