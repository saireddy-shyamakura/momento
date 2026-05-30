"""
fusion.py — Cross-modal fusion layer for Momento V3.

Combines multiple relevance signals into a single score:
- Embedding similarity (CLIP)
- Object detection match (YOLO)
- OCR text relevance

Formula:
    final_score = w_embed * embedding_score
                 + w_obj   * object_match
                 + w_ocr   * ocr_relevance
"""
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class FusionWeights:
    """Configurable weights for cross-modal fusion."""
    embedding: float = 0.6
    object_match: float = 0.2
    ocr_relevance: float = 0.2


def fuse_scores(
    embedding_scores: List[Tuple[float, str]],
    object_scores: Dict[str, float] = None,
    ocr_scores: Dict[str, float] = None,
    weights: FusionWeights = None,
) -> List[Tuple[float, str]]:
    """Fuse multiple relevance signals into combined scores.

    Args:
        embedding_scores: List of (score, path) from vector search.
        object_scores: Dict of path -> object match score (0-1).
        ocr_scores: Dict of path -> OCR relevance score (0-1).
        weights: Fusion weights. Defaults to 0.6/0.2/0.2.

    Returns:
        List of (fused_score, path) sorted descending.
    """
    if weights is None:
        weights = FusionWeights()

    object_scores = object_scores or {}
    ocr_scores = ocr_scores or {}

    fused: Dict[str, float] = {}

    for score, path in embedding_scores:
        obj = object_scores.get(path, 0.0)
        ocr = ocr_scores.get(path, 0.0)
        combined = (
            weights.embedding * score
            + weights.object_match * obj
            + weights.ocr_relevance * ocr
        )
        fused[path] = combined

    # Sort descending by fused score
    sorted_results = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    logger.debug(f"Fusion: combined {len(embedding_scores)} embeddings with "
                 f"{len(object_scores)} object and {len(ocr_scores)} OCR signals")
    return [(score, path) for path, score in sorted_results]