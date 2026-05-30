"""
query_expansion.py — Query expansion for Momento V3.

Generates multiple query variations to improve recall.
- Rule-based: synonym injection, sub-phrase extraction
- Future: LLM-based expansion (pluggable)
"""
from typing import List, Optional

from ..logger import get_logger

logger = get_logger(__name__)

# Simple synonym map for common visual search terms
_SYNONYMS: dict = {
    "dog": ["dog", "puppy", "canine", "dog outdoor"],
    "cat": ["cat", "kitten", "feline"],
    "car": ["car", "vehicle", "automobile"],
    "person": ["person", "people", "human", "man", "woman"],
    "tree": ["tree", "plant", "vegetation"],
    "water": ["water", "ocean", "sea", "river", "lake"],
    "food": ["food", "meal", "dish", "cuisine"],
    "building": ["building", "house", "home", "architecture"],
    "sunset": ["sunset", "sunrise", "dusk", "evening sky"],
    "flower": ["flower", "blossom", "bloom", "floral"],
}


def expand_query(query: str, max_variants: int = 5) -> List[str]:
    """Generate query variants to improve recall.

    Uses rule-based expansion:
    1. Original query is always included.
    2. If a known keyword is found, inject synonyms.
    3. If query is long, extract sub-phrases.

    Args:
        query: Original text query.
        max_variants: Maximum number of expanded queries to generate.

    Returns:
        List of query strings (original + variants).
    """
    original = query.strip()
    if not original:
        return []

    variants: List[str] = [original]
    lower = original.lower()

    # Synonym expansion: if any keyword matches, generate variants
    for keyword, synonyms in _SYNONYMS.items():
        if keyword in lower:
            for syn in synonyms:
                if syn != keyword:
                    expanded = original.lower().replace(keyword, syn, 1)
                    if expanded not in variants:
                        variants.append(expanded)
                    if len(variants) >= max_variants:
                        break
        if len(variants) >= max_variants:
            break

    # Short queries (1 word) get a "photo of" variant
    if len(original.split()) == 1 and len(variants) < max_variants:
        photo_variant = f"a photo of {original}"
        if photo_variant not in variants:
            variants.append(photo_variant)

    # Truncate to max_variants
    result = variants[:max_variants]
    if len(result) > 1:
        logger.debug(f"Query expansion: '{original}' → {len(result)} variants")

    return result