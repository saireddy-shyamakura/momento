from typing import List, Tuple
from .features import extract_image_features, extract_text_features
from .index import Index
from .config import SIMILARITY_THRESHOLD
from .validation import validate_image_path, validate_text_query, ValidationError


def _search(query_vector, top_k: int, index: Index, threshold: float = SIMILARITY_THRESHOLD,
            use_aggregation: bool = False) -> List[Tuple[float, str]]:
    """
    Internal search helper.
    
    Args:
        query_vector: Query vector (shape: 1, dim)
        top_k: Number of results
        index: Index instance
        threshold: Minimum similarity score for a match to be considered valid
        use_aggregation: If True, aggregate scores across composite IDs
        
    Returns:
        List of (score, path) tuples
    """
    if not index.is_built():
        return []

    if use_aggregation:
        results = index.search_aggregated(query_vector, top_k)
        return [(score, path) for score, path in results if score >= threshold]

    raw = index.search(query_vector, top_k)
    return [(score, entry_id) for score, entry_id in raw if score >= threshold]


def image_search(query_image_path: str, index: Index, top_k: int = 3,
                 threshold: float = SIMILARITY_THRESHOLD,
                 use_aggregation: bool = False) -> List[Tuple[float, str]]:
    """
    Search for images similar to a query image.
    
    Args:
        query_image_path: Path to query image
        index: Index instance
        top_k: Number of top results to return
        threshold: Minimum similarity score
        use_aggregation: Aggregate multi-vector scores
        
    Returns:
        List of (similarity_score, image_path) tuples
        
    Raises:
        ValidationError: If query_image_path is invalid
    """
    # Validate image path
    is_valid, error_msg = validate_image_path(query_image_path)
    if not is_valid:
        raise ValidationError(f"Invalid query image: {error_msg}")
    
    query = extract_image_features(query_image_path).reshape(1, -1)
    return _search(query, top_k, index, threshold=threshold, use_aggregation=use_aggregation)


def _text_should_be_prefixed(text: str) -> bool:
    """Determine if a text query should be prefixed with 'a photo of'.

    Returns True when the query looks like a short noun phrase rather
    than a full natural-language sentence:
      - Does not start with a known article/determiner prefix
      - Does not end with sentence punctuation (., !, ?)
      - Contains no verbs (heuristic: auxiliary/be verbs at word-2 position)
      - Is shorter than 6 words
    All conditions must hold to avoid mangling well-formed queries like
    "a beautiful sunset over the mountains" or "show me pictures of cats."
    """
    normalized = text.strip().lower()
    word_count = len(normalized.split())

    # Already has a clear article/determiner prefix
    prefix_words = ("a ", "an ", "the ", "my ", "this ", "that ",
                    "some ", "any ", "these ", "those ", "what ",
                    "which ", "whose ", "photo ", "picture ", "image ")
    if normalized.startswith(prefix_words):
        return False

    # Ends with sentence punctuation  -> natural sentence
    if normalized[-1:] in (".", "!", "?"):
        return False

    # Contains known verb/auxiliary at word-2  -> natural sentence
    verbs = {"is", "are", "was", "were", "be", "been", "being",
             "has", "have", "had", "do", "does", "did",
             "show", "find", "get", "give", "search", "look"}
    first_word = normalized.split()[0] if word_count >= 1 else ""
    if first_word in verbs:
        return False

    # Long multi-word queries are likely already well-formed
    if word_count >= 6:
        return False

    return True


def text_search(text: str, index: Index, top_k: int = 3,
                threshold: float = SIMILARITY_THRESHOLD,
                use_aggregation: bool = False) -> List[Tuple[float, str]]:
    """
    Search for images matching a text description.
    
    Args:
        text: Text query
        index: Index instance
        top_k: Number of top results to return
        threshold: Minimum similarity score
        use_aggregation: Aggregate multi-vector scores
        
    Returns:
        List of (similarity_score, image_path) tuples
        
    Raises:
        ValidationError: If text query is invalid
    """
    # Validate text query
    is_valid, error_msg = validate_text_query(text)
    if not is_valid:
        raise ValidationError(f"Invalid text query: {error_msg}")
    
    text = text.strip()
    
    # Only add prefix if query looks like a bare noun phrase
    if _text_should_be_prefixed(text):
        text = f"a photo of {text}"
    
    query = extract_text_features(text).reshape(1, -1)
    return _search(query, top_k, index, threshold=threshold, use_aggregation=use_aggregation)
