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
    
    # Only add prefix if user didn't already provide a natural sentence
    prefix_words = ("a ", "an ", "the ", "my ", "this ", "some ", "photo ", "picture ", "image ")
    if not text.lower().startswith(prefix_words):
        text = f"a photo of {text}"
    
    query = extract_text_features(text).reshape(1, -1)
    return _search(query, top_k, index, threshold=threshold, use_aggregation=use_aggregation)