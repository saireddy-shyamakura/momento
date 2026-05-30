"""
router.py — Query router for Momento V3.

Classifies a query into a search type and routes to the correct pipeline:
- Semantic: vector search via CLIP
- Exact: SQLite FTS exact lookup
- Hybrid: both + fusion
"""
from enum import Enum
from typing import Optional


class QueryType(Enum):
    """Supported query types."""
    SEMANTIC = "semantic"   # Pure vector search
    EXACT = "exact"         # Exact filename / path lookup
    HYBRID = "hybrid"       # Both vector + exact with fusion


class QueryRouter:
    """Rule-based query router.

    Routes queries based on heuristics:
    - Queries containing file extensions (.jpg, .png) → EXACT
    - Queries that look like file paths (start with / or ~) → EXACT
    - Short keyword queries (1-2 words) → HYBRID
    - Natural language queries → SEMANTIC
    """

    def classify(self, query: str) -> QueryType:
        """Classify a query into a search type.

        Args:
            query: The raw user query string.

        Returns:
            The appropriate QueryType.
        """
        q = query.strip()

        # File extension → exact lookup
        if any(q.lower().endswith(ext) for ext in
               ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif',
                '.mp4', '.avi', '.mov', '.mkv')):
            return QueryType.EXACT

        # Looks like a file path → exact lookup
        if q.startswith('/') or q.startswith('~') or q.startswith('./') or q.startswith('../'):
            return QueryType.EXACT

        # Short keyword queries (1-2 words) → hybrid
        word_count = len(q.split())
        if word_count <= 2:
            return QueryType.HYBRID

        # Default: semantic search
        return QueryType.SEMANTIC

    def route(self, query: str) -> str:
        """Return a human-readable route description."""
        qtype = self.classify(query)
        routes = {
            QueryType.SEMANTIC: "vector_search",
            QueryType.EXACT: "exact_lookup",
            QueryType.HYBRID: "hybrid_search",
        }
        return routes[qtype]