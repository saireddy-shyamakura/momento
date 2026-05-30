"""
storage — Storage separation for Momento V3.

Separates concerns across storage backends:
- Vectors: ChromaDB
- Metadata: SQLite
- Exact index: SQLite FTS5
- Cache: disk
"""
from .metadata_store import MetadataStore

__all__ = ["MetadataStore"]