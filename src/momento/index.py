"""
index.py — ChromaDB vector store wrapper for Momento.

Provides persistent vector storage with:
- Cosine similarity search (hnsw:space=cosine)
- Score aggregation across composite IDs (path|||suffix)
- Automatic retry on transient ChromaDB errors
- Version metadata tracking for schema migration detection
"""

import os
import numpy as np
import chromadb
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import CHROMA_DB_DIR, COMPOSITE_SEP
from .logger import get_logger

logger = get_logger(__name__)

# Current Momento and ChromaDB versions stored as collection metadata
MOMENTO_VERSION = "3.0.0"

# ChromaDB transient errors that warrant a retry
_RETRYABLE_EXCEPTIONS = (
    chromadb.errors.ChromaError,
    ConnectionError,
    TimeoutError,
)


def _retry_chroma(func):
    """Decorator that retries ChromaDB operations on transient failures.

    Uses exponential backoff: 1s → 2s → 4s between retries.
    """
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying ChromaDB operation after error: "
            f"{retry_state.outcome.exception()}"
        ),
    )(func)


class Index:
    """Manages ChromaDB vector indexing for fast image search with persistent storage."""
    
    def __init__(
        self,
        db_path: str = CHROMA_DB_DIR,
        hnsw_construction_ef: int = 200,
        hnsw_search_ef: int = 200,
    ):
        """Initialize ChromaDB client and collection.

        Stores version metadata in the collection to detect schema
        changes across upgrades.

        On corrupt DB, attempts auto-repair by deleting and recreating
        the collection. If repair fails, raises RuntimeError.

        Args:
            db_path: Filesystem path for ChromaDB persistent storage.
            hnsw_construction_ef: HNSW construction_ef parameter for
                index quality vs build time trade-off (default: 200).
            hnsw_search_ef: HNSW search_ef parameter for search
                accuracy vs speed trade-off (default: 200).

        Raises:
            RuntimeError: If ChromaDB fails to initialize (e.g., corrupt DB).
        """
        self.db_path = db_path
        self.hnsw_construction_ef = hnsw_construction_ef
        self.hnsw_search_ef = hnsw_search_ef
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name="images",
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": hnsw_construction_ef,
                    "hnsw:search_ef": hnsw_search_ef,
                }
            )
            # Store version metadata
            self._set_version_metadata()
            self._check_version_mismatch()
        except Exception as e:
            # Attempt auto-repair: delete and recreate collection
            logger.warning(f"ChromaDB initialization failed: {e}")
            logger.info("Attempting auto-repair by recreating the collection...")
            try:
                repair_client = chromadb.PersistentClient(path=db_path)
                try:
                    repair_client.delete_collection("images")
                except Exception:
                    pass
                self.collection = repair_client.create_collection(
                    name="images",
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": hnsw_construction_ef,
                        "hnsw:search_ef": hnsw_search_ef,
                    }
                )
                self.client = repair_client
                self._set_version_metadata()
                logger.info("ChromaDB auto-repair succeeded — collection recreated.")
            except Exception as repair_error:
                raise RuntimeError(
                    f"Failed to initialize ChromaDB index at {db_path}. "
                    f"This may indicate a corrupted database. "
                    f"Run `momento --reset` to rebuild.\n"
                    f"Technical details (initial): {e}\n"
                    f"Technical details (repair): {repair_error}"
                ) from e

    def close(self) -> None:
        """Cleanly close the ChromaDB client and release resources.

        Should be called on graceful shutdown to ensure all pending
        writes are flushed and the WAL is checkpointed.
        """
        try:
            self.client.close()
            logger.debug("ChromaDB client closed cleanly.")
        except Exception as e:
            logger.debug(f"Error closing ChromaDB client (non-critical): {e}")

    def _set_version_metadata(self) -> None:
        """Store current version metadata in the collection.

        This enables detecting when a user upgrades Momento with a
        potentially-incompatible ChromaDB schema.
        """
        try:
            from chromadb import __version__ as chromadb_version
        except ImportError:
            chromadb_version = "unknown"

        try:
            self.collection.modify(
                metadata={
                    "momento_version": MOMENTO_VERSION,
                    "chromadb_version": chromadb_version,
                }
            )
        except Exception as e:
            logger.debug(f"Could not set version metadata (non-critical): {e}")

    def _check_version_mismatch(self) -> None:
        """Check stored vs current ChromaDB version and warn on mismatch.

        A version mismatch can cause silent data loss if the on-disk
        schema format changed between ChromaDB releases.  This issues
        a warning so the user knows to rebuild before it breaks.
        """
        try:
            from chromadb import __version__ as current_cv
        except ImportError:
            return

        try:
            meta = self.collection.metadata
            if meta:
                stored_cv = meta.get("chromadb_version", "")
                stored_mv = meta.get("momento_version", "")
                if stored_cv and stored_cv != current_cv:
                    logger.warning(
                        f"ChromaDB version mismatch: index was built with "
                        f"v{stored_cv} but current version is v{current_cv}. "
                        f"Run `momento --reset` if you experience errors."
                    )
                    print(
                        f"⚠️  ChromaDB version mismatch: index built with "
                        f"v{stored_cv}, currently running v{current_cv}. "
                        f"Run `momento --reset` if you experience issues."
                    )
                if stored_mv and stored_mv != MOMENTO_VERSION:
                    logger.warning(
                        f"Momento version mismatch: index was built with "
                        f"v{stored_mv} but current version is v{MOMENTO_VERSION}."
                    )
        except Exception as e:
            logger.debug(f"Version check skipped (non-critical): {e}")
    
    @_retry_chroma
    def add_vectors(self, paths: List[str], vectors: List[np.ndarray],
                    metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add new vectors to ChromaDB collection.
        
        Args:
            paths: List of string IDs (file paths or composite IDs like 'path::suffix')
            vectors: List of numpy arrays to add
            metadatas: Optional list of metadata dicts per vector
        """
        if len(vectors) == 0:
            logger.warning("No new features to add")
            return
            
        if len(paths) != len(vectors):
            logger.error("Paths and vectors length mismatch")
            return
        
        # Convert numpy arrays to nested lists of floats for ChromaDB
        embeddings = [v.flatten().tolist() for v in vectors]
        
        if metadatas is None:
            metadatas = [{"path": p, "type": "image"} for p in paths]
        
        logger.info(f"Adding {len(paths)} new vectors to ChromaDB")
        self.collection.upsert(
            embeddings=embeddings,
            ids=paths,
            metadatas=metadatas
        )
        logger.info(f"ChromaDB updated: now has {self.collection.count()} total vectors")
    
    @_retry_chroma
    def search(self, query_vector: np.ndarray, top_k: int = 3,
               where: Optional[Dict] = None) -> List[Tuple[float, str]]:
        """
        Search for top_k nearest neighbors.
        
        Args:
            query_vector: Query vector (shape: 1, dim) - should be normalized
            top_k: Number of top results to return
            where: Optional ChromaDB where filter
            
        Returns:
            List of (score, id) tuples sorted by descending score
        """
        results_with_meta = self.search_with_metadata(query_vector, top_k, where)
        return [(score, entry_id) for score, entry_id, _ in results_with_meta]

    @_retry_chroma
    def search_with_metadata(self, query_vector: np.ndarray, top_k: int = 3,
                            where: Optional[Dict] = None) -> List[Tuple[float, str, Dict]]:
        """
        Search for top_k nearest neighbors returning score, ID, and metadata.
        """
        if self.collection.count() == 0:
            logger.debug("Search attempted on empty index")
            return []
        
        effective_k = min(top_k, self.collection.count())
        if effective_k == 0:
            return []
        
        query_embedding = query_vector.flatten().tolist()
        
        kwargs = {"query_embeddings": [query_embedding], "n_results": effective_k}
        if where:
            kwargs["where"] = where
        
        results = self.collection.query(**kwargs)
        
        out = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                score = 1.0 - distance
                entry_id = results["ids"][0][i]
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                out.append((score, entry_id, meta))
                
        logger.debug(f"Search returned {len(out)} results")
        return out

    @_retry_chroma
    def _extract_source_path(self, entry_id: str, meta: dict) -> str:
        """Extract the source file path from a composite ID.

        Composite IDs follow the pattern ``path {COMPOSITE_SEP} suffix``.
        Uses ``rsplit(COMPOSITE_SEP, 1)`` so that any separators in the
        path itself (extremely unlikely) would be preserved as part of
        the source path.

        Falls back to ``source_path`` metadata if available, then to the
        full ID without suffix, then to the full ID itself.
        """
        source = meta.get("source_path")
        if source:
            return source
        parts = entry_id.rsplit(COMPOSITE_SEP, 1)
        if len(parts) == 2:
            return parts[0]
        return entry_id

    def search_aggregated(self, query_vector: np.ndarray, top_k: int = 3,
                          raw_k_multiplier: int = 5) -> List[Tuple[float, str]]:
        """
        Search with score aggregation across composite IDs.

        Multiple vectors for the same source file (e.g. augmentations,
        YOLO crops) are aggregated by taking the **max** score per
        source path.

        Returns:
            List of (best_score, source_path) sorted descending.
        """
        raw_k = min(top_k * raw_k_multiplier, self.collection.count())
        if raw_k == 0:
            return []

        raw_results = self.search_with_metadata(query_vector, top_k=raw_k)

        # Aggregate: group by source path (strip suffix)
        path_scores: Dict[str, float] = defaultdict(float)
        for score, entry_id, meta in raw_results:
            source = self._extract_source_path(entry_id, meta)
            path_scores[source] = max(path_scores[source], score)

        sorted_results = sorted(path_scores.items(), key=lambda x: x[1], reverse=True)
        return [(score, path) for path, score in sorted_results[:top_k]]

    @staticmethod
    def _flatten_id_list(raw_ids) -> List[str]:
        flattened: List[str] = []
        if raw_ids is None:
            return flattened

        for entry in raw_ids:
            if isinstance(entry, list):
                flattened.extend(x for x in entry if x is not None)
            elif entry is not None:
                flattened.append(entry)

        return flattened

    def is_built(self) -> bool:
        """Check if index has any items."""
        return self.collection.count() > 0
    
    def get_vector_count(self) -> int:
        """Get total number of vectors in index."""
        return self.collection.count()
    
    @_retry_chroma
    def get_existing_ids(self, paths: List[str]) -> set:
        """
        Bulk check which paths already exist in the index.
        
        Args:
            paths: List of absolute file paths to check
            
        Returns:
            Set of paths that already exist in the collection
        """
        if not paths:
            return set()
        result = self.collection.get(ids=paths)
        raw_ids = result.get("ids")
        return set(self._flatten_id_list(raw_ids))

    @_retry_chroma
    def item_exists(self, path: str) -> bool:
        """Check if an image path already exists in index."""
        if not path:
            return False
        abs_path = os.path.abspath(path)
        result = self.collection.get(ids=[abs_path])
        raw_ids = result.get("ids")
        return bool(self._flatten_id_list(raw_ids))

    @_retry_chroma
    def delete_all(self) -> None:
        """Delete every entry from the collection."""
        raw_ids = self.collection.get().get("ids")
        ids = self._flatten_id_list(raw_ids)
        if ids:
            self.collection.delete(ids=ids)

    @_retry_chroma
    def delete_paths(self, paths: List[str]) -> int:
        """
        Remove specific paths from the index.

        Args:
            paths: List of file paths to delete (must match stored IDs exactly).

        Returns:
            Number of entries removed.
        """
        if not paths:
            return 0
        self.collection.delete(ids=paths)
        return len(paths)

    def optimize(self) -> None:
        """Optimize the ChromaDB index for read performance.

        Triggers a compaction of the on-disk layout after large indexing
        operations to improve search latency. This is a hint to ChromaDB
        and may be a no-op in some configurations.

        Note: ChromaDB auto-compacts during normal operations — this
        call is primarily informational.  The expensive metadata-write
        hack was removed (see audit P6).
        """
        count = self.get_vector_count()
        logger.info(f"Optimization requested — index has {count} vectors")

    def export_all_data(self) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
        """Export all ids, embeddings, and metadata from the collection.

        Returns:
            Tuple of (ids, embeddings, metadatas) for all vectors.
        """
        try:
            result = self.collection.get()
            ids = result.get("ids", [])
            embeddings = result.get("embeddings", [])
            metadatas = result.get("metadatas", [])
            return (
                self._flatten_id_list(ids),
                [emb for emb in embeddings] if embeddings else [],
                [meta for meta in metadatas] if metadatas else [],
            )
        except Exception as e:
            logger.error(f"Failed to export index data: {e}")
            return [], [], []

    @_retry_chroma
    def import_all_data(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """Import data into the collection.

        Args:
            ids: List of vector IDs.
            embeddings: List of embedding vectors.
            metadatas: List of metadata dicts.

        Returns:
            Number of vectors imported.
        """
        if not ids or not embeddings:
            logger.warning("No data to import")
            return 0

        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(f"Imported {len(ids)} vectors into index")
            self.optimize()
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to import data: {e}")
            return 0

    @_retry_chroma
    def get_all_paths(self) -> List[str]:
        """Return every stored path (used by --verify)."""
        result = self.collection.get()
        return result["ids"]
