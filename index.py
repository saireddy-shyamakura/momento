import numpy as np
import chromadb
from typing import List, Tuple
from config import CHROMA_DB_DIR
from logger import get_logger

logger = get_logger(__name__)

class Index:
    """Manages ChromaDB vector indexing for fast image search with persistent storage."""
    
    def __init__(self, db_path: str = CHROMA_DB_DIR):
        """Initialize ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="images",
            metadata={"hnsw:space": "cosine"} # Cosine similarity space matches our previous Inner Product since vectors are normalized
        )
    
    def add_vectors(self, paths: List[str], vectors: List[np.ndarray]) -> None:
        """
        Add new vectors to ChromaDB collection.
        
        Args:
            paths: List of string file paths (used as IDs)
            vectors: List of new numpy arrays to add
        """
        if len(vectors) == 0:
            logger.warning("No new features to add")
            return
            
        if len(paths) != len(vectors):
            logger.error("Paths and vectors length mismatch")
            return
        
        # Convert numpy arrays to nested lists of floats for ChromaDB
        embeddings = [v.flatten().tolist() for v in vectors]
        
        logger.info(f"Adding {len(paths)} new vectors to ChromaDB")
        self.collection.upsert(
            embeddings=embeddings,
            ids=paths,
            metadatas=[{"path": path} for path in paths]
        )
        logger.info(f"ChromaDB updated: now has {self.collection.count()} total vectors")
    
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[float, str]]:
        """
        Search for top_k nearest neighbors.
        
        Args:
            query_vector: Query vector (shape: 1, dim) - should be normalized
            top_k: Number of top results to return
            
        Returns:
            List of (score, path) tuples sorted by descending score
        """
        if self.collection.count() == 0:
            logger.debug("Search attempted on empty index")
            return []
        
        # Clamp top_k to avoid requesting more results than exist
        effective_k = min(top_k, self.collection.count())
        if effective_k == 0:
            return []
        
        query_embedding = query_vector.flatten().tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k
        )
        
        out = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                # In cosine space, score = 1 - distance
                score = 1.0 - distance
                path = results["ids"][0][i]
                out.append((score, path))
                
        logger.debug(f"Search returned {len(out)} results")
        return out
    
    def is_built(self) -> bool:
        """Check if index has any items."""
        return self.collection.count() > 0
    
    def get_vector_count(self) -> int:
        """Get total number of vectors in index."""
        return self.collection.count()
    
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
        return set(result["ids"])

    def item_exists(self, path: str) -> bool:
        """Check if an image path already exists in index."""
        import os
        if not path:
            return False
        abs_path = os.path.abspath(path)
        result = self.collection.get(ids=[abs_path])
        return len(result["ids"]) > 0

    def delete_all(self) -> None:
        """Delete every entry from the collection."""
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)

    def get_all_paths(self) -> List[str]:
        """Return every stored path (used by --verify)."""
        result = self.collection.get()
        return result["ids"]