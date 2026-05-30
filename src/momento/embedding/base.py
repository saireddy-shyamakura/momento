"""
base.py — Abstract embedding backend interface.

All embedding backends (CLIP, SigLIP, future) must implement this interface.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image


class EmbeddingBackend(ABC):
    """Abstract interface for embedding models."""

    @abstractmethod
    def embed_image(self, image_path: str) -> np.ndarray:
        """Embed a single image from file path.
        
        Returns:
            Normalized feature vector (float32).
        """
        ...

    @abstractmethod
    def embed_image_pil(self, image: Image.Image) -> np.ndarray:
        """Embed a single PIL Image.
        
        Returns:
            Normalized feature vector (float32).
        """
        ...

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query.
        
        Returns:
            Normalized feature vector (float32).
        """
        ...

    @abstractmethod
    def embed_images_batch(
        self, image_paths: List[str], batch_size: int = 32
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Embed a batch of images from file paths.
        
        Returns:
            (successful_paths, embeddings) tuple.
        """
        ...

    @abstractmethod
    def embed_pil_batch(
        self, images: List[Image.Image], batch_size: int = 32
    ) -> List[np.ndarray]:
        """Embed a batch of PIL Images.
        
        Returns:
            List of normalized feature vectors.
        """
        ...

    @abstractmethod
    def clear_cache(self) -> None:
        """Release model from memory (e.g. after indexing completes)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this backend (e.g. 'ViT-B/16')."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...