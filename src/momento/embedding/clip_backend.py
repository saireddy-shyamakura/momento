"""
clip_backend.py — CLIP embedding backend implementing EmbeddingBackend.

Wraps the existing openai/CLIP model behind the unified abstraction.
"""
from typing import List, Optional, Tuple
import numpy as np
import torch
from PIL import Image

from .base import EmbeddingBackend
from ..device import device_manager
from ..logger import get_logger

logger = get_logger(__name__)

_model: Optional[torch.nn.Module] = None
_preprocess: Optional[callable] = None
_current_model_name: Optional[str] = None


def _load_clip(model_name: str) -> Tuple[torch.nn.Module, callable]:
    """Load CLIP model (cached globally)."""
    global _model, _preprocess, _current_model_name
    import clip  # lazy import

    if _model is not None and _current_model_name == model_name:
        return _model, _preprocess

    dev = device_manager.device
    logger.info(f"Loading CLIP model ({model_name}) on device: {dev}")
    _model, _preprocess = clip.load(model_name, device=dev)
    _current_model_name = model_name
    return _model, _preprocess


class ClipBackend(EmbeddingBackend):
    """CLIP-based embedding backend.

    Supports all OpenAI CLIP model variants (ViT-B/32, ViT-B/16, etc.).
    """

    def __init__(self, model_name: str = "ViT-B/16"):
        self._model_name = model_name
        # Pre-load to get dimension
        model, _ = _load_clip(model_name)
        self._dim = model.visual.output_dim if hasattr(model, 'visual') else 512

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_image(self, image_path: str) -> np.ndarray:
        import os
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")
        return self._encode_pil(image, image_path)

    def embed_image_pil(self, image: Image.Image) -> np.ndarray:
        return self._encode_pil(image, "<pil>")

    def embed_text(self, text: str) -> np.ndarray:
        import clip  # lazy import

        if not text or not text.strip():
            raise ValueError("Text query cannot be empty")
        if len(text) > 1000:
            raise ValueError("Text query too long (max 1000 characters)")

        model, _ = _load_clip(self._model_name)
        dev = device_manager.device

        try:
            with torch.inference_mode():
                tokens = clip.tokenize([text]).to(dev)
                features = model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze(0).cpu().numpy().astype(np.float32)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                device_manager.fallback_to_cpu()
                raise RuntimeError("GPU OOM. Switched to CPU — retry.")
            raise RuntimeError(f"Failed to extract text features: {e}")

    def embed_images_batch(
        self, image_paths: List[str], batch_size: int = 32
    ) -> Tuple[List[str], List[np.ndarray]]:
        model, preprocess = _load_clip(self._model_name)
        dev = device_manager.device

        successful_paths: List[str] = []
        all_features: List[np.ndarray] = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images: List[Image.Image] = []
            current_batch_paths: List[str] = []

            for path in batch_paths:
                try:
                    with Image.open(path) as img:
                        images.append(img.convert("RGB"))
                        current_batch_paths.append(path)
                except Exception as e:
                    logger.error(f"Failed to load image {path}: {e}")

            if not images:
                continue

            try:
                with torch.inference_mode():
                    batch_tensor = torch.stack([preprocess(img) for img in images]).to(dev)
                    features = model.encode_image(batch_tensor)
                    features = features / features.norm(dim=-1, keepdim=True)
                    features_np = features.cpu().numpy().astype(np.float32)
                    for j in range(len(current_batch_paths)):
                        all_features.append(features_np[j])
                        successful_paths.append(current_batch_paths[j])
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"GPU OOM processing batch. Stopping early.")
                    device_manager.fallback_to_cpu()
                    break
                logger.error(f"Failed to extract features for batch: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing batch: {e}")

        return successful_paths, all_features

    def embed_pil_batch(
        self, images: List[Image.Image], batch_size: int = 32
    ) -> List[np.ndarray]:
        model, preprocess = _load_clip(self._model_name)
        dev = device_manager.device
        all_features: List[np.ndarray] = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            try:
                tensors = [preprocess(img) for img in batch]
                with torch.inference_mode():
                    stacked = torch.stack(tensors).to(dev)
                    feats = model.encode_image(stacked)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    all_features.extend(feats.cpu().numpy().astype(np.float32))
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    device_manager.fallback_to_cpu()
                    break
                logger.error(f"Batch encode error: {e}")

        return all_features

    def clear_cache(self) -> None:
        global _model, _preprocess, _current_model_name
        if _model is not None:
            try:
                _model = _model.to("cpu")
            except Exception:
                pass
        _model = None
        _preprocess = None
        _current_model_name = None

        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("CLIP model cache cleared")

    def _encode_pil(self, image: Image.Image, source_label: str = "<pil>") -> np.ndarray:
        model, preprocess = _load_clip(self._model_name)
        dev = device_manager.device

        try:
            with torch.inference_mode():
                preprocessed = preprocess(image).unsqueeze(0).to(dev)
                features = model.encode_image(preprocessed)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze(0).cpu().numpy().astype(np.float32)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                device_manager.fallback_to_cpu()
                raise RuntimeError(f"GPU OOM processing {source_label}. Switched to CPU — retry.")
            raise RuntimeError(f"Failed to extract features from {source_label}: {e}")