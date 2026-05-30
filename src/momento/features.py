import torch
import clip
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
from .config import DEVICE, MODEL_NAME, OCR_MIN_TEXT_LENGTH, COMPOSITE_SEP
from .device import device_manager
from .logger import get_logger

logger = get_logger(__name__)

_model: Optional[torch.nn.Module] = None
_preprocess: Optional[callable] = None
_current_model_name: Optional[str] = None


def get_model(model_name: Optional[str] = None) -> Tuple[torch.nn.Module, callable]:
    """
    Get or load the CLIP model and preprocessing function.
    
    Model is cached globally to avoid reloading on each use.
    If model_name differs from cached model, reloads the new model.
    
    Args:
        model_name: Name of CLIP model to load. If None, uses default MODEL_NAME.
                   Supported: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px, ConvNeXt-B
    
    Returns:
        Tuple of (model, preprocess_function)
    """
    global _model, _preprocess, _current_model_name
    
    if model_name is None:
        model_name = MODEL_NAME
    
    # Return cached model if it matches the requested model
    if _model is not None and _current_model_name == model_name:
        return _model, _preprocess
    
    dev = device_manager.device
    logger.info(f"Loading CLIP model ({model_name}) on device: {dev}")
    _model, _preprocess = clip.load(model_name, device=dev)
    _current_model_name = model_name
    return _model, _preprocess


def clear_model_cache() -> None:
    """Clear the cached model to free GPU/CPU memory.

    Call this after indexing completes if no search queries are
    expected immediately.  The model will be lazily reloaded on
    the next call to ``get_model()``.
    """
    global _model, _preprocess, _current_model_name

    if _model is not None:
        dev = _model
        # Move model to CPU before releasing reference to help PyTorch
        # reclaim GPU memory more eagerly
        try:
            _model = _model.to("cpu")
        except Exception:
            pass

    _model = None
    _preprocess = None
    _current_model_name = None

    import gc
    gc.collect()

    # Try to clear CUDA cache (no-op if CUDA is not available)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    logger.info("Model cache cleared — GPU memory released")


def extract_image_features(image_path: str, model_name: Optional[str] = None) -> np.ndarray:
    """
    Extract normalized feature vector from an image.
    
    Handles various image formats and validates file before processing.
    
    Args:
        image_path: Path to image file
        model_name: CLIP model to use. If None, uses default from config.
        
    Returns:
        Normalized feature vector as numpy array (float32)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        RuntimeError: If image cannot be loaded or processed
    """
    import os
    
    # Validate file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not os.path.isfile(image_path):
        raise RuntimeError(f"Path is not a file: {image_path}")
    
    if not os.access(image_path, os.R_OK):
        raise RuntimeError(f"Image file is not readable: {image_path}")
    
    model, preprocess = get_model(model_name)

    try:
        # Try to open and convert image to RGB
        with Image.open(image_path) as img:
            # Convert to RGB to handle RGBA, grayscale, etc.
            image = img.convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except IOError as e:
        raise RuntimeError(f"Cannot read image file {image_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {type(e).__name__}: {e}")

    return _encode_pil_image(image, image_path, model_name)


def _encode_pil_image(image: Image.Image, source_label: str = "<pil>", model_name: Optional[str] = None) -> np.ndarray:
    """Encode a single PIL Image to a normalized CLIP embedding."""
    model, preprocess = get_model(model_name)
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


def extract_image_features_batch(image_paths: List[str], batch_size: int = 32) -> Tuple[List[str], List[np.ndarray]]:
    """
    Extract normalized feature vectors from a batch of images.
    
    Args:
        image_paths: List of paths to image files
        batch_size: Number of images to process at once
        
    Returns:
        Tuple of (successful_paths, list_of_feature_vectors)
    """
    model, preprocess = get_model()
    dev = device_manager.device
    
    successful_paths = []
    all_features = []
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, len(image_paths), batch_size), 1):
        batch_paths = image_paths[i:i+batch_size]
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_paths)} images)...")
        images = []
        current_batch_paths = []
        
        for path in batch_paths:
            try:
                with Image.open(path) as img:
                    image = img.convert("RGB")
                    images.append(preprocess(image))
                    current_batch_paths.append(path)
            except Exception as e:
                logger.error(f"Failed to load/preprocess image {path}: {e}")
                
        if not images:
            continue
            
        try:
            with torch.inference_mode():
                batch_tensor = torch.stack(images).to(dev)
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
            else:
                logger.error(f"Failed to extract features for batch: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing batch: {e}")
            
    return successful_paths, all_features


def extract_pil_features_batch(
    images: List[Image.Image], batch_size: int = 32
) -> List[np.ndarray]:
    """Encode a list of PIL Images in batches. Returns list of embeddings."""
    model, preprocess = get_model()
    dev = device_manager.device
    all_features: List[np.ndarray] = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        tensors = [preprocess(img) for img in batch]
        try:
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


def extract_text_features(text: str, model_name: Optional[str] = None) -> np.ndarray:
    """
    Extract normalized feature vector from text.
    
    Validates text before processing.
    
    Args:
        text: Text description
        model_name: CLIP model to use. If None, uses default from config.
        
    Returns:
        Normalized feature vector as numpy array (float32)
        
    Raises:
        ValueError: If text is empty or invalid
        RuntimeError: If feature extraction fails
    """
    # Validate text
    if not text or not text.strip():
        raise ValueError("Text query cannot be empty")
    
    if len(text) > 1000:
        raise ValueError("Text query too long (max 1000 characters)")
    
    model, _ = get_model(model_name)
    dev = device_manager.device

    try:
        with torch.inference_mode():
            # Tokenize and move to device
            tokens = clip.tokenize([text]).to(dev)
            
            # Extract features and normalize
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy().astype(np.float32)
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            device_manager.fallback_to_cpu()
            raise RuntimeError("GPU OOM. Switched to CPU — retry.")
        raise RuntimeError(f"Failed to extract text features: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error extracting text features: {type(e).__name__}: {e}")


# ── Multi-embedding helpers ───────────────────────────────────────────

def extract_multi_embeddings(image_path: str, model_name: Optional[str] = None) -> List[Tuple[str, np.ndarray]]:
    """
    Generate embeddings for the original image + augmented views.

    Args:
        image_path: Path to the image file
        model_name: CLIP model to use. If None, uses default from config.

    Returns:
        List of (suffix, embedding) tuples.  suffix is 'orig', 'flip', etc.
    """
    from .augment import generate_augmentations

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")

    # Original embedding
    orig_emb = _encode_pil_image(img_rgb, image_path, model_name)
    results: List[Tuple[str, np.ndarray]] = [("orig", orig_emb)]

    # Augmented views
    views = generate_augmentations(img_rgb)
    for suffix, aug_img in views:
        try:
            emb = _encode_pil_image(aug_img, f"{image_path}{COMPOSITE_SEP}{suffix}", model_name)
            results.append((suffix, emb))
        except Exception as e:
            logger.warning(f"Augmentation embed failed ({suffix}): {e}")

    return results


# ── YOLO + CLIP object embeddings ─────────────────────────────────────

def extract_object_embeddings(image_path: str) -> List[Tuple[dict, np.ndarray]]:
    """
    Run YOLO on image, CLIP-encode each detected object crop.

    Returns:
        List of (metadata_dict, embedding) tuples.
    """
    from .yolo import detect_objects, is_available as yolo_available
    if not yolo_available():
        logger.warning("YOLO not available — skipping object detection")
        return []

    detections = detect_objects(image_path)
    results: List[Tuple[dict, np.ndarray]] = []

    for det in detections:
        try:
            emb = _encode_pil_image(det.cropped_image, f"{image_path}{COMPOSITE_SEP}yolo_{det.label}")
            results.append((det.to_metadata(), emb))
        except Exception as e:
            logger.warning(f"Object embed failed ({det.label}): {e}")

    return results


# ── OCR + CLIP text embedding ─────────────────────────────────────────

def extract_ocr_embedding(image_path: str) -> Optional[Tuple[str, np.ndarray]]:
    """
    Run OCR on image, encode extracted text via CLIP text encoder.

    Returns:
        (extracted_text, embedding) or None if no text found.
    """
    from .ocr import extract_text as ocr_extract, is_available as ocr_available
    if not ocr_available():
        logger.warning("EasyOCR not available — skipping OCR")
        return None

    text = ocr_extract(image_path)
    if not text or len(text.strip()) < OCR_MIN_TEXT_LENGTH:
        return None

    try:
        emb = extract_text_features(text)
        return (text, emb)
    except Exception as e:
        logger.warning(f"OCR embedding failed: {e}")
        return None