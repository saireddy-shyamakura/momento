import os
from .features import extract_image_features_batch
from .cache import load_embedding, save_embedding
import numpy as np
from .index import Index
from .validation import validate_folder_path, validate_image_path, is_path_safe
from .config import SUPPORTED_EXTENSIONS
from .logger import get_logger

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

logger = get_logger(__name__)

def _make_progress(total: int, desc: str):
    if _HAS_TQDM:
        return tqdm(total=total, desc=desc)
    return None

def add_images(folder: str, index: Index, batch_size: int = 32) -> int:
    """
    Add images from a folder to the index using batch processing.
    
    Validates folder exists and is readable. Skips invalid files with logging.
    Batches feature extraction and index updates to minimize index rebuilds.
    
    Args:
        folder: Path to folder containing images
        index: Index instance
        batch_size: Number of images to process at once
        
    Returns:
        Number of images successfully added
    """
    # Validate folder path
    is_valid, error_msg = validate_folder_path(folder)
    if not is_valid:
        logger.error(f"Cannot add images: {error_msg}")
        return 0

    paths_to_process = []
    failed = 0
    
    logger.info(f"Scanning folder (recursive): {folder}")
    
    # Collect all candidate image paths recursively
    candidate_paths = []
    for root, dirs, files in os.walk(folder, followlinks=False):
        for d in dirs:
            full = os.path.join(root, d)
            if os.path.islink(full):
                logger.debug(f"Skipping symlink directory: {full}")
        for file in files:
            path = os.path.abspath(os.path.join(root, file))

            # Skip non-image files
            if not path.lower().endswith(SUPPORTED_EXTENSIONS):
                continue

            # Path traversal guard: reject symlinks that escape base folder
            is_safe, safety_msg = is_path_safe(path, folder)
            if not is_safe:
                logger.warning(f"Skipping {file}: {safety_msg}")
                failed += 1
                continue

            # Validate file before processing
            is_valid, error_msg = validate_image_path(path)
            if not is_valid:
                logger.warning(f"Skipping {file}: {error_msg}")
                failed += 1
                continue

            candidate_paths.append(path)

    # Bulk check which paths are already indexed (single DB query)
    existing_ids = index.get_existing_ids(candidate_paths)
    skipped = len(existing_ids)
    if skipped > 0:
        logger.info(f"Resuming: {skipped} images already indexed.")
    
    paths_to_process = [p for p in candidate_paths if p not in existing_ids]

    if not paths_to_process:
        if skipped > 0:
            logger.info(f"No new images. Skipped {skipped} already-indexed files.")
        else:
            logger.info("No new images found")
        print(f"Done: 0 added, {skipped} skipped, {failed} failed")
        return 0

    total = len(paths_to_process)
    logger.info(f"Found {total} new images to process.")
    
    # Process in batches
    total_batches = (total + batch_size - 1) // batch_size
    progress = _make_progress(total_batches, "Indexing images")
    
    successful_paths = []
    new_features = []
    processed = 0
    
    for i in range(0, total, batch_size):
        batch = paths_to_process[i:i+batch_size]

        # Use cache when available to avoid recomputing embeddings
        cached_features = {}
        uncached = []
        for p in batch:
            emb = load_embedding(p)
            if emb is not None:
                cached_features[p] = emb
            else:
                uncached.append(p)

        # Extract features for uncached images
        if uncached:
            batch_paths, batch_features = extract_image_features_batch(uncached, batch_size=len(uncached))
            # Save extracted embeddings to cache
            for bp, bf in zip(batch_paths, batch_features):
                try:
                    save_embedding(bp, np.asarray(bf))
                except Exception:
                    pass
                successful_paths.append(bp)
                new_features.append(bf)

        # Add cached ones to results
        for p, emb in cached_features.items():
            successful_paths.append(p)
            new_features.append(np.asarray(emb))

        processed += len(batch)
        if progress:
            progress.update(1)
        else:
            print(f"Processing: {processed}/{total} images...")
            
    if progress:
        progress.close()
    
    added = len(successful_paths)
    failed += (total - added)

    if added > 0:
        metadatas = [{"path": p, "source_path": p, "type": "image"} for p in successful_paths]
        index.add_vectors(successful_paths, new_features, metadatas)
        logger.info(f"Added {added} images, skipped {skipped}, failed {failed}")
    else:
        logger.warning(f"Failed to process any new images out of {total} attempted.")
        
    print(f"Done: {added} added, {skipped} skipped, {failed} failed")
    return added
