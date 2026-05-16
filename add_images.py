import os
import logging
from features import extract_image_features_batch
from index import Index
from validation import validate_folder_path, validate_image_path
from config import SUPPORTED_EXTENSIONS
from logger import get_logger

logger = get_logger(__name__)

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
    skipped = 0
    
    logger.info(f"Scanning folder: {folder}")
    
    for file in os.listdir(folder):
        path = os.path.abspath(os.path.join(folder, file))

        # Skip non-image files
        if not path.lower().endswith(SUPPORTED_EXTENSIONS):
            continue

        # Skip already indexed files
        if index.item_exists(path):
            skipped += 1
            continue

        # Validate file before processing
        is_valid, error_msg = validate_image_path(path)
        if not is_valid:
            logger.warning(f"Skipping {file}: {error_msg}")
            failed += 1
            continue

        paths_to_process.append(path)

    if not paths_to_process:
        if skipped > 0:
            logger.info(f"No new images. Skipped {skipped} already-indexed files.")
        else:
            logger.info("No new images found")
        return 0

    logger.info(f"Found {len(paths_to_process)} new images to process.")
    
    # Process in batches
    successful_paths, new_features = extract_image_features_batch(paths_to_process, batch_size=batch_size)
    
    added = len(successful_paths)
    failed += (len(paths_to_process) - added)

    if added > 0:
        index.add_vectors(successful_paths, new_features)
        logger.info(f"Added {added} images, skipped {skipped}, failed {failed}")
    else:
        logger.warning(f"Failed to process any new images out of {len(paths_to_process)} attempted.")
        
    return added
