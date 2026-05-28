"""
index_utils.py — Utility functions for index management.

Centralizes common index operations like initialization, verification,
and statistics retrieval.
"""

import os
from typing import Dict, Any

from .index import Index
from .config import CHROMA_DB_DIR, COMPOSITE_SEP
from .logger import get_logger

logger = get_logger(__name__)


def get_or_create_index() -> Index:
    """Get or create the default index.
    
    Returns:
        Index instance
    """
    logger.info("Initializing index...")
    try:
        index = Index()
        vector_count = index.get_vector_count()
        logger.info(f"Index loaded: {vector_count} vectors")
        return index
    except Exception as e:
        logger.error(f"Failed to initialize index: {e}")
        raise


def verify_index(index: Index) -> Dict[str, Any]:
    """Verify index and remove stale entries.
    
    Removes entries whose source files no longer exist on disk.
    
    Args:
        index: Index instance to verify
        
    Returns:
        Dict with verification results
    """
    logger.info("Verifying index...")
    
    try:
        paths = index.get_all_paths()
        total_entries = len(paths)
        
        stale = []
        for path in paths:
            # Handle composite IDs (e.g., "path|||suffix")
            base_path = path.split(COMPOSITE_SEP)[0]
            if not os.path.exists(base_path):
                stale.append(path)
        
        stale_count = len(stale)
        
        if stale:
            index.delete_paths(stale)
            logger.info(f"Removed {stale_count} stale entries")
        else:
            logger.info("Index is clean — no stale entries found")
        
        return {
            'total_entries': total_entries,
            'stale_count': stale_count,
            'is_clean': stale_count == 0,
        }
    
    except Exception as e:
        logger.error(f"Index verification failed: {e}")
        raise


def get_index_stats(index: Index) -> Dict[str, Any]:
    """Get detailed index statistics.
    
    Args:
        index: Index instance
        
    Returns:
        Dict with index stats
    """
    try:
        vector_count = index.get_vector_count()
        all_paths = index.get_all_paths()
        
        # Count by type (estimate based on suffixes)
        image_count = len([p for p in all_paths if COMPOSITE_SEP not in p or f'{COMPOSITE_SEP}orig' in p])
        video_count = len([p for p in all_paths if f'{COMPOSITE_SEP}frame_' in p])
        object_count = len([p for p in all_paths if f'{COMPOSITE_SEP}yolo_' in p])
        ocr_count = len([p for p in all_paths if f'{COMPOSITE_SEP}ocr' in p])
        
        return {
            'total_vectors': vector_count,
            'total_entries': len(all_paths),
            'estimated_images': image_count,
            'estimated_videos': video_count,
            'estimated_objects': object_count,
            'estimated_ocr': ocr_count,
            'db_path': CHROMA_DB_DIR,
        }
    
    except Exception as e:
        logger.error(f"Failed to get index stats: {e}")
        raise


def reset_index(index: Index) -> bool:
    """Reset (clear) the entire index.
    
    Args:
        index: Index instance to reset
        
    Returns:
        True if successful
    """
    try:
        logger.info("Resetting index...")
        index.delete_all()
        logger.info("Index reset complete")
        return True
    except Exception as e:
        logger.error(f"Failed to reset index: {e}")
        raise


def get_index_health(index: Index) -> Dict[str, Any]:
    """Get index health summary.
    
    Args:
        index: Index instance
        
    Returns:
        Dict with health information
    """
    try:
        stats = get_index_stats(index)
        verify = verify_index(index)
        
        return {
            'status': 'healthy' if verify['is_clean'] else 'warning',
            'vectors': stats['total_vectors'],
            'entries': stats['total_entries'],
            'stale_count': verify['stale_count'],
            'db_location': CHROMA_DB_DIR,
        }
    
    except Exception as e:
        logger.error(f"Failed to get index health: {e}")
        return {
            'status': 'error',
            'error': str(e),
        }
