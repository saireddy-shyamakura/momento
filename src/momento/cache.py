"""
cache.py — Per-file embedding cache with LRU eviction.

Stores embeddings as NumPy .npz files keyed by a safe filename derived
from the absolute path. Each cache entry stores the file mtime to
invalidate cache when the source file changes.

When the cache exceeds a maximum size, the least recently accessed
entries are evicted.
"""

import os
import time
import hashlib
import numpy as np
from typing import Optional

from .config import EMBEDDING_CACHE_DIR
from .logger import get_logger

logger = get_logger(__name__)

# Maximum cache size in bytes (default: 5 GB)
CACHE_MAX_SIZE_BYTES = 5 * 1024 * 1024 * 1024

# Fraction of entries to evict when cache is full (oldest 20%)
_EVICT_FRACTION = 0.20


def _ensure_cache_dir():
    try:
        os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create cache dir {EMBEDDING_CACHE_DIR}: {e}")


def _path_to_key(path: str) -> str:
    """Deterministic short filename for a given absolute path."""
    h = hashlib.sha1(path.encode('utf-8')).hexdigest()
    return f"{h}.npz"


def _access_file_for(path: str) -> str:
    """Path to the companion .access timestamp file for a cache entry."""
    return _cache_file_for(path) + ".access"


def _cache_file_for(path: str) -> str:
    _ensure_cache_dir()
    key = _path_to_key(os.path.abspath(path))
    return os.path.join(EMBEDDING_CACHE_DIR, key)


def _get_cache_size() -> int:
    """Get total size of all cached .npz files in bytes."""
    total = 0
    try:
        for fname in os.listdir(EMBEDDING_CACHE_DIR):
            if fname.endswith('.npz'):
                fpath = os.path.join(EMBEDDING_CACHE_DIR, fname)
                try:
                    total += os.path.getsize(fpath)
                except OSError:
                    pass
    except FileNotFoundError:
        pass
    return total


def _evict_if_needed() -> None:
    """Evict oldest entries if cache exceeds maximum size."""
    try:
        current_size = _get_cache_size()
        if current_size <= CACHE_MAX_SIZE_BYTES:
            return

        # Collect all .npz files with their access times
        entries = []
        for fname in os.listdir(EMBEDDING_CACHE_DIR):
            if not fname.endswith('.npz'):
                continue
            fpath = os.path.join(EMBEDDING_CACHE_DIR, fname)
            access_path = fpath + ".access"
            try:
                if os.path.exists(access_path):
                    with open(access_path, 'r') as f:
                        atime = float(f.read().strip())
                else:
                    atime = os.path.getatime(fpath)
                entries.append((atime, fpath, access_path))
            except (OSError, ValueError):
                pass

        # Sort by access time (oldest first)
        entries.sort(key=lambda x: x[0])

        # Evict oldest fraction
        target = int(len(entries) * _EVICT_FRACTION) + 1
        freed = 0
        for atime, fpath, access_path in entries[:target]:
            try:
                sz = os.path.getsize(fpath)
                os.remove(fpath)
                if os.path.exists(access_path):
                    os.remove(access_path)
                freed += sz
            except OSError:
                pass

        logger.info(
            f"Cache eviction: removed {target} entries, "
            f"freed {freed / (1024**2):.1f} MB "
            f"(was {current_size / (1024**3):.2f} GB, "
            f"limit {CACHE_MAX_SIZE_BYTES / (1024**3):.2f} GB)"
        )
    except Exception as e:
        logger.debug(f"Cache eviction skipped: {e}")


def _touch_access(path: str) -> None:
    """Update the access timestamp for a cached embedding."""
    try:
        access_path = _access_file_for(path)
        with open(access_path, 'w') as f:
            f.write(f"{time.time()}")
    except Exception as e:
        logger.debug(f"Failed to update access time for {path}: {e}")


def clear_cache() -> None:
    """Remove all cached embeddings."""
    try:
        count = 0
        for fname in os.listdir(EMBEDDING_CACHE_DIR):
            fpath = os.path.join(EMBEDDING_CACHE_DIR, fname)
            if fname.endswith('.npz') or fname.endswith('.npz.access'):
                os.remove(fpath)
                count += 1
        logger.info(f"Cache cleared: removed {count} files")
        print(f"✓ Embedding cache cleared ({count} files removed).")
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        print(f"Error clearing cache: {e}")


def get_cache_size_mb() -> float:
    """Return the total size of the embedding cache in MB."""
    return _get_cache_size() / (1024 * 1024)


def get_cache_entry_count() -> int:
    """Return the number of cached embeddings."""
    count = 0
    try:
        for fname in os.listdir(EMBEDDING_CACHE_DIR):
            if fname.endswith('.npz'):
                count += 1
    except FileNotFoundError:
        pass
    return count


def load_embedding(path: str) -> Optional[np.ndarray]:
    """Return cached embedding if it exists and is up-to-date, else None."""
    try:
        cache_file = _cache_file_for(path)
        if not os.path.exists(cache_file):
            return None

        src_mtime = os.path.getmtime(path)
        data = np.load(cache_file)
        cached_mtime = float(data.get('mtime', 0.0))
        if abs(cached_mtime - src_mtime) > 1e-6:
            # Source file changed — invalidate
            return None

        # Track access for LRU eviction
        _touch_access(path)

        emb = data['embedding']
        return emb
    except Exception as e:
        logger.debug(f"Failed to load cache for {path}: {e}")
        return None


def save_embedding(path: str, embedding: np.ndarray) -> None:
    """Save embedding to cache alongside source mtime.

    Triggers LRU eviction if cache exceeds maximum size.
    """
    try:
        cache_file = _cache_file_for(path)
        src_mtime = os.path.getmtime(path)
        np.savez_compressed(cache_file, embedding=embedding, mtime=src_mtime)
        _touch_access(path)

        # Check if eviction is needed (after save to reduce race conditions)
        _evict_if_needed()
    except Exception as e:
        logger.debug(f"Failed to save cache for {path}: {e}")