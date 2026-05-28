"""
indexer.py — Unified indexing orchestrator for all Momento features.

Handles sequential and parallel execution of:
1. Image indexing (with multi-embedding support)
2. Video keyframe indexing
3. YOLO object detection
4. OCR text extraction

All features are enabled by default. Errors in one feature don't stop others.
Supports checkpoint/resume for crash recovery and graceful shutdown.

Phase 3.1: Independent features (Videos, Objects, OCR) run in parallel using
concurrent.futures.ThreadPoolExecutor while respecting dependency ordering:
Images must complete before Objects (YOLO needs images loaded). Videos and OCR
can run in parallel with everything.
"""

import os
import json
import time
import psutil
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from .index import Index
from .logger import get_logger
from .ingest import (
    add_images,
    add_images_multi,
    add_videos,
    add_objects,
    add_ocr,
)
from .config import (
    ENABLE_MULTI_EMBED,
    ENABLE_VIDEO_INDEXING,
    ENABLE_YOLO,
    ENABLE_OCR,
    BASE_DIR,
)
from .shutdown import is_shutdown_requested

logger = get_logger(__name__)

CHECKPOINT_FILE = os.path.join(BASE_DIR, "indexing_checkpoint.json")

# Minimum free memory in bytes before we warn the user
_MIN_FREE_MEMORY = 2 * 1024 * 1024 * 1024  # 2 GB


@dataclass
class IndexingStats:
    """Statistics from indexing operation."""
    images_added: int = 0
    videos_added: int = 0
    objects_added: int = 0
    ocr_added: int = 0
    total_vectors: int = 0
    duration_secs: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
    
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0


class FeatureName(Enum):
    """Feature names for logging."""
    IMAGES = "Image Indexing"
    VIDEOS = "Video Indexing"
    OBJECTS = "YOLO Object Detection"
    OCR = "OCR Text Extraction"


@dataclass
class IndexingCheckpoint:
    """Persistent checkpoint for crash recovery."""
    folder: str = ""
    features_completed: List[str] = field(default_factory=list)
    current_feature: str = ""
    started_at: float = 0.0

    def save(self) -> None:
        """Write checkpoint to disk."""
        try:
            os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(asdict(self), f)
        except OSError as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    @classmethod
    def load(cls, folder: str) -> Optional["IndexingCheckpoint"]:
        """Load checkpoint for the given folder, or None if not found."""
        if not os.path.exists(CHECKPOINT_FILE):
            return None
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                data = json.load(f)
            if data.get("folder") == folder:
                return cls(**data)
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
        return None

    @classmethod
    def clear(cls) -> None:
        """Delete the checkpoint file."""
        try:
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
        except OSError as e:
            logger.warning(f"Failed to clear checkpoint: {e}")


def _check_memory() -> bool:
    """Check available system memory.

    Returns:
        True if enough memory is available, False if user should be warned.
    """
    try:
        mem = psutil.virtual_memory()
        if mem.available < _MIN_FREE_MEMORY:
            free_gb = mem.available / (1024**3)
            print(f"\n⚠️  Low memory detected: {free_gb:.1f} GB free")
            print("   Indexing may be slow or fail. Consider closing other applications.")
            return False
    except Exception:
        pass  # psutil may not be available in all environments
    return True


class Indexer:
    """Orchestrator for indexing all Momento features."""
    
    def __init__(self, index: Index):
        """Initialize indexer.
        
        Args:
            index: Index instance to add vectors to
        """
        self.index = index
        self.stats = IndexingStats()
    
    def index_all_features(self, folder: str) -> IndexingStats:
        """Index all enabled features for a folder.
        
        Features execute with parallel execution where possible:
        - Images must complete first (YOLO and OCR depend on images being loaded)
        - Videos, Objects, and OCR can run in parallel after Images
        - One feature failure doesn't stop others
        
        Supports checkpoint/resume: if indexing is interrupted,
        restarting with the same folder skips completed features.
        
        Args:
            folder: Path to folder to index
            
        Returns:
            IndexingStats with results
        """
        start_time = time.time()
        self.stats = IndexingStats()
        
        # Check memory before starting
        _check_memory()
        
        # Try to resume from checkpoint
        checkpoint = IndexingCheckpoint.load(folder)
        resumed = checkpoint is not None
        completed = set(checkpoint.features_completed if checkpoint else [])
        
        if resumed:
            logger.info(f"Resuming indexing from checkpoint (completed: {completed})")
            print(f"\n🔄 Resuming previous indexing session...")
        
        logger.info(f"Starting auto-index for: {folder}")
        print(f"\n🗂️  Indexing folder: {folder}\n")
        
        checkpoint_data = IndexingCheckpoint(
            folder=folder,
            started_at=start_time,
        )
        
        # ── Step 1: Images (must complete first) ──────────────────────
        if is_shutdown_requested():
            print(f"\n⏹️  Shutdown requested — stopping before Images")
            checkpoint_data.save()
            self.stats.duration_secs = time.time() - start_time
            self.stats.total_vectors = self.index.get_vector_count()
            return self.stats
        
        if "IMAGES" not in completed:
            checkpoint_data.current_feature = "IMAGES"
            checkpoint_data.save()
            
            self._index_images(folder)
            
            checkpoint_data.features_completed.append("IMAGES")
            checkpoint_data.save()
        else:
            logger.info("Skipping already-completed feature: Image Indexing")
            print(f"⏩ Image Indexing already completed — skipping\n")
        
        # If shutdown requested after images, save progress and exit
        if is_shutdown_requested():
            print(f"\n⏹️  Shutdown requested — stopping before parallel features")
            checkpoint_data.save()
            self.stats.duration_secs = time.time() - start_time
            self.stats.total_vectors = self.index.get_vector_count()
            return self.stats
        
        # ── Step 2: Parallel features (Videos, Objects, OCR) ──────────
        # These are independent of each other and can run concurrently
        parallel_features = []
        
        if ENABLE_VIDEO_INDEXING and "VIDEOS" not in completed:
            parallel_features.append(("VIDEOS", self._index_videos, (folder,)))
        
        if ENABLE_YOLO and "OBJECTS" not in completed:
            parallel_features.append(("OBJECTS", self._index_objects, (folder,)))
        
        if ENABLE_OCR and "OCR" not in completed:
            parallel_features.append(("OCR", self._index_ocr, (folder,)))
        
        if parallel_features:
            print(f"\n⚡ Running {len(parallel_features)} features in parallel...\n")
            
            with ThreadPoolExecutor(max_workers=len(parallel_features)) as executor:
                future_map = {}
                for feat_name, func, args in parallel_features:
                    logger.info(f"Submitting {feat_name} to parallel executor")
                    future = executor.submit(func, *args)
                    future_map[future] = feat_name
                
                for future in as_completed(future_map):
                    feat_name = future_map[future]
                    try:
                        future.result()  # Re-raise any exception from the feature
                        checkpoint_data.features_completed.append(feat_name)
                        checkpoint_data.save()
                        logger.info(f"Parallel feature {feat_name} completed successfully")
                    except Exception as e:
                        logger.error(f"Parallel feature {feat_name} failed: {e}")
                        self.stats.add_error(f"{feat_name} failed: {e}")
        else:
            # Log which features were skipped because they're already completed
            for feat_name in ["VIDEOS", "OBJECTS", "OCR"]:
                if feat_name in completed:
                    name_map = {
                        "VIDEOS": "Video Indexing",
                        "OBJECTS": "YOLO Object Detection",
                        "OCR": "OCR Text Extraction",
                    }
                    logger.info(f"Skipping already-completed feature: {name_map[feat_name]}")
                    print(f"⏩ {name_map[feat_name]} already completed — skipping\n")
        
        # Finalize stats
        self.stats.duration_secs = time.time() - start_time
        self.stats.total_vectors = self.index.get_vector_count()
        
        # Clear checkpoint on successful completion
        if not is_shutdown_requested():
            IndexingCheckpoint.clear()
        
        logger.info(f"Indexing complete in {self.stats.duration_secs:.2f}s")
        return self.stats
    
    def _index_images(self, folder: str) -> None:
        """Index images with optional multi-embedding."""
        try:
            logger.info("Starting image indexing...")
            print("📷 Indexing images...", end=" ", flush=True)
            
            if is_shutdown_requested():
                print("⏹️  cancelled")
                return
            
            if ENABLE_MULTI_EMBED:
                count = add_images_multi(folder, self.index)
                self.stats.images_added = count
                print(f"✓ {count * 6} vectors from {count} images (multi-embed)\n")
            else:
                count = add_images(folder, self.index)
                self.stats.images_added = count
                print(f"✓ {count} vectors from {count} images\n")
            
            logger.info(f"Image indexing complete: {self.stats.images_added} added")
            
        except Exception as e:
            error_msg = f"Image indexing failed: {str(e)}"
            logger.error(error_msg)
            self.stats.add_error(error_msg)
            print(f"✗ {error_msg}\n")
    
    def _index_videos(self, folder: str) -> None:
        """Index video keyframes."""
        if not ENABLE_VIDEO_INDEXING:
            logger.info("Video indexing disabled in config")
            return
        
        if is_shutdown_requested():
            return
        
        try:
            logger.info("Starting video indexing...")
            print("🎬 Indexing videos...", end=" ", flush=True)
            
            count = add_videos(folder, self.index)
            self.stats.videos_added = count
            print(f"✓ {count} videos indexed\n")
            
            logger.info(f"Video indexing complete: {self.stats.videos_added} added")
            
        except Exception as e:
            error_msg = f"Video indexing failed: {str(e)}"
            logger.error(error_msg)
            self.stats.add_error(error_msg)
            print(f"✗ {error_msg}\n")
    
    def _index_objects(self, folder: str) -> None:
        """Index YOLO detected objects."""
        if not ENABLE_YOLO:
            logger.info("YOLO object detection disabled in config")
            return
        
        if is_shutdown_requested():
            return
        
        try:
            logger.info("Starting YOLO object detection...")
            print("🎯 Detecting objects with YOLO...", end=" ", flush=True)
            
            count = add_objects(folder, self.index)
            self.stats.objects_added = count
            print(f"✓ {count} objects indexed\n")
            
            logger.info(f"YOLO indexing complete: {self.stats.objects_added} added")
            
        except Exception as e:
            error_msg = f"YOLO object detection failed: {str(e)}"
            logger.error(error_msg)
            self.stats.add_error(error_msg)
            print(f"✗ {error_msg}\n")
    
    def _index_ocr(self, folder: str) -> None:
        """Index OCR text from images."""
        if not ENABLE_OCR:
            logger.info("OCR text extraction disabled in config")
            return
        
        if is_shutdown_requested():
            return
        
        try:
            logger.info("Starting OCR text extraction...")
            print("📝 Extracting text with OCR...", end=" ", flush=True)
            
            count = add_ocr(folder, self.index)
            self.stats.ocr_added = count
            print(f"✓ {count} texts indexed\n")
            
            logger.info(f"OCR indexing complete: {self.stats.ocr_added} added")
            
        except Exception as e:
            error_msg = f"OCR text extraction failed: {str(e)}"
            logger.error(error_msg)
            self.stats.add_error(error_msg)
            print(f"✗ {error_msg}\n")