"""
file_picker.py — User interface for folder and file selection.

Handles:
- Prompting for folder path
- Path validation
- Confirming before indexing with disk space estimation
- Previewing what will be indexed
"""

import os
import shutil
from typing import Tuple, Dict, List

from .validation import validate_folder_path
from .config import SUPPORTED_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
from .logger import get_logger

logger = get_logger(__name__)

# Estimated bytes per vector in ChromaDB (embedding + metadata + index overhead)
# CLIP ViT-B/16 produces 512-dim float32 vectors ≈ 2KB each
# ChromaDB overhead ≈ 2x for HNSW index + metadata
_BYTES_PER_VECTOR = 512 * 4 * 2  # ~4 KB per vector

# Estimated vectors per unit
_VECTORS_PER_IMAGE = 6       # 1 original + 5 augmentations
_VECTORS_PER_VIDEO_FRAME = 1
_FRAMES_PER_VIDEO = 25       # average across config max 50
_VECTORS_PER_OBJECT = 3      # average detected objects per image
_VECTORS_PER_OCR = 1         # 1 text embedding per image


class FilePicker:
    """Handles folder and file selection UI."""
    
    def prompt_folder_path(self) -> str:
        """Prompt user for a folder path.
        
        Returns:
            Absolute path to validated folder
            
        Raises:
            ValueError: If user provides invalid path
        """
        print("\n" + "="*50)
        print("📂 Select Folder to Index")
        print("="*50)
        
        max_attempts = 3
        attempts = 0
        
        while attempts < max_attempts:
            path = input("Enter folder path (or 'q' to quit): ").strip()
            
            if path.lower() == 'q':
                print("Cancelled.")
                import sys
                sys.exit(0)
            
            if not path:
                print("Path cannot be empty.")
                attempts += 1
                continue
            
            # Expand user home directory
            path = os.path.expanduser(path)
            path = os.path.abspath(path)
            
            # Validate
            is_valid, error_msg = validate_folder_path(path)
            if not is_valid:
                print(f"❌ Invalid: {error_msg}")
                attempts += 1
                continue
            
            logger.info(f"Folder selected: {path}")
            return path
        
        print(f"Failed to select valid folder after {max_attempts} attempts.")
        raise ValueError("Invalid folder path")
    
    def confirm_folder(self, folder: str) -> bool:
        """Confirm folder before indexing with preview and disk space check.
        
        Args:
            folder: Path to folder
            
        Returns:
            True if user confirms, False otherwise
        """
        preview = self.preview_indexable_files(folder)
        
        print("\n" + "="*50)
        print("📋 Indexing Preview")
        print("="*50)
        print(f"Folder:      {folder}")
        print(f"Images:      {preview['image_count']}")
        print(f"Videos:      {preview['video_count']}")
        print(f"Total items: {preview['total_count']}")
        print("="*50)
        
        if preview['total_count'] == 0:
            print("\n⚠️  No supported media files found in this folder!")
            return False
        
        # Estimate space needed
        space_info = self._estimate_space_needed(preview)
        free_gb = space_info['free_gb']
        needed_gb = space_info['needed_gb']
        vector_estimate = space_info['estimated_vectors']
        
        print(f"\n📊 Storage Estimate:")
        print(f"  Estimated vectors: {vector_estimate:,}")
        print(f"  Estimated storage: {needed_gb:.1f} GB")
        print(f"  Free disk space:   {free_gb:.1f} GB")
        
        if needed_gb > free_gb * 0.8:
            print(f"\n⚠️  Warning: Indexing may use {needed_gb:.1f} GB but only {free_gb:.1f} GB is available!")
            print("   Consider freeing disk space or indexing fewer files.")
            if needed_gb > free_gb:
                print("\n❌ Not enough disk space. Cannot proceed with indexing.")
                return False
            response = input("\nLow disk space — continue anyway? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("Cancelled.")
                return False
        
        print("\nThis will index all features:")
        print("  📷 Multi-embedding (image augmentation)")
        print("  🎬 Video keyframes")
        print("  🎯 YOLO object detection")
        print("  📝 OCR text extraction")
        
        while True:
            response = input("\nProceed with indexing? (yes/no): ").strip().lower()
            
            if response in ['yes', 'y']:
                print("Starting indexing...")
                return True
            elif response in ['no', 'n']:
                print("Cancelled.")
                return False
            else:
                print("Please enter 'yes' or 'no'.")
    
    def _estimate_space_needed(self, preview: Dict[str, int]) -> Dict:
        """Estimate disk space needed for indexing this folder.
        
        Args:
            preview: Dict with image_count and video_count
            
        Returns:
            Dict with free_gb, needed_gb, estimated_vectors
        """
        # Calculate estimated total vectors
        image_vectors = preview['image_count'] * _VECTORS_PER_IMAGE
        video_vectors = preview['video_count'] * _VECTORS_PER_VIDEO_FRAME * _FRAMES_PER_VIDEO
        object_vectors = preview['image_count'] * _VECTORS_PER_OBJECT
        ocr_vectors = preview['image_count'] * _VECTORS_PER_OCR
        
        total_vectors = image_vectors + video_vectors + object_vectors + ocr_vectors
        
        # Calculate raw storage (ChromaDB has ~2x overhead + some fixed costs)
        raw_bytes = total_vectors * _BYTES_PER_VECTOR
        overhead_bytes = raw_bytes * 0.5  # ChromaDB WAL + internal structures
        total_bytes = raw_bytes + overhead_bytes + 50 * 1024 * 1024  # +50MB fixed overhead
        
        needed_gb = total_bytes / (1024**3)
        
        # Get disk free space
        try:
            disk_usage = shutil.disk_usage(os.path.dirname(self._get_data_dir() or "/"))
            free_gb = disk_usage.free / (1024**3)
        except Exception:
            free_gb = float('inf')  # Can't determine, skip check
        
        return {
            'free_gb': free_gb,
            'needed_gb': needed_gb,
            'estimated_vectors': total_vectors,
        }
    
    @staticmethod
    def _get_data_dir() -> str:
        """Get the database data directory."""
        from .config import BASE_DIR
        return BASE_DIR
    
    def preview_indexable_files(self, folder: str) -> Dict[str, int]:
        """Preview files that will be indexed.
        
        Args:
            folder: Path to folder
            
        Returns:
            Dict with file counts
        """
        image_count = 0
        video_count = 0
        
        try:
            for root, _dirs, files in os.walk(folder, followlinks=False):
                for file in files:
                    file_lower = file.lower()
                    
                    if file_lower.endswith(SUPPORTED_EXTENSIONS):
                        image_count += 1
                    elif file_lower.endswith(SUPPORTED_VIDEO_EXTENSIONS):
                        video_count += 1
        
        except Exception as e:
            logger.error(f"Error scanning folder: {e}")
        
        total_count = image_count + video_count
        
        return {
            'image_count': image_count,
            'video_count': video_count,
            'total_count': total_count,
        }
    
    def select_image_from_folder(self, folder: str) -> str:
        """Let user select an image from a folder for searching.
        
        Args:
            folder: Path to folder
            
        Returns:
            Path to selected image or empty string if cancelled
        """
        images = []
        
        try:
            for root, _dirs, files in os.walk(folder, followlinks=False):
                for file in files:
                    if file.lower().endswith(SUPPORTED_EXTENSIONS):
                        full_path = os.path.join(root, file)
                        images.append(full_path)
        except Exception as e:
            logger.error(f"Error scanning folder: {e}")
            return ""
        
        if not images:
            print("No images found in folder.")
            return ""
        
        # Simple picker - just list and let user choose
        print("\n📷 Available images:")
        for i, img in enumerate(images[:10], 1):
            print(f"  {i}. {os.path.basename(img)}")
        
        if len(images) > 10:
            print(f"  ... and {len(images) - 10} more")
        
        try:
            choice = int(input("Enter image number (or 0 to cancel): ")) - 1
            if 0 <= choice < len(images):
                return images[choice]
        except (ValueError, IndexError):
            pass
        
        return ""