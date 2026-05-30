"""
checkpoint.py — Checkpoint and recovery system for interrupted indexing operations.

Saves progress periodically so interrupted indexing can resume without
re-processing already completed items.

Checkpoint tracks:
- Folder being indexed
- Collection ID
- Feature status (completed, in-progress, pending)
- Processed files per feature
- Current position for resumable features
- Configuration snapshot at time of checkpoint
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from .config import BASE_DIR, MomentoConfig
from .logger import get_logger

logger = get_logger(__name__)

CHECKPOINT_FILE = os.path.join(BASE_DIR, "indexing_checkpoint.json")


class FeatureStatus(str, Enum):
    """Status of a feature in the indexing pipeline."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FeatureCheckpoint:
    """Checkpoint data for a single feature."""
    status: str = FeatureStatus.PENDING.value
    count: int = 0
    processed_files: List[str] = field(default_factory=list)
    current_file: Optional[str] = None
    current_position: Optional[int] = None  # For resumable operations like video frames
    error: Optional[str] = None


@dataclass
class IndexingCheckpoint:
    """Checkpoint for an indexing operation."""
    folder: str
    collection_id: str
    timestamp: str
    status: str = "in_progress"
    
    # Feature status tracking
    features_status: Dict[str, FeatureCheckpoint] = field(default_factory=lambda: {
        "images": FeatureCheckpoint(),
        "multi_embed": FeatureCheckpoint(),
        "videos": FeatureCheckpoint(),
        "yolo": FeatureCheckpoint(),
        "ocr": FeatureCheckpoint(),
    })
    
    # Configuration snapshot (to detect config changes)
    config_snapshot: Optional[Dict[str, Any]] = None


class CheckpointManager:
    """Manage checkpoint save/load for crash recovery."""
    
    def __init__(self, checkpoint_file: str = CHECKPOINT_FILE):
        """Initialize checkpoint manager."""
        self.checkpoint_file = checkpoint_file
        self.current_checkpoint: Optional[IndexingCheckpoint] = None
    
    def create_checkpoint(
        self,
        folder: str,
        collection_id: str,
        config: Optional[MomentoConfig] = None
    ) -> IndexingCheckpoint:
        """Create a new checkpoint for indexing operation.
        
        Args:
            folder: Folder being indexed
            collection_id: ChromaDB collection ID
            config: Current configuration (saved in checkpoint)
            
        Returns:
            Created checkpoint
        """
        checkpoint = IndexingCheckpoint(
            folder=folder,
            collection_id=collection_id,
            timestamp=_get_timestamp(),
            config_snapshot=asdict(config) if config else None
        )
        self.current_checkpoint = checkpoint
        self._save_checkpoint(checkpoint)
        logger.info(f"Created checkpoint for: {folder}")
        return checkpoint
    
    @staticmethod
    def _validate_checkpoint_schema(data: dict) -> bool:
        """Basic schema validation for checkpoint data.

        Ensures required top-level keys exist and have the expected
        types to prevent corrupted/malicious JSON from crashing the
        application or causing silent data corruption.

        Returns:
            True if data passes schema validation.
        """
        required_str_keys = ("folder", "collection_id", "timestamp")
        for key in required_str_keys:
            val = data.get(key)
            if not isinstance(val, str) or not val:
                return False

        features = data.get("features_status")
        if features is not None and not isinstance(features, dict):
            return False

        if features:
            for feat_name, feat_data in features.items():
                if not isinstance(feat_data, dict):
                    return False
                # status must be a known value if present
                status = feat_data.get("status")
                if status is not None and status not in (
                    FeatureStatus.PENDING.value,
                    FeatureStatus.IN_PROGRESS.value,
                    FeatureStatus.COMPLETED.value,
                    FeatureStatus.FAILED.value,
                ):
                    return False
        return True

    def load_checkpoint(self) -> Optional[IndexingCheckpoint]:
        """Load existing checkpoint if present and valid.

        Validates the JSON structure against a basic schema to
        prevent corrupted or malicious data from being loaded.

        Returns:
            Loaded checkpoint, or None if no valid checkpoint exists.
        """
        if not os.path.exists(self.checkpoint_file):
            return None

        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                logger.warning("Checkpoint file is not a valid JSON object — ignoring")
                return None

            if not self._validate_checkpoint_schema(data):
                logger.warning("Checkpoint schema validation failed — ignoring")
                return None

            # Reconstruct checkpoint
            checkpoint = IndexingCheckpoint(
                folder=data["folder"],
                collection_id=data["collection_id"],
                timestamp=data["timestamp"],
                status=data.get("status", "in_progress"),
                config_snapshot=data.get("config_snapshot")
            )

            # Reconstruct feature statuses
            for feature_name, feature_data in data.get("features_status", {}).items():
                checkpoint.features_status[feature_name] = FeatureCheckpoint(
                    status=feature_data.get("status", FeatureStatus.PENDING.value),
                    count=feature_data.get("count", 0),
                    processed_files=feature_data.get("processed_files", []),
                    current_file=feature_data.get("current_file"),
                    current_position=feature_data.get("current_position"),
                    error=feature_data.get("error")
                )

            self.current_checkpoint = checkpoint
            logger.info(f"Loaded checkpoint from: {self.checkpoint_file}")
            return checkpoint

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def update_feature_status(
        self,
        feature_name: str,
        status: FeatureStatus,
        processed_files: Optional[List[str]] = None,
        current_file: Optional[str] = None,
        current_position: Optional[int] = None,
        count: Optional[int] = None,
        error: Optional[str] = None
    ) -> None:
        """Update status of a feature in current checkpoint.
        
        Args:
            feature_name: Name of feature (images, videos, yolo, etc.)
            status: New status
            processed_files: List of processed files
            current_file: Currently processing file
            current_position: Position in current file (for resumable operations)
            count: Number of items processed
            error: Error message if any
        """
        if self.current_checkpoint is None:
            logger.warning("No active checkpoint to update")
            return
        
        if feature_name not in self.current_checkpoint.features_status:
            self.current_checkpoint.features_status[feature_name] = FeatureCheckpoint()
        
        feature_cp = self.current_checkpoint.features_status[feature_name]
        feature_cp.status = status.value
        
        if processed_files is not None:
            feature_cp.processed_files = processed_files
        if current_file is not None:
            feature_cp.current_file = current_file
        if current_position is not None:
            feature_cp.current_position = current_position
        if count is not None:
            feature_cp.count = count
        if error is not None:
            feature_cp.error = error
        
        self._save_checkpoint(self.current_checkpoint)
    
    def mark_completed(self, feature_name: str) -> None:
        """Mark a feature as completed.
        
        Args:
            feature_name: Name of feature
        """
        self.update_feature_status(feature_name, FeatureStatus.COMPLETED)
        logger.info(f"Marked feature '{feature_name}' as completed")
    
    def should_resume_feature(self, feature_name: str) -> bool:
        """Check if a feature should be resumed.
        
        Args:
            feature_name: Name of feature
            
        Returns:
            True if feature has in-progress or partial status
        """
        if self.current_checkpoint is None:
            return False
        
        feature_cp = self.current_checkpoint.features_status.get(feature_name)
        if feature_cp is None:
            return False
        
        # Resume if in progress or has partial data
        return (
            feature_cp.status == FeatureStatus.IN_PROGRESS.value or
            (feature_cp.status == FeatureStatus.COMPLETED.value and feature_cp.count == 0)
        )
    
    def get_resumed_state(self, feature_name: str) -> Dict[str, Any]:
        """Get resume state for a feature.
        
        Args:
            feature_name: Name of feature
            
        Returns:
            Dictionary with resume state info
        """
        if self.current_checkpoint is None:
            return {}
        
        feature_cp = self.current_checkpoint.features_status.get(feature_name)
        if feature_cp is None:
            return {}
        
        return {
            "processed_files": feature_cp.processed_files,
            "current_file": feature_cp.current_file,
            "current_position": feature_cp.current_position,
            "count": feature_cp.count,
        }
    
    def clear_checkpoint(self) -> None:
        """Clear checkpoint file (call after successful completion)."""
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                logger.info("Checkpoint cleared")
                self.current_checkpoint = None
        except Exception as e:
            logger.warning(f"Failed to clear checkpoint: {e}")
    
    def _save_checkpoint(self, checkpoint: IndexingCheckpoint) -> None:
        """Internal method to save checkpoint to file.
        
        Args:
            checkpoint: Checkpoint to save
        """
        try:
            os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
            
            # Prepare data
            data = {
                "folder": checkpoint.folder,
                "collection_id": checkpoint.collection_id,
                "timestamp": checkpoint.timestamp,
                "status": checkpoint.status,
                "config_snapshot": checkpoint.config_snapshot,
                "features_status": {}
            }
            
            # Serialize feature statuses
            for feature_name, feature_cp in checkpoint.features_status.items():
                data["features_status"][feature_name] = {
                    "status": feature_cp.status,
                    "count": feature_cp.count,
                    "processed_files": feature_cp.processed_files,
                    "current_file": feature_cp.current_file,
                    "current_position": feature_cp.current_position,
                    "error": feature_cp.error,
                }
            
            # Write checkpoint
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")


def _get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# Global checkpoint manager instance
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager() -> CheckpointManager:
    """Get or create global checkpoint manager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()
    return _checkpoint_manager
