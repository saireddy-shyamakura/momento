"""Unit tests for checkpoint.py — CheckpointManager and data classes.

Tests checkpoint creation, loading, feature status updates,
resume detection, schema validation, and clearing.
"""
import os
import json
import pytest


def _import_checkpoint():
    from momento.checkpoint import (
        CheckpointManager, IndexingCheckpoint, FeatureCheckpoint,
        FeatureStatus, get_checkpoint_manager,
    )
    return CheckpointManager, IndexingCheckpoint, FeatureCheckpoint, FeatureStatus, get_checkpoint_manager


class TestFeatureCheckpoint:
    """FeatureCheckpoint dataclass defaults."""

    def test_default_values(self):
        _, _, FeatureCheckpoint, _, _ = _import_checkpoint()
        fc = FeatureCheckpoint()
        assert fc.status == "pending"
        assert fc.count == 0
        assert fc.processed_files == []
        assert fc.current_file is None
        assert fc.current_position is None
        assert fc.error is None


class TestIndexingCheckpoint:
    """IndexingCheckpoint dataclass creation."""

    def test_creation(self):
        _, IndexingCheckpoint, _, _, _ = _import_checkpoint()
        cp = IndexingCheckpoint(folder="/test", collection_id="col123", timestamp="2024-01-01T00:00:00")
        assert cp.folder == "/test"
        assert cp.collection_id == "col123"
        assert cp.status == "in_progress"
        assert "images" in cp.features_status
        assert "multi_embed" in cp.features_status
        assert "videos" in cp.features_status
        assert "yolo" in cp.features_status
        assert "ocr" in cp.features_status


class TestCheckpointManagerCreate:
    """Creating checkpoints."""

    def test_create_checkpoint(self, tmp_path):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "checkpoint.json")
        mgr = CheckpointManager(checkpoint_file=cp_file)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("momento.checkpoint._get_timestamp", lambda: "2024-01-01T00:00:00")
            cp = mgr.create_checkpoint(folder="/test", collection_id="col123")

        assert cp.folder == "/test"
        assert cp.collection_id == "col123"
        assert mgr.current_checkpoint is not None
        assert os.path.exists(cp_file)

    def test_create_checkpoint_with_config(self, tmp_path):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "checkpoint.json")
        mgr = CheckpointManager(checkpoint_file=cp_file)

        from momento.config import MomentoConfig
        config = MomentoConfig(similarity_threshold=0.5)

        cp = mgr.create_checkpoint(folder="/test", collection_id="col123", config=config)
        assert cp.config_snapshot is not None
        assert cp.config_snapshot["similarity_threshold"] == 0.5


class TestCheckpointManagerLoad:
    """Loading checkpoints."""

    def test_load_checkpoint(self, tmp_path):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "checkpoint.json")
        mgr = CheckpointManager(checkpoint_file=cp_file)

        # Create
        mgr.create_checkpoint(folder="/test", collection_id="col123")

        # Load in new manager
        mgr2 = CheckpointManager(checkpoint_file=cp_file)
        loaded = mgr2.load_checkpoint()
        assert loaded is not None
        assert loaded.folder == "/test"
        assert loaded.collection_id == "col123"

    def test_load_missing_checkpoint(self, tmp_path):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "nonexistent.json")
        mgr = CheckpointManager(checkpoint_file=cp_file)
        assert mgr.load_checkpoint() is None

    def test_load_invalid_json(self, tmp_path):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "bad.json")
        with open(cp_file, "w") as f:
            f.write("not json")
        mgr = CheckpointManager(checkpoint_file=cp_file)
        assert mgr.load_checkpoint() is None

    def test_load_corrupted_data(self, tmp_path):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "bad.json")
        with open(cp_file, "w") as f:
            json.dump({"folder": 123}, f)  # folder should be string
        mgr = CheckpointManager(checkpoint_file=cp_file)
        # Should fail schema validation
        result = mgr.load_checkpoint()
        assert result is None


class TestCheckpointFeatureStatus:
    """Updating feature statuses."""

    def test_update_feature_status(self, tmp_path):
        CheckpointManager, _, _, FeatureStatus, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "checkpoint.json")
        mgr = CheckpointManager(checkpoint_file=cp_file)

        mgr.create_checkpoint(folder="/test", collection_id="col123")
        mgr.update_feature_status("images", FeatureStatus.COMPLETED, count=10)

        assert mgr.current_checkpoint.features_status["images"].status == "completed"
        assert mgr.current_checkpoint.features_status["images"].count == 10

    def test_update_new_feature(self, tmp_path):
        CheckpointManager, _, _, FeatureStatus, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "checkpoint.json")
        mgr = CheckpointManager(checkpoint_file=cp_file)

        mgr.create_checkpoint(folder="/test", collection_id="col123")
        mgr.update_feature_status("custom_feature", FeatureStatus.IN_PROGRESS)
        assert "custom_feature" in mgr.current_checkpoint.features_status

    def test_update_no_active_checkpoint(self, tmp_path):
        CheckpointManager, _, _, FeatureStatus, _ = _import_checkpoint()
        mgr = CheckpointManager(checkpoint_file="/nonexistent/checkpoint.json")

        # Should not raise
        mgr.update_feature_status("images", FeatureStatus.COMPLETED)

    def test_mark_completed(self, tmp_path):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "checkpoint.json")
        mgr = CheckpointManager(checkpoint_file=cp_file)

        mgr.create_checkpoint(folder="/test", collection_id="col123")
        mgr.mark_completed("yolo")
        assert mgr.current_checkpoint.features_status["yolo"].status == "completed"


class TestCheckpointResume:
    """Resume feature detection."""

    def test_should_resume_in_progress(self, tmp_path):
        CheckpointManager, _, _, FeatureStatus, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "checkpoint.json")
        mgr = CheckpointManager(checkpoint_file=cp_file)

        mgr.create_checkpoint(folder="/test", collection_id="col123")
        mgr.update_feature_status("images", FeatureStatus.IN_PROGRESS)
        assert mgr.should_resume_feature("images") is True

    def test_should_not_resume_completed(self, tmp_path):
        CheckpointManager, _, _, FeatureStatus, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "checkpoint.json")
        mgr = CheckpointManager(checkpoint_file=cp_file)

        mgr.create_checkpoint(folder="/test", collection_id="col123")
        mgr.update_feature_status("images", FeatureStatus.COMPLETED, count=10)
        # Completed with count > 0 → no resume
        assert mgr.should_resume_feature("images") is False

    def test_should_resume_completed_with_zero_count(self, tmp_path):
        CheckpointManager, _, _, FeatureStatus, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "checkpoint.json")
        mgr = CheckpointManager(checkpoint_file=cp_file)

        mgr.create_checkpoint(folder="/test", collection_id="col123")
        mgr.update_feature_status("images", FeatureStatus.COMPLETED, count=0)
        # Completed with count == 0 → should resume
        assert mgr.should_resume_feature("images") is True

    def test_should_not_resume_missing_feature(self, tmp_path):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "checkpoint.json")
        mgr = CheckpointManager(checkpoint_file=cp_file)

        mgr.create_checkpoint(folder="/test", collection_id="col123")
        assert mgr.should_resume_feature("nonexistent") is False

    def test_should_not_resume_no_checkpoint(self, tmp_path):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        mgr = CheckpointManager(checkpoint_file="/nonexistent/checkpoint.json")
        assert mgr.should_resume_feature("images") is False


class TestCheckpointGetResumeState:
    """Getting resume state for a feature."""

    def test_get_resume_state(self, tmp_path):
        CheckpointManager, _, _, FeatureStatus, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "checkpoint.json")
        mgr = CheckpointManager(checkpoint_file=cp_file)

        mgr.create_checkpoint(folder="/test", collection_id="col123")
        mgr.update_feature_status(
            "images", FeatureStatus.IN_PROGRESS,
            processed_files=["/a.jpg", "/b.jpg"],
            current_file="/c.jpg",
            count=2,
        )

        state = mgr.get_resumed_state("images")
        assert state["processed_files"] == ["/a.jpg", "/b.jpg"]
        assert state["current_file"] == "/c.jpg"
        assert state["count"] == 2

    def test_get_resume_state_no_checkpoint(self, tmp_path):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        mgr = CheckpointManager(checkpoint_file="/nonexistent/checkpoint.json")
        assert mgr.get_resumed_state("images") == {}

    def test_get_resume_state_missing_feature(self, tmp_path):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "checkpoint.json")
        mgr = CheckpointManager(checkpoint_file=cp_file)
        mgr.create_checkpoint(folder="/test", collection_id="col123")
        assert mgr.get_resumed_state("nonexistent") == {}


class TestCheckpointClear:
    """Clearing checkpoints."""

    def test_clear_checkpoint(self, tmp_path):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        cp_file = os.path.join(tmp_path, "checkpoint.json")
        mgr = CheckpointManager(checkpoint_file=cp_file)

        mgr.create_checkpoint(folder="/test", collection_id="col123")
        assert os.path.exists(cp_file)
        mgr.clear_checkpoint()
        assert not os.path.exists(cp_file)
        assert mgr.current_checkpoint is None


class TestSchemaValidation:
    """Checkpoint schema validation."""

    def test_valid_schema(self, tmp_path):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        CheckpointManager._validate_checkpoint_schema({
            "folder": "/test",
            "collection_id": "col123",
            "timestamp": "2024-01-01T00:00:00",
        })

    def test_missing_folder(self):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        assert CheckpointManager._validate_checkpoint_schema({
            "collection_id": "col123",
            "timestamp": "2024-01-01T00:00:00",
        }) is False

    def test_invalid_feature_status(self):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        assert CheckpointManager._validate_checkpoint_schema({
            "folder": "/test",
            "collection_id": "col123",
            "timestamp": "2024-01-01T00:00:00",
            "features_status": {"images": {"status": "invalid_status"}},
        }) is False

    def test_non_dict_features(self):
        CheckpointManager, _, _, _, _ = _import_checkpoint()
        assert CheckpointManager._validate_checkpoint_schema({
            "folder": "/test",
            "collection_id": "col123",
            "timestamp": "2024-01-01T00:00:00",
            "features_status": "not_a_dict",
        }) is False


class TestFeatureStatusEnum:
    """FeatureStatus enum values."""

    def test_enum_values(self):
        _, _, _, FeatureStatus, _ = _import_checkpoint()
        assert FeatureStatus.PENDING.value == "pending"
        assert FeatureStatus.IN_PROGRESS.value == "in_progress"
        assert FeatureStatus.COMPLETED.value == "completed"
        assert FeatureStatus.FAILED.value == "failed"


class TestGetCheckpointManager:
    """Global checkpoint manager singleton."""

    def test_singleton(self):
        _, _, _, _, get_checkpoint_manager = _import_checkpoint()
        mgr1 = get_checkpoint_manager()
        mgr2 = get_checkpoint_manager()
        assert mgr1 is mgr2