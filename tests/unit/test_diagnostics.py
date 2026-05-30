"""Unit tests for diagnostics.py — health, stats, and benchmark.

Tests DoctorResult, run_doctor, get_index_stats, run_benchmark.
Uses mocking to avoid loading models or accessing real databases.
"""
from unittest.mock import MagicMock, patch
import pytest


class TestDoctorResult:
    """DoctorResult dataclass."""

    def test_is_healthy_with_no_errors(self):
        from momento.diagnostics import DoctorResult
        result = DoctorResult()
        assert result.is_healthy() is True

    def test_is_not_healthy_with_errors(self):
        from momento.diagnostics import DoctorResult
        result = DoctorResult(errors=["something failed"])
        assert result.is_healthy() is False


class TestRunDoctor:
    """run_doctor health check."""

    def test_run_doctor_returns_result(self):
        with patch("momento.diagnostics._check_python_version", return_value=(True, "3.12.0")), \
             patch("momento.diagnostics._check_momento_version", return_value="3.0.0"), \
             patch("momento.diagnostics._check_clip_model", return_value=True), \
             patch("momento.diagnostics._check_chromadb", return_value=True), \
             patch("momento.diagnostics._check_disk_space", return_value=(100.0, 50.0)), \
             patch("momento.diagnostics._check_gpu", return_value=(False, "")), \
             patch("momento.device.DeviceManager"), \
             patch("momento.diagnostics.os.path.exists", return_value=False):
            from momento.diagnostics import run_doctor
            result = run_doctor()
            assert result.python_version == "3.12.0"
            assert result.momento_version == "3.0.0"

    def test_doctor_detects_python_version_failure(self):
        with patch("momento.diagnostics._check_python_version", return_value=(False, "3.10.0")), \
             patch("momento.diagnostics._check_momento_version", return_value="3.0.0"), \
             patch("momento.diagnostics._check_clip_model", return_value=True), \
             patch("momento.diagnostics._check_chromadb", return_value=True), \
             patch("momento.diagnostics._check_disk_space", return_value=(100.0, 50.0)), \
             patch("momento.diagnostics._check_gpu", return_value=(False, "")), \
             patch("momento.device.DeviceManager"), \
             patch("momento.diagnostics.os.path.exists", return_value=False):
            from momento.diagnostics import run_doctor
            result = run_doctor()
            assert len(result.errors) > 0


class TestGetIndexStats:
    """get_index_stats function."""

    def test_get_index_stats(self):
        # Index is lazily imported inside get_index_stats's function body
        mock_index = MagicMock()
        mock_index.get_vector_count.return_value = 100
        mock_index.get_all_paths.return_value = ["/a.jpg", "/b.jpg"]

        with patch("momento.index.Index", return_value=mock_index), \
             patch("momento.diagnostics.os.path.exists", return_value=False):
            from momento.diagnostics import get_index_stats
            stats = get_index_stats("/fake/path")
            assert stats.total_vectors == 100
            assert stats.total_entries == 2


class TestRunBenchmark:
    """run_benchmark function."""

    def test_run_benchmark_empty_index(self):
        mock_index = MagicMock()
        mock_index.get_vector_count.return_value = 0

        with patch("momento.index.Index", return_value=mock_index), \
             patch("momento.features.extract_text_features"):
            from momento.diagnostics import run_benchmark
            result = run_benchmark("/fake/path", iterations=1)
            assert result.index_vector_count == 0
