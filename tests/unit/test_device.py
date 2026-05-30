"""Unit tests for device.py — DeviceManager singleton, env overrides, and fallback.

Covers:
- Auto-detection of CUDA / MPS / CPU
- MOMENTO_DEVICE environment variable override
- OOM fallback to CPU
- dtype selection
- get_free_memory_mb / clear_cache (non-GPU paths)
"""

from unittest.mock import patch, MagicMock
import pytest
import torch

from momento.device import DeviceManager, device_manager


class TestDeviceManagerInit:
    """Tests for DeviceManager initialization and singleton."""

    @patch.dict("os.environ", {}, clear=True)
    def test_auto_detect_cpu_when_no_gpu(self):
        """When neither CUDA nor MPS is available, device should be 'cpu'."""
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            manager = DeviceManager()
            assert manager.device == "cpu"
            assert manager.dtype == torch.float32

    @patch.dict("os.environ", {}, clear=True)
    def test_auto_detect_cuda(self):
        """When CUDA is available, device should be 'cuda'."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2), \
             patch("torch.cuda.get_device_name", return_value="Test GPU"):
            manager = DeviceManager()
            assert manager.device == "cuda"

    @patch.dict("os.environ", {}, clear=True)
    def test_auto_detect_mps(self):
        """When MPS is available (no CUDA), device should be 'mps'."""
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True):
            manager = DeviceManager()
            assert manager.device == "mps"

    @patch.dict("os.environ", {"MOMENTO_DEVICE": "cpu"})
    def test_env_override_cpu(self):
        """MOMENTO_DEVICE=cpu forces CPU even if CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = DeviceManager()
            assert manager.device == "cpu"

    @patch.dict("os.environ", {"MOMENTO_DEVICE": "cuda"})
    def test_env_override_cuda_available(self):
        """MOMENTO_DEVICE=cuda works when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_name", return_value="Mock"):
            manager = DeviceManager()
            assert manager.device == "cuda"

    @patch.dict("os.environ", {"MOMENTO_DEVICE": "cuda"})
    def test_env_override_cuda_unavailable_falls_back(self):
        """MOMENTO_DEVICE=cuda falls back to cpu if CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            manager = DeviceManager()
            assert manager.device == "cpu"

    @patch.dict("os.environ", {"MOMENTO_DEVICE": "mps"})
    def test_env_override_mps_unavailable_falls_back(self):
        """MOMENTO_DEVICE=mps falls back to cpu if MPS unavailable."""
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            manager = DeviceManager()
            assert manager.device == "cpu"

    @patch.dict("os.environ", {"MOMENTO_DEVICE": "invalid"})
    def test_invalid_env_ignored(self):
        """Invalid MOMENTO_DEVICE values are ignored; auto-detect runs."""
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            manager = DeviceManager()
            assert manager.device == "cpu"

    def test_singleton_instance(self):
        """device_manager should be a DeviceManager instance."""
        assert isinstance(device_manager, DeviceManager)


class TestDeviceManagerFallback:
    """Tests for fallback_to_cpu method."""

    def test_fallback_from_cuda(self):
        """fallback_to_cpu changes device to cpu and dtype to float32."""
        manager = DeviceManager()
        manager._device = "cuda"
        manager._dtype = torch.float16
        manager.fallback_to_cpu()
        assert manager.device == "cpu"
        assert manager.dtype == torch.float32

    def test_fallback_from_cpu_is_noop(self):
        """fallback_to_cpu on CPU device should not raise."""
        manager = DeviceManager()
        manager._device = "cpu"
        manager.fallback_to_cpu()
        assert manager.device == "cpu"


class TestDeviceManagerProperties:
    """Tests for DeviceManager properties."""

    def test_torch_device_property(self):
        """torch_device returns a torch.device object."""
        manager = DeviceManager()
        manager._device = "cpu"
        assert isinstance(manager.torch_device, torch.device)
        assert str(manager.torch_device) == "cpu"

    def test_to_device_moves_tensor(self):
        """to_device moves a tensor to the target device with correct dtype."""
        manager = DeviceManager()
        manager._device = "cpu"
        manager._dtype = torch.float32
        tensor = torch.randn(3, 3)
        result = manager.to_device(tensor)
        assert result.device.type == "cpu"
        assert result.dtype == torch.float32

    def test_get_free_memory_mb_cpu_returns_none(self):
        """On CPU, get_free_memory_mb returns None."""
        manager = DeviceManager()
        manager._device = "cpu"
        assert manager.get_free_memory_mb() is None

    def test_get_free_memory_mb_cuda(self):
        """On CUDA, get_free_memory_mb returns a number."""
        with patch("torch.cuda.mem_get_info", return_value=(2 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)):
            manager = DeviceManager()
            manager._device = "cuda"
            free_mb = manager.get_free_memory_mb()
            assert free_mb == 2048.0

    def test_clear_cache_cpu_no_error(self):
        """clear_cache on CPU does nothing and doesn't raise."""
        manager = DeviceManager()
        manager._device = "cpu"
        manager.clear_cache()  # Should not raise

    def test_clear_cache_cuda(self):
        """clear_cache on CUDA calls torch.cuda.empty_cache."""
        with patch("torch.cuda.empty_cache") as mock_empty:
            manager = DeviceManager()
            manager._device = "cuda"
            manager.clear_cache()
            mock_empty.assert_called_once()

    @patch.dict("os.environ", {}, clear=True)
    def test_mps_dtype_is_float32(self):
        """MPS device dtype should be float32 (not float16)."""
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True):
            manager = DeviceManager()
            assert manager.dtype == torch.float32

    @patch.dict("os.environ", {}, clear=True)
    def test_cuda_dtype_is_float32(self):
        """CUDA device dtype should be float32."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_name", return_value="Mock"):
            manager = DeviceManager()
            assert manager.dtype == torch.float32