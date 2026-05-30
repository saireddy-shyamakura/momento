"""Unit tests for get_device() in config.py.

Uses unittest.mock.patch to mock torch availability checks so that
tests pass on any hardware — no physical GPU required.

Validates: Requirements 5.1, 5.2, 5.3, 5.4
"""

from unittest.mock import patch, MagicMock
import pytest


class TestGetDevice:
    """Tests for config.get_device() covering all three hardware paths."""

    def test_cuda_available_returns_cuda(self):
        """
        WHEN torch.cuda.is_available is mocked to return True,
        get_device() SHALL return "cuda".

        Validates: Requirements 5.1, 5.4
        """
        mock_cuda_device = MagicMock()
        mock_cuda_device.device_count.return_value = 1
        mock_cuda_device.get_device_name.return_value = "Mock GPU"

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_name", return_value="Mock GPU"):
            from momento.config import get_device
            result = get_device()

        assert result == "cuda"

    def test_mps_available_returns_mps(self):
        """
        WHEN torch.cuda.is_available is mocked to return False and
        torch.backends.mps.is_available is mocked to return True,
        get_device() SHALL return "mps".

        Validates: Requirements 5.2, 5.4
        """
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True):
            from momento.config import get_device
            result = get_device()

        assert result == "mps"

    def test_no_gpu_returns_cpu(self):
        """
        WHEN both torch.cuda.is_available and torch.backends.mps.is_available
        are mocked to return False, get_device() SHALL return "cpu".

        Validates: Requirements 5.3, 5.4
        """
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            from momento.config import get_device
            result = get_device()

        assert result == "cpu"
