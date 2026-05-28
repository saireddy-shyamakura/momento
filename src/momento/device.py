"""
device.py — Centralized device management for Momento.

Provides a DeviceManager that handles:
- Auto-detection of CUDA / MPS / CPU
- Environment variable override (MOMENTO_DEVICE)
- MPS dtype fallback (float16 → float32)
- OOM auto-fallback to CPU
- Memory availability checks
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages compute device selection and dtype handling across all backends."""

    def __init__(self):
        self._device: str = self._detect_device()
        self._dtype: torch.dtype = self._select_dtype()
        logger.info(f"DeviceManager initialized: device={self._device}, dtype={self._dtype}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        """Current device string ('cuda', 'mps', or 'cpu')."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Recommended dtype for the current device."""
        return self._dtype

    @property
    def torch_device(self) -> torch.device:
        """Return a torch.device object."""
        return torch.device(self._device)

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the active device with the correct dtype."""
        return tensor.to(device=self._device, dtype=self._dtype)

    def fallback_to_cpu(self) -> None:
        """Force fallback to CPU (e.g. after an OOM error)."""
        if self._device != "cpu":
            logger.warning(f"Falling back from {self._device} → cpu")
            self._device = "cpu"
            self._dtype = torch.float32

    def get_free_memory_mb(self) -> float | None:
        """Return free GPU memory in MB, or None if not applicable."""
        if self._device == "cuda":
            free, _ = torch.cuda.mem_get_info()
            return free / (1024 * 1024)
        return None

    def clear_cache(self) -> None:
        """Clear GPU cache if applicable."""
        if self._device == "cuda":
            torch.cuda.empty_cache()
        elif self._device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect_device(self) -> str:
        """Detect the best available device, respecting MOMENTO_DEVICE env var."""

        # Environment variable override
        env_device = os.environ.get("MOMENTO_DEVICE", "").strip().lower()
        if env_device in ("cuda", "mps", "cpu"):
            logger.info(f"Device override from MOMENTO_DEVICE={env_device}")
            if env_device == "cuda" and not torch.cuda.is_available():
                logger.warning("MOMENTO_DEVICE=cuda but CUDA unavailable, falling back to cpu")
                return "cpu"
            if env_device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MOMENTO_DEVICE=mps but MPS unavailable, falling back to cpu")
                return "cpu"
            return env_device

        # Auto-detect
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA available: {gpu_count} GPU(s) — {gpu_name}")
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available")
            return "mps"

        logger.info("No GPU detected, using CPU")
        return "cpu"

    def _select_dtype(self) -> torch.dtype:
        """Pick the safest dtype for the active device.

        MPS has limited float16 support in many ops, so we default to
        float32 there.  CUDA can use float16 for CLIP without issues.
        """
        if self._device == "mps":
            return torch.float32
        return torch.float32  # CLIP outputs float32 by default


# Singleton instance — importable from anywhere
device_manager = DeviceManager()
