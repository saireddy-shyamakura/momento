"""
conftest.py — Root-level pytest configuration for Momento.

Enforces CPU thread limits, provides session-scoped reusable fixtures,
and ensures memory is cleaned up between every test.

Default test execution (``pytest``) only runs ``tests/unit/`` and excludes
slow integration tests.  Run ``pytest -- -m "slow" tests/integration/``
to include heavy model-dependent tests.
"""

# ── CPU thread limits (must be imported before numpy/torch) ──────────
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import gc
from typing import Any
from pathlib import Path

import pytest
from PIL import Image


# ── Session-scoped reusable resources ────────────────────────────────

@pytest.fixture(scope="session")
def tiny_image() -> Image.Image:
    """64×64 RGB PIL image — reusable by every test, zero creation cost."""
    return Image.new("RGB", (64, 64), color="white")


@pytest.fixture(scope="session")
def tiny_image_path(tmp_path_factory) -> str:
    """Path to a tiny JPEG file on disk."""
    d: Path = tmp_path_factory.mktemp("images")
    p = d / "tiny.jpg"
    # Minimal valid JPEG: SOI + APP0 + DQT + SOF0 + SOS + EOI
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 256)
    return str(p)


@pytest.fixture(scope="session")
def small_index_path(tmp_path_factory) -> str:
    """Temporary path for a ChromaDB index.  Created once per session."""
    return str(tmp_path_factory.mktemp("chroma_db"))


# ── Per-test memory cleanup ──────────────────────────────────────────

@pytest.fixture(autouse=True)
def _cleanup_after_every_test() -> Any:
    """Run after every test: garbage-collect, clear torch caches.

    This prevents cumulative memory growth across tests.
    """
    yield
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass