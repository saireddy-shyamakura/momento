"""
embedding — Unified embedding abstraction for Momento.

All downstream code should import from here, never from clip_backend directly.
"""
from .base import EmbeddingBackend
from .clip_backend import ClipBackend

__all__ = ["EmbeddingBackend", "ClipBackend"]