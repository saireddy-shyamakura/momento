"""Unit tests for cache.py — embedding cache with LRU eviction."""

import os
import time
import numpy as np

from momento.cache import (
    load_embedding,
    save_embedding,
    clear_cache,
    get_cache_size_mb,
    get_cache_entry_count,
)


class TestEmbeddingCache:
    """Tests for per-file embedding cache."""

    def test_save_and_load_embedding(self, tmp_path, monkeypatch):
        monkeypatch.setattr("momento.cache.EMBEDDING_CACHE_DIR", str(tmp_path / "cache"))
        test_path = str(tmp_path / "test_img.png")
        # Create a dummy "source file" with mtime
        with open(test_path, 'wb') as f:
            f.write(b"dummy")
        emb = np.random.randn(512).astype(np.float32)
        save_embedding(test_path, emb)
        loaded = load_embedding(test_path)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, emb)

    def test_load_returns_none_for_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("momento.cache.EMBEDDING_CACHE_DIR", str(tmp_path / "cache"))
        result = load_embedding("/nonexistent/path.jpg")
        assert result is None

    def test_cache_miss_after_mtime_change(self, tmp_path, monkeypatch):
        monkeypatch.setattr("momento.cache.EMBEDDING_CACHE_DIR", str(tmp_path / "cache"))
        test_path = str(tmp_path / "test_img.png")
        with open(test_path, 'wb') as f:
            f.write(b"dummy")
        emb = np.random.randn(512).astype(np.float32)
        save_embedding(test_path, emb)

        # Modify the source file to change its mtime
        time.sleep(0.1)
        with open(test_path, 'wb') as f:
            f.write(b"modified")
        # Should now miss cache
        loaded = load_embedding(test_path)
        assert loaded is None

    def test_clear_cache(self, tmp_path, monkeypatch):
        cache_dir = str(tmp_path / "cache")
        monkeypatch.setattr("momento.cache.EMBEDDING_CACHE_DIR", cache_dir)
        test_path = str(tmp_path / "test_img.png")
        with open(test_path, 'wb') as f:
            f.write(b"dummy")
        save_embedding(test_path, np.random.randn(512).astype(np.float32))
        assert get_cache_entry_count() >= 1
        clear_cache()
        assert get_cache_entry_count() == 0

    def test_get_cache_size_mb_zero_on_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("momento.cache.EMBEDDING_CACHE_DIR", str(tmp_path / "empty"))
        assert get_cache_size_mb() == 0.0

    def test_get_cache_entry_count_zero_on_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("momento.cache.EMBEDDING_CACHE_DIR", str(tmp_path / "empty"))
        assert get_cache_entry_count() == 0

    def test_save_overwrites_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("momento.cache.EMBEDDING_CACHE_DIR", str(tmp_path / "cache"))
        test_path = str(tmp_path / "test_img.png")
        with open(test_path, 'wb') as f:
            f.write(b"dummy")
        emb1 = np.ones(512, dtype=np.float32)
        emb2 = np.ones(512, dtype=np.float32) * 2
        save_embedding(test_path, emb1)
        save_embedding(test_path, emb2)
        loaded = load_embedding(test_path)
        np.testing.assert_array_almost_equal(loaded, emb2)

    def test_cache_handles_corrupted_file_gracefully(self, tmp_path, monkeypatch):
        """A corrupted .npz file should be treated as a cache miss."""
        cache_dir = str(tmp_path / "cache")
        monkeypatch.setattr("momento.cache.EMBEDDING_CACHE_DIR", cache_dir)
        test_path = str(tmp_path / "test_img.png")
        with open(test_path, 'wb') as f:
            f.write(b"dummy")

        # Write a corrupted cache file manually
        from momento.cache import _cache_file_for
        cache_file = _cache_file_for(test_path)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            f.write(b"not a valid npz")

        # Should not raise, should return None
        loaded = load_embedding(test_path)
        assert loaded is None