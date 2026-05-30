"""Unit tests for storage_manager.py — storage management.

Tests StorageManager methods including:
- format_size
- clear_cache / clear_logs
- optimize_database
- backup / restore / export / import
- _validate_sql_statement allowlist
- cleanup_old_logs
"""
import os
import pytest
from unittest.mock import patch


def _import_storage_manager():
    from momento.storage_manager import StorageManager, _get_dir_size, get_storage_manager
    return StorageManager, _get_dir_size, get_storage_manager


class TestFormatSize:
    """Human-readable size formatting."""

    def test_format_bytes(self):
        StorageManager, _, _ = _import_storage_manager()
        assert StorageManager.format_size(0) == "0.00 B"
        assert StorageManager.format_size(1023) == "1023.00 B"
        assert StorageManager.format_size(1024) == "1.00 KB"
        assert StorageManager.format_size(1048576) == "1.00 MB"
        assert StorageManager.format_size(1073741824) == "1.00 GB"


class TestClearCache:
    """Clearing embedding cache."""

    def test_clear_cache_no_cache_dir(self):
        StorageManager, _, _ = _import_storage_manager()
        with patch("momento.storage_manager.EMBEDDING_CACHE_DIR", "/nonexistent/path"):
            count = StorageManager.clear_cache()
            assert count == 0

    def test_clear_cache_with_files(self, tmp_path):
        StorageManager, _, _ = _import_storage_manager()
        cache_dir = os.path.join(tmp_path, "cache")
        os.makedirs(cache_dir)
        for fname in ["a.npz", "b.pkl", "c.access"]:
            open(os.path.join(cache_dir, fname), "w").close()
        open(os.path.join(cache_dir, "other.txt"), "w").close()

        with patch("momento.storage_manager.EMBEDDING_CACHE_DIR", cache_dir):
            count = StorageManager.clear_cache()
            assert count == 3


class TestClearLogs:
    """Clearing log files."""

    def test_clear_logs_no_log_dir(self):
        StorageManager, _, _ = _import_storage_manager()
        with patch("momento.storage_manager.LOG_DIR", "/nonexistent/path"):
            count = StorageManager.clear_logs()
            assert count == 0

    def test_clear_logs_with_files(self, tmp_path):
        StorageManager, _, _ = _import_storage_manager()
        log_dir = os.path.join(tmp_path, "logs")
        os.makedirs(log_dir)
        open(os.path.join(log_dir, "momento.log"), "w").close()
        open(os.path.join(log_dir, "error.log"), "w").close()

        with patch("momento.storage_manager.LOG_DIR", log_dir):
            count = StorageManager.clear_logs()
            assert count == 2


class TestOptimizeDatabase:
    """Database optimization."""

    def test_optimize_no_db(self):
        StorageManager, _, _ = _import_storage_manager()
        with patch("momento.storage_manager.CHROMA_DB_DIR", "/nonexistent"):
            assert StorageManager.optimize_database() is False

    def test_optimize_success(self, tmp_path):
        StorageManager, _, _ = _import_storage_manager()
        db_dir = os.path.join(tmp_path, "chroma_db")
        os.makedirs(db_dir)
        db_file = os.path.join(db_dir, "chroma.sqlite3")
        import sqlite3
        conn = sqlite3.connect(db_file)
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()

        with patch("momento.storage_manager.CHROMA_DB_DIR", db_dir):
            result = StorageManager.optimize_database()
            assert result is True


class TestBackupRestore:
    """Backup and restore operations."""

    def test_backup_no_db(self):
        StorageManager, _, _ = _import_storage_manager()
        with patch("momento.storage_manager.CHROMA_DB_DIR", "/nonexistent"):
            result = StorageManager.backup_database()
            assert result is None

    def test_backup_success(self, tmp_path):
        StorageManager, _, _ = _import_storage_manager()
        db_dir = os.path.join(tmp_path, "chroma_db")
        os.makedirs(db_dir)
        db_file = os.path.join(db_dir, "chroma.sqlite3")
        with open(db_file, "w") as f:
            f.write("test db content")

        backup_dir = os.path.join(tmp_path, "backups")
        with patch("momento.storage_manager.CHROMA_DB_DIR", db_dir):
            result = StorageManager.backup_database(backup_dir=str(backup_dir))
            assert result is not None
            assert os.path.exists(result)

    def test_restore_no_backup(self):
        StorageManager, _, _ = _import_storage_manager()
        result = StorageManager.restore_database("/nonexistent/backup.db")
        assert result is False

    def test_restore_success(self, tmp_path):
        StorageManager, _, _ = _import_storage_manager()
        db_dir = os.path.join(tmp_path, "chroma_db")
        os.makedirs(db_dir)
        db_file = os.path.join(db_dir, "chroma.sqlite3")
        with open(db_file, "w") as f:
            f.write("original")

        backup_file = os.path.join(tmp_path, "backup.db")
        with open(backup_file, "w") as f:
            f.write("backup content")

        with patch("momento.storage_manager.CHROMA_DB_DIR", db_dir):
            result = StorageManager.restore_database(str(backup_file))
            assert result is True


class TestValidateSqlStatement:
    """SQL statement validation for import safety."""

    def test_allowed_insert(self):
        StorageManager, _, _ = _import_storage_manager()
        assert StorageManager._validate_sql_statement("INSERT INTO test VALUES (1)") is True

    def test_allowed_create(self):
        StorageManager, _, _ = _import_storage_manager()
        assert StorageManager._validate_sql_statement("CREATE TABLE test (id INTEGER)") is True

    def test_blocked_drop(self):
        StorageManager, _, _ = _import_storage_manager()
        assert StorageManager._validate_sql_statement("DROP TABLE test") is False

    def test_blocked_attach(self):
        StorageManager, _, _ = _import_storage_manager()
        assert StorageManager._validate_sql_statement("ATTACH DATABASE '/etc/passwd' AS evil") is False


class TestGetDirSize:
    """Directory size calculation."""

    def test_empty_dir(self, tmp_path):
        _, _get_dir_size, _ = _import_storage_manager()
        size = _get_dir_size(str(tmp_path))
        assert size == 0

    def test_dir_with_files(self, tmp_path):
        _, _get_dir_size, _ = _import_storage_manager()
        f1 = os.path.join(tmp_path, "a.txt")
        with open(f1, "wb") as f:
            f.write(b"x" * 100)
        f2 = os.path.join(tmp_path, "b.txt")
        with open(f2, "wb") as f:
            f.write(b"y" * 200)

        size = _get_dir_size(str(tmp_path))
        assert size == 300