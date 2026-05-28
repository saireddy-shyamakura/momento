"""Unit tests for lock.py — process lock with TTL."""

import os
import time
from unittest.mock import patch

import pytest

from momento.lock import LockFile


class TestLockFile:
    """Tests LockFile acquire/release and stale detection."""

    def test_acquire_creates_file(self, tmp_path):
        lock_path = str(tmp_path / "test.lock")
        lock = LockFile(lock_path)
        assert lock.acquire() is True
        assert os.path.exists(lock_path)

    def test_release_removes_file(self, tmp_path):
        lock_path = str(tmp_path / "test.lock")
        lock = LockFile(lock_path)
        lock.acquire()
        lock.release()
        assert not os.path.exists(lock_path)

    def test_acquire_returns_false_when_held(self, tmp_path):
        lock_path = str(tmp_path / "test.lock")
        lock1 = LockFile(lock_path)
        lock2 = LockFile(lock_path)
        lock1.acquire()
        assert lock2.acquire() is False
        lock1.release()

    def test_acquire_succeeds_after_release(self, tmp_path):
        lock_path = str(tmp_path / "test.lock")
        lock = LockFile(lock_path)
        lock.acquire()
        lock.release()
        assert lock.acquire() is True
        lock.release()

    def test_ttl_stale_lock_auto_cleaned(self, tmp_path):
        """A lock with very short TTL that is expired should be auto-released."""
        lock_path = str(tmp_path / "ttl.lock")
        old_pid = 999999
        old_time = time.time() - 100
        with open(lock_path, 'w') as f:
            f.write(f"{old_pid},{old_time}")

        lock = LockFile(lock_path, ttl=10)
        assert lock.acquire() is True
        assert os.path.exists(lock_path)
        lock.release()

    def test_valid_lock_not_cleaned(self, tmp_path):
        """A lock within TTL should not be stolen."""
        lock_path = str(tmp_path / "valid.lock")
        own_pid = os.getpid()
        now = time.time()
        with open(lock_path, 'w') as f:
            f.write(f"{own_pid},{now}")

        lock = LockFile(lock_path, ttl=60)
        assert lock.acquire() is False

    def test_unparseable_lock_cleaned(self, tmp_path):
        """A lock file with garbage content should be removed."""
        lock_path = str(tmp_path / "garbage.lock")
        with open(lock_path, 'w') as f:
            f.write("not_a_number")

        lock = LockFile(lock_path)
        assert lock.acquire() is True
        lock.release()

    def test_release_multiple_times_no_error(self, tmp_path):
        """Releasing an already-released lock should not raise."""
        lock_path = str(tmp_path / "multi.lock")
        lock = LockFile(lock_path)
        lock.acquire()
        lock.release()
        lock.release()

    def test_acquire_after_stale_dead_pid(self, tmp_path):
        """Stale lock from a dead PID should be cleaned and acquired."""
        lock_path = str(tmp_path / "deadpid.lock")
        dead_pid = 2**31 - 1
        old_time = time.time() - 3600
        with open(lock_path, 'w') as f:
            f.write(f"{dead_pid},{old_time}")

        lock = LockFile(lock_path, ttl=600)
        assert lock.acquire() is True
        lock.release()

    def test_custom_ttl_constructor(self):
        """Ensure custom TTL is stored."""
        lock = LockFile("/tmp/test.lock", ttl=300)
        assert lock.ttl == 300

    @patch("os.kill")
    def test_pid_alive_detection(self, mock_kill):
        lock = LockFile("/tmp/test.lock")
        mock_kill.return_value = None
        assert lock._pid_alive(1234) is True

    @patch("os.kill")
    def test_pid_dead_detection(self, mock_kill):
        lock = LockFile("/tmp/test.lock")
        mock_kill.side_effect = ProcessLookupError()
        assert lock._pid_alive(1234) is False

    @patch("os.kill")
    def test_pid_permission_denied_returns_alive(self, mock_kill):
        lock = LockFile("/tmp/test.lock")
        mock_kill.side_effect = PermissionError()
        assert lock._pid_alive(1234) is True