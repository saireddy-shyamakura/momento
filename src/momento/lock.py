"""
lock.py — Process lock with TTL support for Momento.

Provides cross-instance exclusion with automatic stale-lock cleanup
based on both PID-aliveness AND a time-to-live (TTL) threshold.

Uses atomic O_EXCL file creation to prevent TOCTOU race conditions.
"""

import os
import time
import logging

logger = logging.getLogger(__name__)

LOCK_TTL_SECONDS = 6 * 3600  # 6 hours — a lock this old is almost certainly stale


class LockFile:
    """PID-based lock file with TTL and stale-lock detection.

    Uses the atomic ``os.open(path, O_CREAT | O_EXCL | O_WRONLY)`` pattern
    to prevent two processes from acquiring the lock simultaneously.
    """

    def __init__(self, path: str, ttl: int = LOCK_TTL_SECONDS):
        """Initialize the LockFile.

        Args:
            path: Filesystem path for the lock file.
            ttl: Seconds after which a lock is considered stale,
                 even if the owning PID appears alive.
        """
        self.path = path
        self.ttl = ttl

    def _pid_alive(self, pid: int) -> bool:
        """Check if a process with the given PID is currently running.

        Args:
            pid: Process ID to check.

        Returns:
            True if the process exists and is accessible.
        """
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # PID is alive but belongs to another user
            return True

    def _read_lock_data(self) -> tuple[int, float] | None:
        """Read the lock file contents, returning (pid, timestamp).

        Returns:
            Tuple of (pid, timestamp) if the file exists and is parseable,
            otherwise None.
        """
        try:
            with open(self.path, 'r') as f:
                parts = f.read().strip().split(',')
                if len(parts) >= 2:
                    return int(parts[0]), float(parts[1])
                if len(parts) == 1:
                    return int(parts[0]), time.time()  # legacy format
        except (ValueError, OSError, EOFError):
            pass
        return None

    def _atomic_write(self) -> bool:
        """Atomically create the lock file using O_CREAT | O_EXCL.

        Returns:
            True if the file was created, False if another process holds the lock.
        """
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(f"{os.getpid()},{time.time()}")
            except OSError:
                os.close(fd)
            return True
        except FileExistsError:
            return False

    def acquire(self) -> bool:
        """Attempt to acquire the lock.

        Uses atomic file creation. If that fails, checks the existing lock
        for staleness. Only removes and retries if the lock is confirmed stale.

        Returns:
            True if the lock was successfully acquired, False otherwise.
        """
        # Fast path — attempt atomic acquire
        if self._atomic_write():
            return True

        # Lock exists — read and check for staleness
        lock_data = self._read_lock_data()
        if lock_data is None:
            # Unparseable — try to replace it
            try:
                os.remove(self.path)
            except OSError:
                return False
            return self._atomic_write()

        old_pid, timestamp = lock_data
        age = time.time() - timestamp

        if age < self.ttl and self._pid_alive(old_pid):
            # Lock is still valid
            return False

        # Stale — clean it up
        reason = ""
        if not self._pid_alive(old_pid):
            reason = f"dead PID {old_pid}"
        else:
            reason = f"lock older than {self.ttl}s ({age:.0f}s ago)"
        logger.warning(f"Removing stale lock file ({reason})")

        try:
            os.remove(self.path)
        except OSError as e:
            logger.error(f"Failed to remove stale lock file: {e}")
            return False

        return self._atomic_write()

    def release(self) -> None:
        """Release the lock if it exists."""
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError as e:
                logger.error(f"Failed to remove lock file on release: {e}")