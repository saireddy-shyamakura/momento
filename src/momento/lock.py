"""
lock.py — Process lock with TTL support for Momento.

Provides cross-instance exclusion with automatic stale-lock cleanup
based on both PID-aliveness AND a time-to-live (TTL) threshold.
"""

import os
import time
import logging

logger = logging.getLogger(__name__)

LOCK_TTL_SECONDS = 6 * 3600  # 6 hours — a lock this old is almost certainly stale


class LockFile:
    """PID-based lock file with TTL and stale-lock detection."""

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

    def acquire(self) -> bool:
        """Attempt to acquire the lock.

        Automatically cleans up stale locks from:
        - Dead processes (PID no longer alive)
        - Locks older than TTL (even if PID is alive — e.g., leftover from
          a machine that hibernated)

        Returns:
            True if the lock was successfully acquired, False otherwise.
        """
        if os.path.exists(self.path):
            lock_data = self._read_lock_data()
            if lock_data is not None:
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
            else:
                logger.warning("Removing unparseable lock file")

            try:
                os.remove(self.path)
            except OSError as e:
                logger.error(f"Failed to remove stale lock file: {e}")
                return False

        try:
            with open(self.path, 'w') as f:
                f.write(f"{os.getpid()},{time.time()}")
            return True
        except OSError as e:
            logger.error(f"Failed to create lock file: {e}")
            return False

    def release(self) -> None:
        """Release the lock if it exists."""
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError as e:
                logger.error(f"Failed to remove lock file on release: {e}")