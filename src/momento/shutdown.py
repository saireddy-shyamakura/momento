"""
shutdown.py — Graceful shutdown handling for Momento.

Provides a global shutdown flag and signal handlers that can be
imported without circular dependencies.
"""

import sys
import signal
import logging

logger = logging.getLogger(__name__)

# Global flag checked by long-running operations to abort cleanly.
# Set to True by signal handlers.
_shutdown_requested: bool = False


def _signal_handler(signum: int, frame) -> None:
    """Handle SIGINT / SIGTERM by requesting graceful shutdown.

    The first signal sets a flag that long-running operations check.
    A second signal forces immediate exit.
    """
    global _shutdown_requested
    signum_name = signal.Signals(signum).name
    if _shutdown_requested:
        print(f"\n⚠️  {signum_name} received again — forcing immediate exit.")
        sys.exit(1)
    _shutdown_requested = True
    print(f"\n⚠️  {signum_name} received — finishing current batch, then exiting...")
    print("   (Press Ctrl+C again to force exit)")
    logger.warning(f"Graceful shutdown requested via {signum_name}")


def is_shutdown_requested() -> bool:
    """Return True if a shutdown signal has been received.

    Long-running operations should check this periodically and abort
    cleanly if True.
    """
    return _shutdown_requested


def install_signal_handlers() -> None:
    """Install SIGINT and SIGTERM handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def reset_shutdown_flag() -> None:
    """Reset the shutdown flag (used at start of a fresh run)."""
    global _shutdown_requested
    _shutdown_requested = False