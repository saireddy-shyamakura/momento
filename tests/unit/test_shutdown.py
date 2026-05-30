"""Unit tests for shutdown.py — graceful shutdown handling.

Tests the global shutdown flag, signal handlers,
is_shutdown_requested, and reset_shutdown_flag.
"""
import signal
import pytest


def _import_shutdown():
    from momento.shutdown import (
        is_shutdown_requested, reset_shutdown_flag,
        install_signal_handlers, _shutdown_requested,
    )
    return is_shutdown_requested, reset_shutdown_flag, install_signal_handlers, _shutdown_requested


class TestShutdownFlag:
    """Shutdown flag read/write."""

    def test_default_is_false(self):
        is_shutdown_requested, _, _, _ = _import_shutdown()
        assert is_shutdown_requested() is False

    def test_reset_sets_false(self):
        is_shutdown_requested, reset_shutdown_flag, _, _ = _import_shutdown()
        import momento.shutdown as mod
        mod._shutdown_requested = True
        assert is_shutdown_requested() is True
        reset_shutdown_flag()
        assert is_shutdown_requested() is False

    def test_reset_when_already_false(self):
        _, reset_shutdown_flag, _, _ = _import_shutdown()
        reset_shutdown_flag()


class TestShutdownSignalHandlers:
    """Signal handler installation."""

    def test_install_signal_handlers(self):
        _, _, install_signal_handlers, _ = _import_shutdown()
        try:
            install_signal_handlers()
        except Exception as e:
            pytest.fail(f"install_signal_handlers raised: {e}")