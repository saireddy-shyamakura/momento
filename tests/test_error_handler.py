"""Unit tests for error_handler.py — error accumulation, summary formatting.

Covers:
- IndexingErrorHandler add_error / add_warning
- has_errors / has_warnings
- get_error_summary / get_warning_summary
- print_summary formatting
- reset
- should_continue_indexing
- handle_fatal_error, handle_validation_error, handle_user_interrupt
"""

import sys
import pytest

from momento.error_handler import (
    IndexingErrorHandler,
    ValidationError,
    handle_fatal_error,
    handle_validation_error,
    handle_user_interrupt,
)


class TestIndexingErrorHandler:
    """Tests for IndexingErrorHandler."""

    def test_init_has_empty_errors(self):
        handler = IndexingErrorHandler()
        assert handler.errors == []
        assert handler.warnings == []
        assert handler.has_errors() is False
        assert handler.has_warnings() is False

    def test_add_error_appends_and_returns_message(self):
        handler = IndexingErrorHandler()
        try:
            raise RuntimeError("connection lost")
        except RuntimeError as e:
            msg = handler.add_error("FeatureA", e)
        assert "FeatureA" in msg
        assert "connection lost" in msg
        assert handler.has_errors() is True

    def test_add_error_multiple_accumulates(self):
        handler = IndexingErrorHandler()
        handler.add_error("A", ValueError("err1"))
        handler.add_error("B", OSError("err2"))
        assert len(handler.errors) == 2

    def test_add_warning(self):
        handler = IndexingErrorHandler()
        handler.add_warning("disk space low")
        assert handler.has_warnings() is True
        assert handler.warnings == ["disk space low"]

    def test_get_error_summary_returns_copy(self):
        handler = IndexingErrorHandler()
        handler.add_error("X", Exception("e1"))
        summary = handler.get_error_summary()
        assert summary == handler.errors
        # Modifying the returned list should not affect internal state
        summary.append("extra")
        assert len(handler.errors) == 1

    def test_get_warning_summary_returns_copy(self):
        handler = IndexingErrorHandler()
        handler.add_warning("warn1")
        summary = handler.get_warning_summary()
        assert summary == ["warn1"]

    def test_should_continue_indexing_returns_true(self):
        handler = IndexingErrorHandler()
        assert handler.should_continue_indexing() is True

    def test_reset_clears_errors_and_warnings(self):
        handler = IndexingErrorHandler()
        handler.add_error("A", Exception("e"))
        handler.add_warning("w")
        handler.reset()
        assert handler.has_errors() is False
        assert handler.has_warnings() is False
        assert handler.errors == []
        assert handler.warnings == []

    def test_print_summary_no_errors_no_warnings(self, capsys):
        handler = IndexingErrorHandler()
        handler.print_summary()
        captured = capsys.readouterr()
        assert captured.out == ""  # No output if no errors/warnings

    def test_print_summary_with_errors(self, capsys):
        handler = IndexingErrorHandler()
        handler.add_error("FeatureX", ValueError("bad data"))
        handler.add_error("FeatureY", OSError("permission denied"))
        handler.print_summary()
        captured = capsys.readouterr()
        assert "Errors encountered" in captured.out
        assert "FeatureX" in captured.out
        assert "FeatureY" in captured.out

    def test_print_summary_with_warnings(self, capsys):
        handler = IndexingErrorHandler()
        handler.add_warning("low memory")
        handler.print_summary()
        captured = capsys.readouterr()
        assert "Warnings" in captured.out
        assert "low memory" in captured.out

    def test_print_summary_truncates_many_errors(self, capsys):
        handler = IndexingErrorHandler()
        for i in range(10):
            handler.add_error(f"E{i}", Exception(f"error {i}"))
        handler.print_summary()
        captured = capsys.readouterr()
        assert "... and 5 more" in captured.out

    def test_print_summary_truncates_many_warnings(self, capsys):
        handler = IndexingErrorHandler()
        for i in range(6):
            handler.add_warning(f"warn {i}")
        handler.print_summary()
        captured = capsys.readouterr()
        assert "... and 3 more" in captured.out

    def test_has_errors_true_after_error(self):
        handler = IndexingErrorHandler()
        assert handler.has_errors() is False
        handler.add_error("A", Exception("e"))
        assert handler.has_errors() is True

    def test_has_warnings_true_after_warning(self):
        handler = IndexingErrorHandler()
        assert handler.has_warnings() is False
        handler.add_warning("w")
        assert handler.has_warnings() is True


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_is_exception_subclass(self):
        assert issubclass(ValidationError, Exception)

    def test_can_raise_and_catch(self):
        with pytest.raises(ValidationError, match="test error"):
            raise ValidationError("test error")


class TestHandleFatalError:
    """Tests for handle_fatal_error."""

    def test_exits_with_code(self):
        with pytest.raises(SystemExit) as exc_info:
            handle_fatal_error("fatal crash")
        assert exc_info.value.code == 1

    def test_exits_with_custom_code(self):
        with pytest.raises(SystemExit) as exc_info:
            handle_fatal_error("custom exit", exit_code=42)
        assert exc_info.value.code == 42

    def test_prints_error_message(self, capsys):
        with pytest.raises(SystemExit):
            handle_fatal_error("something terrible happened")
        captured = capsys.readouterr()
        assert "something terrible happened" in captured.err


class TestHandleValidationError:
    """Tests for handle_validation_error."""

    def test_raises_validation_error(self):
        with pytest.raises(ValidationError, match="invalid input"):
            handle_validation_error("invalid input")

    def test_prints_message(self, capsys):
        try:
            handle_validation_error("bad path")
        except ValidationError:
            pass
        captured = capsys.readouterr()
        assert "bad path" in captured.out


class TestHandleUserInterrupt:
    """Tests for handle_user_interrupt."""

    def test_exits_with_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            handle_user_interrupt()
        assert exc_info.value.code == 0