"""Unit tests for logger.py — structured logging.

Tests set_log_format, JsonFormatter, MetricLogger, and get_logger.
"""
import json
import logging
import pytest


def _import_logger():
    from momento.logger import set_log_format, JsonFormatter, MetricLogger, get_logger
    return set_log_format, JsonFormatter, MetricLogger, get_logger


class TestSetLogFormat:
    """Log format switching."""

    def test_set_text_format(self):
        set_log_format, _, _, _ = _import_logger()
        try:
            set_log_format("text")
        except Exception as e:
            pytest.fail(f"set_log_format('text') raised: {e}")

    def test_set_json_format(self):
        set_log_format, _, _, _ = _import_logger()
        try:
            set_log_format("json")
        except Exception as e:
            pytest.fail(f"set_log_format('json') raised: {e}")

    def test_invalid_format(self):
        set_log_format, _, _, _ = _import_logger()
        with pytest.raises(ValueError, match="Invalid log format"):
            set_log_format("invalid")


class TestJsonFormatter:
    """JSON log formatter output."""

    def test_json_formatter_output(self):
        _, JsonFormatter, _, _ = _import_logger()
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["module"] == "test"
        assert parsed["message"] == "hello"

    def test_json_formatter_with_exception(self):
        _, JsonFormatter, _, _ = _import_logger()
        formatter = JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="test.py",
                lineno=1, msg="failed", args=(), exc_info=exc_info,
            )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "error" in parsed
        assert "error_stack" in parsed


class TestMetricLogger:
    """MetricLogger context manager."""

    def test_metric_logger_success(self):
        _, _, MetricLogger, _ = _import_logger()
        with MetricLogger("test", "test_event"):
            pass

    def test_metric_logger_with_extra(self):
        _, _, MetricLogger, _ = _import_logger()
        with MetricLogger("test", "test_event", vector_count=5):
            pass

    def test_metric_logger_exception(self):
        _, _, MetricLogger, _ = _import_logger()
        try:
            with MetricLogger("test", "fail_event"):
                raise RuntimeError("something broke")
        except RuntimeError:
            pass


class TestGetLogger:
    """get_logger returns a logger."""

    def test_get_logger(self):
        _, _, _, get_logger = _import_logger()
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"