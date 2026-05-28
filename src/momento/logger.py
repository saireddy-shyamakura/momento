"""
logger.py — Structured logging for Momento.

Provides both text and JSON log formats with support for:
- Rotating file handler (always on)
- Console output (text or JSON)
- Structured event logging with duration tracking
"""

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler

# Import LOG_DIR lazily to avoid circular imports
# config imports logger, so we compute the path independently here
import platformdirs as _pd

log_dir = Path(_pd.user_data_dir("momento", appauthor=False)) / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

text_formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Global log format: "text" or "json"
_LOG_FORMAT: str = "text"


class JsonFormatter(logging.Formatter):
    """JSON log formatter that outputs structured log records.

    Produces JSON objects with keys: timestamp, level, module, event,
    duration_ms, vector_count, error_stack, and message.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["error"] = str(record.exc_info[0].__name__)
            log_entry["error_stack"] = "".join(
                traceback.format_exception(*record.exc_info)
            )

        # Include extra fields passed via extra={}
        for key in ("event", "duration_ms", "vector_count"):
            if hasattr(record, key):
                value = getattr(record, key)
                if value is not None:
                    log_entry[key] = value

        return json.dumps(log_entry, default=str)


def set_log_format(fmt: str) -> None:
    """Set the global log format to 'text' or 'json'.

    Args:
        fmt: Either 'text' or 'json'.

    Raises:
        ValueError: If fmt is not 'text' or 'json'.
    """
    global _LOG_FORMAT
    if fmt not in ("text", "json"):
        raise ValueError(f"Invalid log format: {fmt!r}. Must be 'text' or 'json'.")
    _LOG_FORMAT = fmt
    # Update console handler formatter
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(_get_formatter())


def _get_formatter() -> logging.Formatter:
    """Return the appropriate formatter based on current log format."""
    if _LOG_FORMAT == "json":
        return JsonFormatter()
    return text_formatter


root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Avoid duplicate handlers if imported multiple times or reloaded
if not root_logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_get_formatter())

    log_file = log_dir / "momento.log"
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(text_formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def setup_logger(name: str = "momento", log_level: int = logging.INFO) -> logging.Logger:
    """Legacy wrapper for backward compatibility.

    Args:
        name: Logger name.
        log_level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class MetricLogger:
    """Helper for structured event logging with timing and metrics.

    Usage::

        with MetricLogger(__name__, "index_image", vector_count=5) as m:
            # ... do work ...
            # On exit, logs: {"event": "index_image", "duration_ms": 123.4, "vector_count": 5, ...}
    """

    def __init__(self, name: str, event: str, **extra_fields: Any):
        self._logger = logging.getLogger(name)
        self._event = event
        self._extra = extra_fields
        self._start: Optional[float] = None

    def __enter__(self) -> "MetricLogger":
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        duration_ms = (time.time() - self._start) * 1000 if self._start else 0.0
        extra = {
            "event": self._event,
            "duration_ms": round(duration_ms, 2),
            **self._extra,
        }
        if exc_type is not None:
            self._logger.error(
                f"{self._event} failed: {exc_val}",
                extra=extra,
                exc_info=(exc_type, exc_val, exc_tb),
            )
            return True  # Don't suppress the exception
        else:
            self._logger.info(f"{self._event} completed", extra=extra)
            return True