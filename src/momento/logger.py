import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Import LOG_DIR lazily to avoid circular imports
# config imports logger, so we compute the path independently here
import platformdirs as _pd

log_dir = Path(_pd.user_data_dir("momento", appauthor=False)) / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Avoid duplicate handlers if imported multiple times or reloaded
if not root_logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    log_file = log_dir / "momento.log"
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def setup_logger(name: str = "momento", log_level: int = logging.INFO) -> logging.Logger:
    """Legacy wrapper for backward compatibility."""
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
