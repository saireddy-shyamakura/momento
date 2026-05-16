import logging
import sys
from pathlib import Path


# Auto-configure root logger on import
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

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
    file_handler = logging.FileHandler(log_file)
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
