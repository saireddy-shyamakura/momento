import os
import torch
import platformdirs

from .logger import get_logger

# Package-internal reference (for bundled assets only — NOT for user data)
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# User-facing data directory: ~/.local/share/momento  (XDG on Linux/Mac, %APPDATA% on Windows)
_DATA_DIR = platformdirs.user_data_dir("momento", appauthor=False)
CHROMA_DB_DIR = os.path.join(_DATA_DIR, "chroma_db")
LOG_DIR = os.path.join(_DATA_DIR, "logs")

# Keep BASE_DIR as an alias for any code still referencing it (e.g. lock file path)
BASE_DIR = _DATA_DIR

MODEL_NAME = "ViT-B/16"
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
SIMILARITY_THRESHOLD = 0.20

logger = get_logger(__name__)

def get_device() -> str:
    """
    Priority:

    1. Auto-detect CUDA (NVIDIA GPU)
    2. Auto-detect MPS (Apple Silicon)
    3. Fall back to CPU
    
    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    
    # Auto-detect CUDA
    if torch.cuda.is_available():
        device = "cuda"
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA available: {gpu_count} GPU(s) detected - {gpu_name}")
        return device
    
    # Auto-detect MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) available")
        return "mps"
    
    # Fall back to CPU
    logger.info("No GPU detected, using CPU")
    return "cpu"


DEVICE = get_device()