import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
MODEL_NAME = "ViT-B/16"
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
SIMILARITY_THRESHOLD = 0.20

from logger import get_logger

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