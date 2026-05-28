"""
ocr.py — OCR text extraction for Momento.

Extracts text from images using EasyOCR, which can then be embedded
via CLIP's text encoder for text-in-image search.

Requires: easyocr
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import easyocr
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False

from .device import device_manager

_ocr_reader: Optional["easyocr.Reader"] = None


def is_available() -> bool:
    return _HAS_OCR


def get_reader(languages: List[str] | None = None) -> "easyocr.Reader":
    global _ocr_reader
    if not _HAS_OCR:
        raise RuntimeError("easyocr required. Install: pip install easyocr")
    if _ocr_reader is None:
        langs = languages or ["en"]
        gpu_enabled = (device_manager.device == "cuda")
        logger.info(f"Loading EasyOCR reader for languages: {langs} (gpu={gpu_enabled})")
        _ocr_reader = easyocr.Reader(langs, gpu=gpu_enabled)
    return _ocr_reader


def extract_text(
    image_path: str,
    languages: List[str] | None = None,
    min_confidence: float = 0.3,
) -> str:
    """
    Extract text from an image file using OCR.

    Args:
        image_path: Path to the image file.
        languages: List of language codes (default: ['en']).
        min_confidence: Minimum confidence to include a text detection.

    Returns:
        Concatenated extracted text, or empty string if no text found.
    """
    if not _HAS_OCR:
        raise RuntimeError("easyocr required. Install: pip install easyocr")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    reader = get_reader(languages)
    results = reader.readtext(image_path)

    texts = []
    for bbox, text, conf in results:
        if conf >= min_confidence and text.strip():
            texts.append(text.strip())

    combined = " ".join(texts)
    if combined:
        logger.debug(f"OCR extracted {len(texts)} text regions from {os.path.basename(image_path)}")
    return combined


def extract_text_from_pil(
    img: "Image.Image",
    languages: List[str] | None = None,
    min_confidence: float = 0.3,
) -> str:
    """Extract text from an in-memory PIL Image."""
    if not _HAS_OCR:
        raise RuntimeError("easyocr required. Install: pip install easyocr")
    import numpy as np

    reader = get_reader(languages)
    img_array = np.array(img.convert("RGB"))
    results = reader.readtext(img_array)

    texts = []
    for bbox, text, conf in results:
        if conf >= min_confidence and text.strip():
            texts.append(text.strip())

    return " ".join(texts)
