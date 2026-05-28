"""
augment.py — Image augmentation pipeline for multi-embedding generation.

Produces multiple views of an image to improve search recall by capturing
different visual aspects (flipped, cropped, brightness-adjusted, rotated).
"""

from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# Each augmentation is a (name_suffix, transform_fn) tuple
AugmentedView = Tuple[str, Image.Image]


def _horizontal_flip(img: Image.Image) -> Image.Image:
    """Mirror the image horizontally."""
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def _center_crop(img: Image.Image, ratio: float = 0.75) -> Image.Image:
    """Crop the center portion of the image."""
    w, h = img.size
    new_w, new_h = int(w * ratio), int(h * ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def _brightness_jitter(img: Image.Image, factor: float = 1.3) -> Image.Image:
    """Adjust brightness."""
    return ImageEnhance.Brightness(img).enhance(factor)


def _contrast_jitter(img: Image.Image, factor: float = 1.3) -> Image.Image:
    """Adjust contrast."""
    return ImageEnhance.Contrast(img).enhance(factor)


def _rotate(img: Image.Image, angle: float = 15.0) -> Image.Image:
    """Rotate the image by a small angle, filling empty areas with edge pixels."""
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=None)


def _grayscale(img: Image.Image) -> Image.Image:
    """Convert to grayscale then back to RGB (retains 3-channel format)."""
    return img.convert("L").convert("RGB")


# Registry of augmentations: (suffix_name, transform_function)
AUGMENTATION_REGISTRY: List[Tuple[str, callable]] = [
    ("flip", _horizontal_flip),
    ("crop", _center_crop),
    ("bright", _brightness_jitter),
    ("contrast", _contrast_jitter),
    ("rotate", _rotate),
]


def generate_augmentations(
    img: Image.Image,
    augmentations: List[Tuple[str, callable]] | None = None,
) -> List[AugmentedView]:
    """
    Generate augmented views of an image.

    Args:
        img: Source PIL Image (should already be RGB).
        augmentations: Optional list of (name, fn) tuples. Defaults to
                       AUGMENTATION_REGISTRY.

    Returns:
        List of (suffix, augmented_image) tuples.  The original image
        is **not** included — the caller should handle it separately.
    """
    if augmentations is None:
        augmentations = AUGMENTATION_REGISTRY

    views: List[AugmentedView] = []
    for name, transform_fn in augmentations:
        try:
            augmented = transform_fn(img.copy())
            views.append((name, augmented))
        except Exception as e:
            logger.warning(f"Augmentation '{name}' failed: {e}")

    return views
