"""Unit tests for augment.py — image augmentation pipeline."""

import io
import numpy as np
from PIL import Image
import pytest

from momento.augment import (
    generate_augmentations,
    _horizontal_flip,
    _center_crop,
    _brightness_jitter,
    _contrast_jitter,
    _rotate,
    _grayscale,
    AUGMENTATION_REGISTRY,
)


def _make_rgb_image(width: int = 64, height: int = 64) -> Image.Image:
    """Create a simple RGB test image."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _make_grayscale_image(width: int = 64, height: int = 64) -> Image.Image:
    """Create a simple grayscale test image."""
    arr = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    return Image.fromarray(arr, "L")


class TestAugmentationTransforms:
    """Verify each individual transform produces valid output."""

    def test_horizontal_flip_output_type(self):
        img = _make_rgb_image()
        result = _horizontal_flip(img)
        assert isinstance(result, Image.Image)

    def test_horizontal_flip_size_preserved(self):
        img = _make_rgb_image(64, 32)
        result = _horizontal_flip(img)
        assert result.size == (64, 32)

    def test_center_crop_smaller(self):
        img = _make_rgb_image(100, 100)
        result = _center_crop(img, ratio=0.75)
        assert result.size == (75, 75)

    def test_center_crop_full_ratio(self):
        img = _make_rgb_image(100, 100)
        result = _center_crop(img, ratio=1.0)
        assert result.size == (100, 100)

    def test_brightness_jitter_output_type(self):
        img = _make_rgb_image()
        result = _brightness_jitter(img)
        assert isinstance(result, Image.Image)

    def test_contrast_jitter_output_type(self):
        img = _make_rgb_image()
        result = _contrast_jitter(img)
        assert isinstance(result, Image.Image)

    def test_rotate_output_type(self):
        img = _make_rgb_image()
        result = _rotate(img, angle=15.0)
        assert isinstance(result, Image.Image)

    def test_rotate_same_size(self):
        img = _make_rgb_image(64, 64)
        result = _rotate(img, angle=15.0)
        assert result.size == (64, 64)

    def test_grayscale_stays_rgb(self):
        img = _make_rgb_image()
        result = _grayscale(img)
        assert result.mode == "RGB"

    def test_grayscale_from_grayscale_stays_rgb(self):
        img = _make_grayscale_image()
        result = _grayscale(img)
        assert result.mode == "RGB"


class TestGenerateAugmentations:
    """Tests for the main generate_augmentations function."""

    def test_returns_all_default_augmentations(self):
        img = _make_rgb_image()
        views = generate_augmentations(img)
        # Default registry has 5 transforms
        assert len(views) == 5

    def test_each_view_is_tuple_of_str_and_image(self):
        img = _make_rgb_image()
        views = generate_augmentations(img)
        for name, view in views:
            assert isinstance(name, str)
            assert isinstance(view, Image.Image)

    def test_suffixes_match_registry_names(self):
        img = _make_rgb_image()
        views = generate_augmentations(img)
        names = [n for n, _ in views]
        expected = [n for n, _ in AUGMENTATION_REGISTRY]
        assert names == expected

    def test_original_image_not_modified(self):
        img = _make_rgb_image()
        orig_pixels = list(img.getdata())
        _ = generate_augmentations(img)
        # Original image should be unchanged
        assert list(img.getdata()) == orig_pixels

    def test_custom_augmentations_list(self):
        img = _make_rgb_image()
        custom = [("custom_flip", _horizontal_flip)]
        views = generate_augmentations(img, augmentations=custom)
        assert len(views) == 1
        assert views[0][0] == "custom_flip"

    def test_empty_augmentations_list(self):
        img = _make_rgb_image()
        views = generate_augmentations(img, augmentations=[])
        assert len(views) == 0

    def test_invalid_transform_skipped(self):
        def broken(_img):
            raise ValueError("broken")
        img = _make_rgb_image()
        custom = [("good", _horizontal_flip), ("bad", broken), ("good2", _horizontal_flip)]
        views = generate_augmentations(img, augmentations=custom)
        assert len(views) == 2

    def test_all_views_have_same_mode(self):
        img = _make_rgb_image()
        views = generate_augmentations(img)
        for _, v in views:
            assert v.mode == "RGB"

    def test_all_views_have_same_size_as_input(self):
        img = _make_rgb_image(128, 64)
        views = generate_augmentations(img)
        for _, v in views:
            assert v.size == (128, 64), f"Expected (128, 64) but got {v.size}"