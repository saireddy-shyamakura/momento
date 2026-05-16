"""Unit tests for validation.py — all 5 validators."""

import os
import tempfile
import pytest
from validation import (
    validate_image_path,
    validate_text_query,
    validate_folder_path,
    validate_positive_int,
    validate_choice,
    ValidationError,
)


# ── validate_image_path ─────────────────────────────────────────────

class TestValidateImagePath:
    """Tests for validate_image_path()."""

    def test_valid_jpg(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header
        is_valid, err = validate_image_path(str(img))
        assert is_valid is True
        assert err == ""

    def test_valid_png(self, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG")
        is_valid, err = validate_image_path(str(img))
        assert is_valid is True
        assert err == ""

    def test_valid_webp(self, tmp_path):
        img = tmp_path / "photo.webp"
        img.write_bytes(b"RIFF")
        is_valid, err = validate_image_path(str(img))
        assert is_valid is True
        assert err == ""

    def test_valid_bmp(self, tmp_path):
        img = tmp_path / "photo.bmp"
        img.write_bytes(b"BM")
        is_valid, err = validate_image_path(str(img))
        assert is_valid is True
        assert err == ""

    def test_empty_string(self):
        is_valid, err = validate_image_path("")
        assert is_valid is False
        assert "empty" in err.lower()

    def test_none_input(self):
        is_valid, err = validate_image_path(None)
        assert is_valid is False
        assert "empty" in err.lower()

    def test_whitespace_only(self):
        is_valid, err = validate_image_path("   ")
        assert is_valid is False
        assert "empty" in err.lower()

    def test_file_not_found(self):
        is_valid, err = validate_image_path("/nonexistent/path/image.jpg")
        assert is_valid is False
        assert "not found" in err.lower()

    def test_path_is_directory(self, tmp_path):
        is_valid, err = validate_image_path(str(tmp_path))
        assert is_valid is False
        assert "not a file" in err.lower()

    def test_unsupported_extension(self, tmp_path):
        txt = tmp_path / "notes.txt"
        txt.write_text("hello")
        is_valid, err = validate_image_path(str(txt))
        assert is_valid is False
        assert "invalid image format" in err.lower()

    def test_gif_not_supported(self, tmp_path):
        gif = tmp_path / "anim.gif"
        gif.write_bytes(b"GIF89a")
        is_valid, err = validate_image_path(str(gif))
        assert is_valid is False
        assert "invalid image format" in err.lower()

    def test_strips_whitespace(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")
        is_valid, err = validate_image_path(f"  {img}  ")
        assert is_valid is True

    def test_case_insensitive_extension(self, tmp_path):
        img = tmp_path / "photo.JPG"
        img.write_bytes(b"\xff\xd8\xff")
        is_valid, err = validate_image_path(str(img))
        assert is_valid is True


# ── validate_text_query ──────────────────────────────────────────────

class TestValidateTextQuery:
    """Tests for validate_text_query()."""

    def test_valid_query(self):
        is_valid, err = validate_text_query("a cat sitting on a couch")
        assert is_valid is True
        assert err == ""

    def test_single_word(self):
        is_valid, err = validate_text_query("sunset")
        assert is_valid is True

    def test_empty_string(self):
        is_valid, err = validate_text_query("")
        assert is_valid is False
        assert "empty" in err.lower()

    def test_none_input(self):
        is_valid, err = validate_text_query(None)
        assert is_valid is False

    def test_whitespace_only(self):
        is_valid, err = validate_text_query("   ")
        assert is_valid is False
        assert "empty" in err.lower()

    def test_max_length_boundary(self):
        text = "a" * 1000
        is_valid, err = validate_text_query(text)
        assert is_valid is True

    def test_exceeds_max_length(self):
        text = "a" * 1001
        is_valid, err = validate_text_query(text)
        assert is_valid is False
        assert "too long" in err.lower()


# ── validate_folder_path ─────────────────────────────────────────────

class TestValidateFolderPath:
    """Tests for validate_folder_path()."""

    def test_valid_folder(self, tmp_path):
        is_valid, err = validate_folder_path(str(tmp_path))
        assert is_valid is True
        assert err == ""

    def test_empty_string(self):
        is_valid, err = validate_folder_path("")
        assert is_valid is False
        assert "empty" in err.lower()

    def test_none_input(self):
        is_valid, err = validate_folder_path(None)
        assert is_valid is False

    def test_whitespace_only(self):
        is_valid, err = validate_folder_path("   ")
        assert is_valid is False
        assert "empty" in err.lower()

    def test_folder_not_found(self):
        is_valid, err = validate_folder_path("/nonexistent/folder")
        assert is_valid is False
        assert "not found" in err.lower()

    def test_path_is_file_not_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        is_valid, err = validate_folder_path(str(f))
        assert is_valid is False
        assert "not a directory" in err.lower()

    def test_strips_whitespace(self, tmp_path):
        is_valid, err = validate_folder_path(f"  {tmp_path}  ")
        assert is_valid is True


# ── validate_positive_int ────────────────────────────────────────────

class TestValidatePositiveInt:
    """Tests for validate_positive_int()."""

    def test_valid_value(self):
        is_valid, err = validate_positive_int(5, "top_k")
        assert is_valid is True
        assert err == ""

    def test_value_of_one(self):
        is_valid, err = validate_positive_int(1, "top_k")
        assert is_valid is True

    def test_max_boundary(self):
        is_valid, err = validate_positive_int(100, "top_k")
        assert is_valid is True

    def test_exceeds_max(self):
        is_valid, err = validate_positive_int(101, "top_k")
        assert is_valid is False
        assert "maximum" in err.lower()

    def test_zero(self):
        is_valid, err = validate_positive_int(0, "top_k")
        assert is_valid is False
        assert "positive" in err.lower()

    def test_negative(self):
        is_valid, err = validate_positive_int(-3, "top_k")
        assert is_valid is False
        assert "positive" in err.lower()

    def test_float_rejected(self):
        is_valid, err = validate_positive_int(3.5, "top_k")
        assert is_valid is False
        assert "integer" in err.lower()

    def test_string_rejected(self):
        is_valid, err = validate_positive_int("5", "top_k")
        assert is_valid is False
        assert "integer" in err.lower()

    def test_custom_name_in_error(self):
        is_valid, err = validate_positive_int(-1, "batch_size")
        assert "batch_size" in err


# ── validate_choice ──────────────────────────────────────────────────

class TestValidateChoice:
    """Tests for validate_choice()."""

    def test_valid_choice(self):
        is_valid, err = validate_choice("1", ["1", "2", "3", "q"])
        assert is_valid is True
        assert err == ""

    def test_valid_quit(self):
        is_valid, err = validate_choice("q", ["1", "2", "3", "q"])
        assert is_valid is True

    def test_invalid_choice(self):
        is_valid, err = validate_choice("5", ["1", "2", "3", "q"])
        assert is_valid is False
        assert "invalid choice" in err.lower()

    def test_empty_string(self):
        is_valid, err = validate_choice("", ["1", "2"])
        assert is_valid is False
        assert "empty" in err.lower()

    def test_none_input(self):
        is_valid, err = validate_choice(None, ["1", "2"])
        assert is_valid is False

    def test_whitespace_only(self):
        is_valid, err = validate_choice("   ", ["1", "2"])
        assert is_valid is False
        assert "empty" in err.lower()

    def test_strips_whitespace(self):
        is_valid, err = validate_choice("  1  ", ["1", "2"])
        assert is_valid is True

    def test_shows_valid_options_in_error(self):
        is_valid, err = validate_choice("x", ["a", "b", "c"])
        assert "a, b, c" in err


# ── ValidationError ──────────────────────────────────────────────────

class TestValidationError:
    """Tests for the ValidationError exception class."""

    def test_is_exception(self):
        assert issubclass(ValidationError, Exception)

    def test_can_raise_and_catch(self):
        with pytest.raises(ValidationError, match="test error"):
            raise ValidationError("test error")
