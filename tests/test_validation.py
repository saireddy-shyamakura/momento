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


# ── Property-Based Tests (Hypothesis) ───────────────────────────────
# Validates: Requirements 1.2, 1.3, 1.6

import string
from hypothesis import given, assume, settings
from hypothesis import strategies as st


# Helpers to call all five validators with a single string argument.
# validate_positive_int and validate_choice need special handling since
# they have different signatures.

def _call_all_validators_with_string(s: str):
    """Call all five validators and return their results as a list."""
    results = []
    results.append(validate_image_path(s))
    results.append(validate_text_query(s))
    results.append(validate_folder_path(s))
    # validate_positive_int expects an int; pass the string directly to
    # exercise the type-rejection branch.
    results.append(validate_positive_int(s, "value"))
    # validate_choice: use a fixed options list; the string may or may not be in it.
    results.append(validate_choice(s, ["a", "b", "c"]))
    return results


class TestValidatorProperties:
    """
    Property-based tests for the five validators.

    **Validates: Requirements 1.2, 1.3, 1.6**
    """

    # ── Property 1: Idempotence ──────────────────────────────────────
    @settings(max_examples=200)
    @given(s=st.text())
    def test_validator_idempotence(self, s: str):
        """
        Calling any validator twice with the same input must return the
        same (bool, str) tuple both times.

        **Validates: Requirements 1.6**
        """
        first = _call_all_validators_with_string(s)
        second = _call_all_validators_with_string(s)
        assert first == second, (
            f"Validators are not idempotent for input {s!r}: "
            f"first={first}, second={second}"
        )

    # ── Property 2: Return-value shape ──────────────────────────────
    @settings(max_examples=200)
    @given(s=st.text())
    def test_return_value_shape(self, s: str):
        """
        Every validator must return a 2-tuple whose first element is a
        bool and whose second element is a str.

        **Validates: Requirements 1.2, 1.3**
        """
        for result in _call_all_validators_with_string(s):
            assert isinstance(result, tuple) and len(result) == 2, (
                f"Expected 2-tuple, got {result!r}"
            )
            ok, msg = result
            assert isinstance(ok, bool), f"First element must be bool, got {type(ok)}"
            assert isinstance(msg, str), f"Second element must be str, got {type(msg)}"

    # ── Property 3: Valid inputs → (True, "") ───────────────────────
    @settings(max_examples=200)
    @given(s=st.text(min_size=1, max_size=1000))
    def test_valid_inputs_return_true_empty(self, s: str):
        """
        For validate_text_query, any non-empty string of length ≤ 1000
        that is not purely whitespace must return (True, "").

        **Validates: Requirements 1.2**
        """
        assume(s.strip() != "")          # filter out whitespace-only strings
        assume(len(s.strip()) <= 1000)   # stay within the documented limit

        ok, msg = validate_text_query(s)
        assert ok is True, f"Expected True for {s!r}, got ok={ok}, msg={msg!r}"
        assert msg == "", f"Expected empty error string for {s!r}, got {msg!r}"

    # ── Property 4: Invalid inputs → (False, non-empty str) ─────────
    @settings(max_examples=200)
    @given(s=st.text(max_size=0))
    def test_invalid_inputs_return_false_nonempty_empty_string(self, s: str):
        """
        An empty string is invalid for every string-accepting validator;
        each must return (False, <non-empty error string>).

        **Validates: Requirements 1.3**
        """
        # s is always "" here (max_size=0)
        for validator, args in [
            (validate_text_query, (s,)),
            (validate_folder_path, (s,)),
            (validate_image_path, (s,)),
            (validate_choice, (s, ["a", "b"])),
        ]:
            ok, msg = validator(*args)
            assert ok is False, (
                f"{validator.__name__} returned True for empty input"
            )
            assert isinstance(msg, str) and len(msg) > 0, (
                f"{validator.__name__} returned empty error message for invalid input"
            )

    @settings(max_examples=200)
    @given(value=st.one_of(st.floats(allow_nan=False), st.text(), st.none()))
    def test_invalid_inputs_return_false_nonempty_non_int(self, value):
        """
        validate_positive_int must return (False, non-empty str) for any
        non-integer input.

        **Validates: Requirements 1.3**
        """
        assume(not isinstance(value, int))  # exclude actual ints
        ok, msg = validate_positive_int(value, "value")
        assert ok is False, (
            f"validate_positive_int returned True for non-int {value!r}"
        )
        assert isinstance(msg, str) and len(msg) > 0, (
            f"validate_positive_int returned empty error for non-int {value!r}"
        )

    @settings(max_examples=200)
    @given(n=st.integers().filter(lambda x: x <= 0 or x > 100))
    def test_invalid_inputs_return_false_nonempty_out_of_range_int(self, n: int):
        """
        validate_positive_int must return (False, non-empty str) for
        integers that are ≤ 0 or > 100.

        **Validates: Requirements 1.3**
        """
        ok, msg = validate_positive_int(n, "value")
        assert ok is False, (
            f"validate_positive_int returned True for out-of-range int {n}"
        )
        assert isinstance(msg, str) and len(msg) > 0, (
            f"validate_positive_int returned empty error for out-of-range int {n}"
        )

    @settings(max_examples=200)
    @given(
        s=st.text(min_size=1),
        options=st.lists(st.text(min_size=1), min_size=1, max_size=5),
    )
    def test_invalid_choice_returns_false_nonempty(self, s: str, options: list):
        """
        validate_choice must return (False, non-empty str) when the
        stripped choice is not in the options list.

        **Validates: Requirements 1.3**
        """
        assume(s.strip() not in options)  # ensure it's genuinely invalid
        assume(s.strip() != "")           # non-empty after stripping

        ok, msg = validate_choice(s, options)
        assert ok is False, (
            f"validate_choice returned True for {s!r} not in {options}"
        )
        assert isinstance(msg, str) and len(msg) > 0, (
            f"validate_choice returned empty error for invalid choice {s!r}"
        )
