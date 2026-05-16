"""
tests/test_output.py — Example-based tests for output.py

Covers format_bar() and render_result() from src/momento/output.py.
Requirements: 10.1, 10.2, 10.3
"""

import os
import pytest

from momento.output import format_bar, render_result

FILLED = "█"
EMPTY = "░"


# ---------------------------------------------------------------------------
# format_bar() example-based tests
# ---------------------------------------------------------------------------

def test_format_bar_zero_returns_all_empty():
    """format_bar(0.0) should return 10 empty block characters."""
    result = format_bar(0.0)
    assert result == EMPTY * 10


def test_format_bar_one_returns_all_filled():
    """format_bar(1.0) should return 10 filled block characters."""
    result = format_bar(1.0)
    assert result == FILLED * 10


def test_format_bar_partial_score_length():
    """format_bar(0.62) should return a string of exactly 10 characters."""
    result = format_bar(0.62)
    assert len(result) == 10


def test_format_bar_partial_score_filled_count():
    """format_bar(0.62) should have round(0.62 * 10) == 6 filled characters."""
    result = format_bar(0.62)
    assert result.count(FILLED) == round(0.62 * 10)  # == 6


def test_format_bar_partial_score_empty_count():
    """format_bar(0.62) should have 10 - 6 == 4 empty characters."""
    result = format_bar(0.62)
    assert result.count(EMPTY) == 10 - round(0.62 * 10)  # == 4


def test_format_bar_only_block_characters():
    """format_bar result should contain only FILLED and EMPTY characters."""
    for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = format_bar(score)
        for ch in result:
            assert ch in (FILLED, EMPTY), f"Unexpected character {ch!r} in bar for score={score}"


# ---------------------------------------------------------------------------
# render_result() example-based tests
# ---------------------------------------------------------------------------

def test_render_result_relative_path_inside_cwd(tmp_path):
    """render_result with a path inside tmp_path CWD should show relative path, not absolute."""
    image_path = str(tmp_path / "images" / "cat.jpg")
    cwd = str(tmp_path)

    result = render_result(1, 0.62, image_path, cwd=cwd)

    # The absolute tmp_path prefix should NOT appear in the output
    assert str(tmp_path) not in result
    # The relative portion should appear
    assert os.path.join("images", "cat.jpg") in result


def test_render_result_contains_percent_match_suffix(tmp_path):
    """render_result output should contain '% match' suffix."""
    image_path = str(tmp_path / "photo.jpg")
    cwd = str(tmp_path)

    result = render_result(1, 0.75, image_path, cwd=cwd)

    assert "% match" in result


def test_render_result_format_structure(tmp_path):
    """render_result should follow the pattern: '{rank}. {path}  {bar} {pct}% match'."""
    image_path = str(tmp_path / "photo.jpg")
    cwd = str(tmp_path)

    result = render_result(3, 0.5, image_path, cwd=cwd)

    # Should start with the rank
    assert result.startswith("3.")
    # Should contain the bar characters
    assert FILLED in result or EMPTY in result
    # Should end with '% match'
    assert result.endswith("% match")


def test_render_result_percentage_value(tmp_path):
    """render_result should display round(score * 100) as the percentage."""
    image_path = str(tmp_path / "photo.jpg")
    cwd = str(tmp_path)

    result = render_result(1, 0.62, image_path, cwd=cwd)

    # round(0.62 * 100) == 62
    assert "62% match" in result
