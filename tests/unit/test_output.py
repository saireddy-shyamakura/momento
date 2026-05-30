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

    assert "62% match" in result

# ---------------------------------------------------------------------------
# Hypothesis Property Tests
# ---------------------------------------------------------------------------

from hypothesis import given, settings, HealthCheck, strategies as st
import string

class TestOutputProperties:

    @given(score=st.floats(min_value=0.0, max_value=1.0))
    def test_property_10_bar_is_exactly_10_chars(self, score):
        """Property 10: Similarity bar is exactly 10 characters."""
        assert len(format_bar(score)) == 10

    @given(score=st.floats(min_value=0.0, max_value=1.0))
    def test_property_11_bar_filled_count_proportional(self, score):
        """Property 11: Similarity bar filled count is proportional to score."""
        assert format_bar(score).count("█") == round(score * 10)

    @given(
        score=st.floats(min_value=0.0, max_value=1.0),
        rank=st.integers(min_value=1, max_value=1000),
        path_components=st.lists(st.text(alphabet=string.ascii_letters, min_size=1, max_size=10), min_size=1, max_size=5)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_12_relative_path_display(self, tmp_path, score, rank, path_components):
        """Property 12: Relative path display for paths inside CWD."""
        # Construct absolute path under a tmp CWD
        cwd = str(tmp_path)
        image_path = os.path.join(cwd, *path_components)
        
        result = render_result(rank, score, image_path, cwd=cwd)
        
        # assert render_result() output does not contain the absolute prefix
        assert cwd not in result
