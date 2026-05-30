"""
output.py — Result rendering utilities for Momento search output.

Provides format_bar(), render_result(), and open_file() so that
result formatting can be unit-tested independently of the CLI.
"""

import os
import platform
import subprocess

BAR_WIDTH = 10
FILLED = "█"
EMPTY = "░"


def format_bar(score: float, width: int = BAR_WIDTH) -> str:
    """
    Render a fixed-width similarity bar composed of block characters.

    Args:
        score: A float in [0.0, 1.0] representing similarity.
        width: Total number of characters in the bar (default 10).

    Returns:
        A string of exactly `width` characters using FILLED and EMPTY blocks.

    Examples:
        format_bar(0.62, 10)  ->  "██████░░░░"
        format_bar(0.0,  10)  ->  "░░░░░░░░░░"
        format_bar(1.0,  10)  ->  "██████████"
    """
    filled = round(score * width)
    return FILLED * filled + EMPTY * (width - filled)


def render_result(rank: int, score: float, path: str, cwd: str | None = None) -> str:
    """
    Format a single search result line for display.

    Args:
        rank:  1-based position in the result list.
        score: Similarity score in [0.0, 1.0].
        path:  Absolute (or relative) path to the matched image.
        cwd:   Directory to compute the relative path from.
               Defaults to os.getcwd() when None.

    Returns:
        A formatted string such as:
            "1. images/cat.jpg  ██████░░░░ 62% match"

    Notes:
        On Windows, os.path.relpath() raises ValueError for cross-drive paths.
        In that case the absolute path is used as the display path.
    """
    if cwd is None:
        cwd = os.getcwd()
    try:
        display_path = os.path.relpath(path, cwd)
    except ValueError:
        # On Windows, relpath raises ValueError for cross-drive paths
        display_path = path
    bar = format_bar(score)
    pct = round(score * 100)
    return f"{rank}. {display_path}  {bar} {pct}% match"


def open_file(path: str) -> None:
    """
    Open a file in the system default viewer (cross-platform).

    Validates the path exists and is a regular file before passing it to
    the OS launcher.  This prevents security issues where a crafted index
    entry could cause xdg-open to interpret a URI or execute a handler.

    Args:
        path: Path to the file to open.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If the platform's opener is unavailable or fails.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    system = platform.system()
    try:
        if system == "Windows":
            if hasattr(os, "startfile"):
                os.startfile(path)
            else:
                raise RuntimeError("os.startfile is not available on this Windows build")
        elif system == "Darwin":
            subprocess.run(["open", path], check=True)
        else:
            subprocess.run(["xdg-open", path], check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"Failed to open {path}: {e}")
