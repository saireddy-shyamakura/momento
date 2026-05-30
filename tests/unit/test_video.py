"""Unit tests for video.py — video keyframe extraction.

Tests is_available, validation, get_video_info error handling,
and extract_keyframes error cases.
"""
import pytest
from unittest.mock import patch


class TestVideoAvailable:
    """Video processing availability."""

    def test_is_available_true_when_importable(self):
        with patch("momento.video._HAS_CV2", True):
            from momento.video import is_available
            assert is_available() is True

    def test_is_available_false_when_not_importable(self):
        with patch("momento.video._HAS_CV2", False):
            from momento.video import is_available
            assert is_available() is False


class TestValidateVideo:
    """Video validation."""

    def test_raises_if_file_not_found(self):
        from momento.video import _validate_video
        with pytest.raises(FileNotFoundError, match="not found"):
            _validate_video("/nonexistent/video.mp4")

    def test_raises_if_path_is_directory(self, tmp_path):
        from momento.video import _validate_video
        with pytest.raises(Exception):
            _validate_video(str(tmp_path))


class TestGetVideoInfo:
    """get_video_info error handling."""

    def test_raises_if_cv2_not_available(self):
        with patch("momento.video._HAS_CV2", False):
            from momento.video import get_video_info
            with pytest.raises(Exception, match="opencv"):
                get_video_info("/fake/video.mp4")


class TestExtractKeyframes:
    """extract_keyframes error handling."""

    def test_interval_raises_if_cv2_not_available(self):
        with patch("momento.video._HAS_CV2", False):
            from momento.video import extract_keyframes_interval
            with pytest.raises(Exception, match="opencv"):
                extract_keyframes_interval("/fake/video.mp4")

    def test_scene_change_raises_if_cv2_not_available(self):
        with patch("momento.video._HAS_CV2", False):
            from momento.video import extract_keyframes_scene_change
            with pytest.raises(Exception, match="opencv"):
                extract_keyframes_scene_change("/fake/video.mp4")

    def test_extract_keyframes_default_strategy(self):
        with patch("momento.video._HAS_CV2", False):
            from momento.video import extract_keyframes
            with pytest.raises(Exception, match="opencv"):
                extract_keyframes("/fake/video.mp4", strategy="interval")