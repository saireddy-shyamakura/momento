"""Unit tests for video.py — keyframe extraction and video info.

Covers:
- _validate_video with various edge cases
- get_video_info
- extract_keyframes_interval (with mocked cv2)
- extract_keyframes_scene_change (with mocked cv2)
- extract_keyframes routing
- Missing dependency handling
- Invalid file and path edge cases
"""

import os
import io
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
import pytest

from momento.video import (
    extract_keyframes,
    extract_keyframes_interval,
    extract_keyframes_scene_change,
    get_video_info,
    VideoExtractionError,
    is_available,
)


class TestIsAvailable:
    """Tests for is_available()."""

    def test_returns_true_when_cv2_importable(self):
        with patch("momento.video._HAS_CV2", True):
            assert is_available() is True

    def test_returns_false_when_cv2_not_importable(self):
        with patch("momento.video._HAS_CV2", False):
            assert is_available() is False


class TestVideoValidation:
    """Tests for _validate_video edge cases."""

    def test_nonexistent_path_raises_file_not_found(self):
        from momento.video import _validate_video
        with pytest.raises(FileNotFoundError, match="not found"):
            _validate_video("/nonexistent/video.mp4")

    def test_directory_path_raises_extraction_error(self, tmp_path):
        from momento.video import _validate_video
        with pytest.raises(VideoExtractionError, match="not a file"):
            _validate_video(str(tmp_path))

    def test_unreadable_file_raises_extraction_error(self, tmp_path):
        from momento.video import _validate_video
        video = tmp_path / "readonly.mp4"
        video.write_text("dummy")
        os.chmod(str(video), 0o000)
        try:
            with pytest.raises(VideoExtractionError, match="not readable"):
                _validate_video(str(video))
        finally:
            os.chmod(str(video), 0o644)


class TestGetVideoInfo:
    """Tests for get_video_info."""

    def test_raises_when_cv2_missing(self, tmp_path):
        dummy = tmp_path / "test.mp4"
        dummy.write_text("dummy")
        with patch("momento.video._HAS_CV2", False):
            with pytest.raises(VideoExtractionError, match="opencv-python-headless"):
                get_video_info(str(dummy))

    def test_raises_when_cannot_open(self, tmp_path):
        dummy = tmp_path / "test.mp4"
        dummy.write_text("dummy")
        with patch("momento.video._HAS_CV2", True), \
             patch("cv2.VideoCapture") as mock_vc:
            cap = MagicMock()
            cap.isOpened.return_value = False
            mock_vc.return_value = cap
            with pytest.raises(VideoExtractionError, match="Cannot open"):
                get_video_info(str(dummy))

    def test_returns_correct_metadata(self, tmp_path):
        dummy = tmp_path / "test.mp4"
        dummy.write_text("dummy")
        with patch("momento.video._HAS_CV2", True), \
             patch("cv2.VideoCapture") as mock_vc:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda prop: {
                5: 30.0,   # CAP_PROP_FPS
                7: 300,    # CAP_PROP_FRAME_COUNT
                3: 1920,   # CAP_PROP_FRAME_WIDTH
                4: 1080,   # CAP_PROP_FRAME_HEIGHT
            }.get(prop, 0.0)
            mock_vc.return_value = cap
            info = get_video_info(str(dummy))
            assert info["fps"] == 30.0
            assert info["frame_count"] == 300
            assert info["duration_sec"] == 10.0
            assert info["width"] == 1920
            assert info["height"] == 1080


class TestExtractKeyframesInterval:
    """Tests for extract_keyframes_interval."""

    def test_raises_on_missing_cv2(self, tmp_path):
        dummy = tmp_path / "test.mp4"
        dummy.write_text("dummy")
        with patch("momento.video._HAS_CV2", False):
            with pytest.raises(VideoExtractionError, match="opencv-python-headless"):
                extract_keyframes_interval(str(dummy))

    def test_raises_on_invalid_metadata(self, tmp_path):
        dummy = tmp_path / "test.mp4"
        dummy.write_text("dummy")
        with patch("momento.video._HAS_CV2", True), \
             patch("cv2.VideoCapture") as mock_vc:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda prop: {
                5: 0.0,   # FPS = 0 → invalid
                7: 0,     # FRAME_COUNT = 0
            }.get(prop, 0.0)
            mock_vc.return_value = cap
            with pytest.raises(VideoExtractionError, match="Invalid video metadata"):
                extract_keyframes_interval(str(dummy))

    def test_extracts_frames_with_mocked_cv2(self, tmp_path):
        dummy = tmp_path / "test.mp4"
        dummy.write_text("dummy")
        with patch("momento.video._HAS_CV2", True), \
             patch("cv2.VideoCapture") as mock_vc, \
             patch("cv2.cvtColor", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda prop: {
                5: 30.0,   # FPS
                7: 90,     # FRAME_COUNT
            }.get(prop, 0.0)
            cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_vc.return_value = cap

            frames = extract_keyframes_interval(str(dummy), interval_sec=1.0, max_frames=3)
            assert len(frames) <= 3
            if frames:
                ts, img = frames[0]
                assert isinstance(ts, float)
                assert isinstance(img, Image.Image)

    def test_stops_when_frames_cannot_be_read(self, tmp_path):
        dummy = tmp_path / "test.mp4"
        dummy.write_text("dummy")
        with patch("momento.video._HAS_CV2", True), \
             patch("cv2.VideoCapture") as mock_vc:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda prop: {
                5: 30.0,
                7: 1000,
            }.get(prop, 0.0)
            # Always fail after first read
            cap.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8)),
                                    (False, None)]
            mock_vc.return_value = cap
            frames = extract_keyframes_interval(str(dummy), interval_sec=0.1, max_frames=100)
            assert len(frames) == 1

    def test_frame_interval_clamped_to_minimum_1(self, tmp_path):
        """When frame_interval < 1, it should be clamped to 1."""
        dummy = tmp_path / "test.mp4"
        dummy.write_text("dummy")
        with patch("momento.video._HAS_CV2", True), \
             patch("cv2.VideoCapture") as mock_vc, \
             patch("cv2.cvtColor", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda prop: {
                5: 30.0,
                7: 5,
            }.get(prop, 0.0)
            cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_vc.return_value = cap
            frames = extract_keyframes_interval(str(dummy), interval_sec=0.01, max_frames=10)
            # Should not crash, should return some frames
            assert len(frames) > 0


class TestExtractKeyframesSceneChange:
    """Tests for extract_keyframes_scene_change."""

    def test_first_frame_always_captured(self, tmp_path):
        dummy = tmp_path / "test.mp4"
        dummy.write_text("dummy")
        with patch("momento.video._HAS_CV2", True), \
             patch("cv2.VideoCapture") as mock_vc, \
             patch("cv2.cvtColor", side_effect=lambda x, _: np.zeros((480, 640), dtype=np.float32)), \
             patch("numpy.mean", return_value=0.0):  # No scene changes
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda prop: {5: 30.0}.get(prop, 0.0)
            # Only first frame succeeds (no more frames)
            cap.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8)),
                                    (False, None)]
            mock_vc.return_value = cap
            frames = extract_keyframes_scene_change(str(dummy), threshold=30.0, max_frames=10)
            assert len(frames) == 1  # First frame always captured

    def test_invalid_fps_raises_error(self, tmp_path):
        dummy = tmp_path / "test.mp4"
        dummy.write_text("dummy")
        with patch("momento.video._HAS_CV2", True), \
             patch("cv2.VideoCapture") as mock_vc:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda prop: {5: 0.0}.get(prop, 0.0)  # FPS = 0
            mock_vc.return_value = cap
            with pytest.raises(VideoExtractionError, match="Invalid FPS"):
                extract_keyframes_scene_change(str(dummy))

    def test_raises_on_missing_cv2(self, tmp_path):
        dummy = tmp_path / "test.mp4"
        dummy.write_text("dummy")
        with patch("momento.video._HAS_CV2", False):
            with pytest.raises(VideoExtractionError, match="opencv-python-headless"):
                extract_keyframes_scene_change(str(dummy))


class TestExtractKeyframesRouting:
    """Tests for the high-level extract_keyframes function."""

    def test_routes_to_interval_by_default(self, tmp_path):
        dummy = tmp_path / "test.mp4"
        dummy.write_text("dummy")
        with patch("momento.video.extract_keyframes_interval") as mock_interval, \
             patch("momento.video._HAS_CV2", True):
            mock_interval.return_value = [(0.0, Image.new("RGB", (1, 1)))]
            result = extract_keyframes(str(dummy))
            mock_interval.assert_called_once()
            assert len(result) == 1

    def test_routes_to_scene_change(self, tmp_path):
        dummy = tmp_path / "test.mp4"
        dummy.write_text("dummy")
        with patch("momento.video.extract_keyframes_scene_change") as mock_scene, \
             patch("momento.video._HAS_CV2", True):
            mock_scene.return_value = [(0.0, Image.new("RGB", (1, 1)))]
            result = extract_keyframes(str(dummy), strategy="scene_change")
            mock_scene.assert_called_once()
            assert len(result) == 1