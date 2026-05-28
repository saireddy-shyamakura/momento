"""
video.py — Video keyframe extraction for Momento.

Extracts representative frames from video files using either:
- Uniform interval sampling (default)
- Scene-change detection (optional, based on frame difference)

Requires: opencv-python-headless
"""

import os
import logging
from typing import List, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class VideoExtractionError(Exception):
    """Raised when video frame extraction fails."""
    pass


def is_available() -> bool:
    """Check if video processing dependencies are installed."""
    return _HAS_CV2


def _validate_video(video_path: str) -> None:
    """Validate that the video file exists and is readable."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.isfile(video_path):
        raise VideoExtractionError(f"Path is not a file: {video_path}")
    if not os.access(video_path, os.R_OK):
        raise VideoExtractionError(f"Video file is not readable: {video_path}")


def get_video_info(video_path: str) -> dict:
    """
    Get basic video metadata.

    Returns:
        Dict with keys: fps, frame_count, duration_sec, width, height
    """
    if not _HAS_CV2:
        raise VideoExtractionError("opencv-python-headless is required for video processing. Install with: pip install opencv-python-headless")

    _validate_video(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoExtractionError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": round(duration, 2),
            "width": width,
            "height": height,
        }
    finally:
        cap.release()


def extract_keyframes_interval(
    video_path: str,
    interval_sec: float = 2.0,
    max_frames: int = 50,
) -> List[Tuple[float, Image.Image]]:
    """
    Extract frames at uniform time intervals.

    Args:
        video_path: Path to video file.
        interval_sec: Seconds between extracted frames.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of (timestamp_sec, PIL.Image) tuples.
    """
    if not _HAS_CV2:
        raise VideoExtractionError("opencv-python-headless is required for video processing")

    _validate_video(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoExtractionError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            raise VideoExtractionError(f"Invalid video metadata (fps={fps}, frames={total_frames})")

        frame_interval = int(fps * interval_sec)
        if frame_interval < 1:
            frame_interval = 1

        frames: List[Tuple[float, Image.Image]] = []
        frame_idx = 0

        while frame_idx < total_frames and len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            # Convert BGR (OpenCV) → RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            timestamp = frame_idx / fps

            frames.append((round(timestamp, 2), pil_image))
            frame_idx += frame_interval

        logger.info(f"Extracted {len(frames)} frames from {os.path.basename(video_path)}")
        return frames

    finally:
        cap.release()


def extract_keyframes_scene_change(
    video_path: str,
    threshold: float = 30.0,
    max_frames: int = 50,
    min_interval_sec: float = 0.5,
) -> List[Tuple[float, Image.Image]]:
    """
    Extract frames at scene changes based on frame difference.

    Args:
        video_path: Path to video file.
        threshold: Mean pixel difference threshold to detect a scene change.
        max_frames: Maximum number of frames to extract.
        min_interval_sec: Minimum seconds between extracted frames.

    Returns:
        List of (timestamp_sec, PIL.Image) tuples.
    """
    if not _HAS_CV2:
        raise VideoExtractionError("opencv-python-headless is required for video processing")

    _validate_video(video_path)
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoExtractionError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise VideoExtractionError("Invalid FPS in video")

        min_frame_gap = int(fps * min_interval_sec)
        frames: List[Tuple[float, Image.Image]] = []
        prev_gray = None
        frame_idx = 0
        last_captured_idx = -min_frame_gap  # Allow first frame

        # Always capture the first frame
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((0.0, Image.fromarray(rgb_frame)))
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            last_captured_idx = 0

        frame_idx = 1
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx - last_captured_idx < min_frame_gap:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = np.mean(np.abs(gray - prev_gray))

            if diff > threshold:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = frame_idx / fps
                frames.append((round(timestamp, 2), Image.fromarray(rgb_frame)))
                last_captured_idx = frame_idx

            prev_gray = gray
            frame_idx += 1

        logger.info(f"Extracted {len(frames)} scene-change frames from {os.path.basename(video_path)}")
        return frames

    finally:
        cap.release()


def extract_keyframes(
    video_path: str,
    strategy: str = "interval",
    interval_sec: float = 2.0,
    max_frames: int = 50,
    scene_threshold: float = 30.0,
) -> List[Tuple[float, Image.Image]]:
    """
    High-level API: extract keyframes using the specified strategy.

    Args:
        video_path: Path to video file.
        strategy: 'interval' or 'scene_change'.
        interval_sec: Seconds between frames (interval strategy).
        max_frames: Max frames to extract.
        scene_threshold: Pixel diff threshold (scene_change strategy).

    Returns:
        List of (timestamp_sec, PIL.Image) tuples.
    """
    if strategy == "scene_change":
        return extract_keyframes_scene_change(video_path, threshold=scene_threshold, max_frames=max_frames)
    else:
        return extract_keyframes_interval(video_path, interval_sec=interval_sec, max_frames=max_frames)
