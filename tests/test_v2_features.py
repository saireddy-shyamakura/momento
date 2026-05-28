import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import torch

from momento.device import DeviceManager
from momento.augment import generate_augmentations
from momento.video import extract_keyframes, VideoExtractionError
from momento.yolo import detect_objects, detect_objects_from_pil, Detection
from momento.ocr import extract_text, extract_text_from_pil


# =====================================================================
# 1. DeviceManager Tests
# =====================================================================

class TestDeviceManager:
    """Tests for DeviceManager (device detection, overrides, and OOM fallback)."""

    @patch.dict(os.environ, {"MOMENTO_DEVICE": "cpu"})
    def test_env_override_cpu(self):
        """MOMENTO_DEVICE=cpu overrides any GPU availability checks."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = DeviceManager()
            assert manager.device == "cpu"
            assert manager.dtype == torch.float32

    @patch.dict(os.environ, {"MOMENTO_DEVICE": "cuda"})
    def test_env_override_cuda_available(self):
        """MOMENTO_DEVICE=cuda succeeds if CUDA is actually available."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_name", return_value="Mock GPU"):
            manager = DeviceManager()
            assert manager.device == "cuda"

    @patch.dict(os.environ, {"MOMENTO_DEVICE": "cuda"})
    def test_env_override_cuda_unavailable(self):
        """MOMENTO_DEVICE=cuda falls back to cpu if CUDA is unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            manager = DeviceManager()
            assert manager.device == "cpu"

    @patch.dict(os.environ, {})
    def test_auto_detect_cuda(self):
        """Auto-detect selects cuda if available and no override exists."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_name", return_value="Mock GPU"):
            manager = DeviceManager()
            assert manager.device == "cuda"

    def test_fallback_to_cpu(self):
        """Calling fallback_to_cpu changes device to CPU and dtype to float32."""
        manager = DeviceManager()
        manager._device = "cuda"
        manager._dtype = torch.float16

        manager.fallback_to_cpu()
        assert manager.device == "cpu"
        assert manager.dtype == torch.float32


# =====================================================================
# 2. Image Augmentation Tests
# =====================================================================

class TestImageAugmentation:
    """Tests for generate_augmentations in augment.py."""

    def test_generate_augmentations_returns_expected_views(self):
        """generate_augmentations produces the correct number of views with expected suffixes."""
        # Create a small white image
        img = Image.new("RGB", (100, 100), color="white")
        views = generate_augmentations(img)
        
        # We expect 5 default augmentations: flip, crop, bright, contrast, rotate
        assert len(views) == 5
        suffixes = [view[0] for view in views]
        assert "flip" in suffixes
        assert "crop" in suffixes
        assert "bright" in suffixes
        assert "contrast" in suffixes
        assert "rotate" in suffixes

        # All returned views must be PIL Images
        for suffix, view_img in views:
            assert isinstance(view_img, Image.Image)
            assert view_img.size != (0, 0)


# =====================================================================
# 3. Video Extraction Tests
# =====================================================================

class TestVideoExtraction:
    """Tests for video frame extraction (strategy routing and validation)."""

    def test_nonexistent_video_raises_file_not_found(self):
        """Extracting keyframes from a missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            extract_keyframes("nonexistent_video.mp4")

    @patch("momento.video._HAS_CV2", False)
    def test_missing_cv2_raises_error(self, tmp_path):
        """If opencv-python-headless is not installed, VideoExtractionError is raised."""
        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_text("dummy contents")
        with pytest.raises(VideoExtractionError, match="required for video processing"):
            extract_keyframes(str(dummy_video))

    @patch("momento.video._HAS_CV2", True)
    @patch("cv2.VideoCapture")
    def test_interval_extraction_logic(self, mock_vc, tmp_path):
        """extract_keyframes route to interval extraction correctly and reads frames."""
        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_text("dummy contents")

        # Mock cv2 VideoCapture instance
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            3: 640,   # CAP_PROP_FRAME_WIDTH
            4: 480,   # CAP_PROP_FRAME_HEIGHT
            5: 30.0,  # CAP_PROP_FPS
            7: 90,    # CAP_PROP_FRAME_COUNT
        }.get(prop, 0.0)

        # Mock frame reading: return a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap.read.return_value = (True, dummy_frame)
        mock_vc.return_value = cap

        frames = extract_keyframes(str(dummy_video), strategy="interval", interval_sec=1.0, max_frames=5)
        
        # Verify call arguments & resulting frame list
        assert len(frames) > 0
        timestamp, image = frames[0]
        assert isinstance(timestamp, float)
        assert isinstance(image, Image.Image)
        assert image.size == (640, 480)


# =====================================================================
# 4. YOLO Object Detection Tests
# =====================================================================

class TestYOLOObjectDetection:
    """Tests for YOLO object detection wrappers."""

    @patch("momento.yolo._HAS_YOLO", False)
    def test_yolo_unavailable_raises_runtime_error(self):
        """If ultralytics is not installed, detect_objects raises RuntimeError."""
        with pytest.raises(RuntimeError, match="ultralytics required"):
            detect_objects("dummy.jpg")

    @patch("momento.yolo._HAS_YOLO", True)
    @patch("momento.yolo.get_yolo_model")
    def test_detect_objects_from_pil(self, mock_get_model):
        """detect_objects_from_pil parses YOLO results and extracts crops correctly."""
        # Create a mock YOLO model
        model = MagicMock()
        
        # Mock the Box container from ultralytics
        box_mock = MagicMock()
        xyxy_mock = MagicMock()
        xyxy_mock.tolist.return_value = [10.0, 20.0, 50.0, 60.0]
        box_mock.xyxy = [xyxy_mock]
        box_mock.conf = [0.85]
        box_mock.cls = [0]
        
        # Mock Result container from ultralytics
        result_mock = MagicMock()
        result_mock.boxes = [box_mock]
        result_mock.names = {0: "person"}
        
        model.return_value = [result_mock]
        mock_get_model.return_value = model

        # Run on a dummy image
        img = Image.new("RGB", (100, 100), color="blue")
        detections = detect_objects_from_pil(img)

        assert len(detections) == 1
        det = detections[0]
        assert isinstance(det, Detection)
        assert det.label == "person"
        assert det.confidence == 0.85
        assert det.bbox == [10, 20, 50, 60]
        assert isinstance(det.cropped_image, Image.Image)
        assert det.cropped_image.size == (40, 40)  # (50-10, 60-20)


# =====================================================================
# 5. OCR Text Extraction Tests
# =====================================================================

class TestOCRTextExtraction:
    """Tests for EasyOCR text extraction wrappers."""

    @patch("momento.ocr._HAS_OCR", False)
    def test_ocr_unavailable_raises_runtime_error(self):
        """If easyocr is not installed, extract_text raises RuntimeError."""
        with pytest.raises(RuntimeError, match="easyocr required"):
            extract_text("dummy.jpg")

    @patch("momento.ocr._HAS_OCR", True)
    @patch("momento.ocr.get_reader")
    def test_extract_text_from_pil(self, mock_get_reader):
        """extract_text_from_pil parses reader results and concatenates text."""
        # Mock easyocr.Reader instance
        reader = MagicMock()
        # EasyOCR returns list of tuples: (bbox, text, confidence)
        reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 5], [0, 5]], "Hello", 0.95),
            ([[20, 0], [30, 0], [30, 5], [20, 5]], "World", 0.88),
        ]
        mock_get_reader.return_value = reader

        # Run on a dummy image
        img = Image.new("RGB", (100, 100), color="red")
        text = extract_text_from_pil(img)

        assert text == "Hello World"
        reader.readtext.assert_called_once()
