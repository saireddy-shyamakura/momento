"""
yolo.py — YOLO object detection for Momento.

Detects objects in images using YOLOv8, returns cropped regions
that can be individually embedded by CLIP for fine-grained search.

Requires: ultralytics
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False


@dataclass
class Detection:
    """A single detected object."""
    label: str
    confidence: float
    bbox: List[float]
    cropped_image: Image.Image = field(repr=False)

    def to_metadata(self) -> dict:
        return {
            "type": "yolo_object",
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox_x1": round(self.bbox[0], 1),
            "bbox_y1": round(self.bbox[1], 1),
            "bbox_x2": round(self.bbox[2], 1),
            "bbox_y2": round(self.bbox[3], 1),
        }


_yolo_model: Optional["YOLO"] = None


def is_available() -> bool:
    return _HAS_YOLO


def get_yolo_model(model_name: str = "yolov8n.pt") -> "YOLO":
    global _yolo_model
    if not _HAS_YOLO:
        raise RuntimeError("ultralytics required. Install: pip install ultralytics")
    if _yolo_model is None:
        logger.info(f"Loading YOLO model: {model_name}")
        _yolo_model = YOLO(model_name)
    return _yolo_model


from .device import device_manager

def detect_objects(
    image_path: str,
    conf_threshold: float = 0.35,
    model_name: str = "yolov8n.pt",
    min_crop_size: int = 32,
) -> List[Detection]:
    if not _HAS_YOLO:
        raise RuntimeError("ultralytics required. Install: pip install ultralytics")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = get_yolo_model(model_name)
    results = model(image_path, conf=conf_threshold, device=device_manager.device, verbose=False)
    if not results:
        return []

    result = results[0]
    detections: List[Detection] = []

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        img_w, img_h = img_rgb.size

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            label = result.names[int(box.cls[0])]

            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))

            if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                continue

            cropped = img_rgb.crop((x1, y1, x2, y2))
            detections.append(Detection(label=label, confidence=conf,
                                        bbox=[x1, y1, x2, y2], cropped_image=cropped))

    logger.info(f"Detected {len(detections)} objects in {os.path.basename(image_path)}")
    return detections


def detect_objects_from_pil(
    img: Image.Image,
    conf_threshold: float = 0.35,
    model_name: str = "yolov8n.pt",
    min_crop_size: int = 32,
) -> List[Detection]:
    if not _HAS_YOLO:
        raise RuntimeError("ultralytics required. Install: pip install ultralytics")
    import numpy as np

    model = get_yolo_model(model_name)
    img_rgb = img.convert("RGB")
    results = model(np.array(img_rgb), conf=conf_threshold, device=device_manager.device, verbose=False)
    if not results:
        return []

    result = results[0]
    detections: List[Detection] = []
    img_w, img_h = img_rgb.size

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        label = result.names[int(box.cls[0])]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))
        if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
            continue
        cropped = img_rgb.crop((x1, y1, x2, y2))
        detections.append(Detection(label=label, confidence=conf,
                                    bbox=[x1, y1, x2, y2], cropped_image=cropped))
    return detections
