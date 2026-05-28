"""
ingest.py — Unified media ingestion pipeline for Momento.

Handles both images and videos.  Supports optional multi-embedding,
YOLO object detection, and OCR text extraction during ingestion.
"""

import os
from typing import List

from .features import (
    extract_image_features_batch,
    extract_multi_embeddings,
    extract_object_embeddings,
    extract_ocr_embedding,
    extract_pil_features_batch,
)
from .index import Index
from .validation import validate_folder_path, validate_image_path, validate_video_path, is_path_safe
from .config import (
    SUPPORTED_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS,
    VIDEO_FRAME_INTERVAL, MAX_FRAMES_PER_VIDEO, COMPOSITE_SEP,
)
from .logger import get_logger

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

logger = get_logger(__name__)


def _make_progress(total: int, desc: str):
    if _HAS_TQDM:
        return tqdm(total=total, desc=desc)
    return None


from .add_images import add_images


# ── Multi-embedding ingestion ────────────────────────────────────────

def add_images_multi(folder: str, index: Index) -> int:
    """Ingest images with multiple augmented embeddings per image."""
    is_valid, error_msg = validate_folder_path(folder)
    if not is_valid:
        logger.error(f"Cannot add images: {error_msg}")
        return 0

    candidate_paths = []
    for root, _dirs, files in os.walk(folder, followlinks=False):
        for file in files:
            path = os.path.abspath(os.path.join(root, file))
            if not path.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            # Path traversal guard
            is_safe, _ = is_path_safe(path, folder)
            if not is_safe:
                continue
            ok, _ = validate_image_path(path)
            if ok:
                candidate_paths.append(path)

    # Check which base paths already have an orig entry
    orig_suffix = f"{COMPOSITE_SEP}orig"
    orig_ids = [f"{p}{orig_suffix}" for p in candidate_paths]
    existing = index.get_existing_ids(orig_ids)
    to_process = [p for p in candidate_paths if f"{p}{orig_suffix}" not in existing]

    if not to_process:
        print(f"Done: 0 added (all {len(candidate_paths)} already indexed)")
        return 0

    progress = _make_progress(len(to_process), "Multi-embedding")
    added = 0

    for path in to_process:
        try:
            embeddings = extract_multi_embeddings(path)
            ids = [f"{path}{COMPOSITE_SEP}{suffix}" for suffix, _ in embeddings]
            vecs = [emb for _, emb in embeddings]
            metas = [{"path": path, "source_path": path, "type": f"augment_{suffix}"}
                     for suffix, _ in embeddings]
            index.add_vectors(ids, vecs, metas)
            added += 1
        except Exception as e:
            logger.error(f"Multi-embed failed for {path}: {e}")

        if progress:
            progress.update(1)
        else:
            print(f"Processed: {added}/{len(to_process)}")

    if progress:
        progress.close()

    print(f"Done: {added} images multi-embedded ({added * 6} vectors)")
    return added


# ── Video ingestion ──────────────────────────────────────────────────

def add_videos(folder: str, index: Index,
               interval_sec: float = VIDEO_FRAME_INTERVAL,
               max_frames: int = MAX_FRAMES_PER_VIDEO) -> int:
    """Ingest videos by extracting keyframes and embedding them."""
    from .video import extract_keyframes, is_available as video_available

    if not video_available():
        print("Error: opencv-python-headless required. Install: pip install opencv-python-headless")
        return 0

    is_valid, error_msg = validate_folder_path(folder)
    if not is_valid:
        logger.error(f"Cannot scan folder: {error_msg}")
        return 0

    video_paths = []
    for root, _dirs, files in os.walk(folder, followlinks=False):
        for file in files:
            path = os.path.abspath(os.path.join(root, file))
            if path.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
                # Path traversal guard
                is_safe, _ = is_path_safe(path, folder)
                if not is_safe:
                    continue
                video_paths.append(path)

    if not video_paths:
        print("No video files found.")
        return 0

    progress = _make_progress(len(video_paths), "Indexing videos")
    total_frames_added = 0

    for vpath in video_paths:
        try:
            frames = extract_keyframes(vpath, interval_sec=interval_sec, max_frames=max_frames)
            if not frames:
                continue

            pil_images = [img for _, img in frames]
            timestamps = [ts for ts, _ in frames]

            embeddings = extract_pil_features_batch(pil_images)

            ids = []
            metas = []
            for j, (ts, emb) in enumerate(zip(timestamps, embeddings)):
                frame_id = f"{vpath}{COMPOSITE_SEP}frame_{j:04d}"
                ids.append(frame_id)
                metas.append({
                    "path": vpath,
                    "source_path": vpath,
                    "type": "video_frame",
                    "timestamp": str(ts),
                    "frame_index": str(j),
                })

            index.add_vectors(ids, list(embeddings), metas)
            total_frames_added += len(ids)

        except Exception as e:
            logger.error(f"Video ingestion failed for {vpath}: {e}")

        if progress:
            progress.update(1)

    if progress:
        progress.close()

    print(f"Done: {total_frames_added} frames indexed from {len(video_paths)} videos")
    return total_frames_added


# ── YOLO object ingestion ────────────────────────────────────────────

def _make_yolo_object_id(image_path: str, metadata: dict) -> str:
    label = str(metadata.get("label", "obj")).replace(" ", "_").lower()
    bbox = metadata.get("bbox") or []
    if len(bbox) == 4:
        try:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            return f"{image_path}{COMPOSITE_SEP}yolo_{label}_{x1}_{y1}_{x2}_{y2}"
        except Exception:
            pass
    return f"{image_path}{COMPOSITE_SEP}yolo_{label}"


def add_objects(folder: str, index: Index) -> int:
    """Run YOLO on all images in folder and index per-object embeddings."""
    from .yolo import is_available as yolo_available
    if not yolo_available():
        print("Error: ultralytics required. Install: pip install ultralytics")
        return 0

    is_valid, error_msg = validate_folder_path(folder)
    if not is_valid:
        logger.error(f"Cannot scan folder: {error_msg}")
        return 0

    image_paths = []
    for root, _dirs, files in os.walk(folder, followlinks=False):
        for file in files:
            path = os.path.abspath(os.path.join(root, file))
            if path.lower().endswith(SUPPORTED_EXTENSIONS):
                # Path traversal guard
                is_safe, _ = is_path_safe(path, folder)
                if not is_safe:
                    continue
                image_paths.append(path)

    if not image_paths:
        print("No images found.")
        return 0

    progress = _make_progress(len(image_paths), "YOLO detection")
    total_objects = 0

    for path in image_paths:
        try:
            obj_embeddings = extract_object_embeddings(path)
            if not obj_embeddings:
                if progress:
                    progress.update(1)
                continue

            ids = []
            vecs = []
            metas = []
            for meta, emb in obj_embeddings:
                obj_id = _make_yolo_object_id(path, meta)
                ids.append(obj_id)
                vecs.append(emb)
                meta["path"] = path
                meta["source_path"] = path
                metas.append(meta)

            index.add_vectors(ids, vecs, metas)
            total_objects += len(ids)

        except Exception as e:
            logger.error(f"YOLO failed for {path}: {e}")

        if progress:
            progress.update(1)

    if progress:
        progress.close()

    print(f"Done: {total_objects} objects indexed from {len(image_paths)} images")
    return total_objects


# ── OCR ingestion ────────────────────────────────────────────────────

def add_ocr(folder: str, index: Index) -> int:
    """Run OCR on all images and index text embeddings."""
    from .ocr import is_available as ocr_available
    if not ocr_available():
        print("Error: easyocr required. Install: pip install easyocr")
        return 0

    is_valid, error_msg = validate_folder_path(folder)
    if not is_valid:
        logger.error(f"Cannot scan folder: {error_msg}")
        return 0

    image_paths = []
    for root, _dirs, files in os.walk(folder, followlinks=False):
        for file in files:
            path = os.path.abspath(os.path.join(root, file))
            if path.lower().endswith(SUPPORTED_EXTENSIONS):
                # Path traversal guard
                is_safe, _ = is_path_safe(path, folder)
                if not is_safe:
                    continue
                image_paths.append(path)

    if not image_paths:
        print("No images found.")
        return 0

    progress = _make_progress(len(image_paths), "OCR extraction")
    total_ocr = 0

    for path in image_paths:
        try:
            result = extract_ocr_embedding(path)
            if result is None:
                if progress:
                    progress.update(1)
                continue

            text, emb = result
            ocr_id = f"{path}{COMPOSITE_SEP}ocr"
            meta = {
                "path": path,
                "source_path": path,
                "type": "ocr",
                "ocr_text": text[:500],  # truncate for metadata storage
            }
            index.add_vectors([ocr_id], [emb], [meta])
            total_ocr += 1

        except Exception as e:
            logger.error(f"OCR failed for {path}: {e}")

        if progress:
            progress.update(1)

    if progress:
        progress.close()

    print(f"Done: {total_ocr} OCR embeddings from {len(image_paths)} images")
    return total_ocr
