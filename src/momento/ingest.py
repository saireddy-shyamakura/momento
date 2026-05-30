"""
ingest.py — Unified media ingestion pipeline for Momento.

Handles both images and videos.  Supports optional multi-embedding,
YOLO object detection, and OCR text extraction during ingestion.

All ingestion functions accept a shared list of image paths to avoid
re-walking the directory tree.  Batching is used throughout to reduce
ChromaDB HNSW index rebuild overhead.
"""

import os
from typing import List, Optional

from .features import (
    extract_multi_embeddings,
    extract_object_embeddings,
    extract_ocr_embedding,
    extract_pil_features_batch,
)
from .index import Index
from .validation import validate_folder_path, is_path_safe
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

# Flush embeddings to ChromaDB in chunks to bound memory and reduce HNSW rebuilds
_CHUNK_SIZE = 200


def _make_progress(total: int, desc: str):
    if _HAS_TQDM and total > 0:
        return tqdm(total=total, desc=desc)
    return None


def _flush_batch(index: Index, ids: list, vecs: list, metas: list) -> int:
    """Flush a batch of vectors to the index, clearing the accumulators."""
    if not ids:
        return 0
    index.add_vectors(ids, vecs, metas)
    count = len(ids)
    ids.clear()
    vecs.clear()
    metas.clear()
    return count


def _collect_image_paths(folder: str) -> List[str]:
    """Walk a folder once and return all valid image paths.

    Shared by multi-embed, YOLO, and OCR ingestion to avoid quadruple
    directory tree walks.
    """
    paths: List[str] = []
    for root, dirs, _ in os.walk(folder, followlinks=False):
        # Do not descend into symlink directories
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
        for file in os.scandir(root):
            if not file.is_file():
                continue
            path = file.path
            if not path.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            is_safe, _ = is_path_safe(path, folder)
            if not is_safe:
                continue
            paths.append(path)
    return paths


from .add_images import add_images


# ── Multi-embedding ingestion ────────────────────────────────────────

def add_images_multi(folder: str, index: Index) -> int:
    """Ingest images with multiple augmented embeddings per image."""
    is_valid, error_msg = validate_folder_path(folder)
    if not is_valid:
        logger.error(f"Cannot add images: {error_msg}")
        return 0

    candidate_paths = _collect_image_paths(folder)

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
    batch_ids: list = []
    batch_vecs: list = []
    batch_metas: list = []

    for path in to_process:
        try:
            embeddings = extract_multi_embeddings(path)
            ids = [f"{path}{COMPOSITE_SEP}{suffix}" for suffix, _ in embeddings]
            vecs = [emb for _, emb in embeddings]
            metas = [{"path": path, "source_path": path, "type": f"augment_{suffix}"}
                     for suffix, _ in embeddings]
            batch_ids.extend(ids)
            batch_vecs.extend(vecs)
            batch_metas.extend(metas)
            added += 1
        except Exception as e:
            logger.error(f"Multi-embed failed for {path}: {e}")

        # Flush to ChromaDB periodically
        if len(batch_ids) >= _CHUNK_SIZE:
            _flush_batch(index, batch_ids, batch_vecs, batch_metas)

        if progress:
            progress.update(1)
        else:
            print(f"Processed: {added}/{len(to_process)}")

    # Flush remaining
    _flush_batch(index, batch_ids, batch_vecs, batch_metas)

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
    for root, dirs, _ in os.walk(folder, followlinks=False):
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
        for file in os.scandir(root):
            if not file.is_file():
                continue
            path = file.path
            if path.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
                is_safe, _ = is_path_safe(path, folder)
                if not is_safe:
                    continue
                video_paths.append(path)

    if not video_paths:
        print("No video files found.")
        return 0

    progress = _make_progress(len(video_paths), "Indexing videos")
    total_frames_added = 0
    batch_ids: list = []
    batch_vecs: list = []
    batch_metas: list = []

    for vpath in video_paths:
        try:
            frames = extract_keyframes(vpath, interval_sec=interval_sec, max_frames=max_frames)
            if not frames:
                continue

            pil_images = [img for _, img in frames]
            timestamps = [ts for ts, _ in frames]

            embeddings = extract_pil_features_batch(pil_images)

            # Explicitly close PIL images to free memory
            for _, img in frames:
                img.close()

            for j, (ts, emb) in enumerate(zip(timestamps, embeddings)):
                frame_id = f"{vpath}{COMPOSITE_SEP}frame_{j:04d}"
                batch_ids.append(frame_id)
                batch_vecs.append(emb)
                batch_metas.append({
                    "path": vpath,
                    "source_path": vpath,
                    "type": "video_frame",
                    "timestamp": str(ts),
                    "frame_index": str(j),
                })

            # Flush to ChromaDB periodically
            if len(batch_ids) >= _CHUNK_SIZE:
                total_frames_added += _flush_batch(index, batch_ids, batch_vecs, batch_metas)

        except Exception as e:
            logger.error(f"Video ingestion failed for {vpath}: {e}")

        if progress:
            progress.update(1)

    # Flush remaining
    total_frames_added += _flush_batch(index, batch_ids, batch_vecs, batch_metas)

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

    image_paths = _collect_image_paths(folder)

    if not image_paths:
        print("No images found.")
        return 0

    progress = _make_progress(len(image_paths), "YOLO detection")
    total_objects = 0
    batch_ids: list = []
    batch_vecs: list = []
    batch_metas: list = []

    for path in image_paths:
        try:
            obj_embeddings = extract_object_embeddings(path)
            if not obj_embeddings:
                if progress:
                    progress.update(1)
                continue

            for meta, emb in obj_embeddings:
                obj_id = _make_yolo_object_id(path, meta)
                batch_ids.append(obj_id)
                batch_vecs.append(emb)
                meta["path"] = path
                meta["source_path"] = path
                batch_metas.append(meta)

            # Flush to ChromaDB periodically
            if len(batch_ids) >= _CHUNK_SIZE:
                total_objects += _flush_batch(index, batch_ids, batch_vecs, batch_metas)

        except Exception as e:
            logger.error(f"YOLO failed for {path}: {e}")

        if progress:
            progress.update(1)

    # Flush remaining
    total_objects += _flush_batch(index, batch_ids, batch_vecs, batch_metas)

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

    image_paths = _collect_image_paths(folder)

    if not image_paths:
        print("No images found.")
        return 0

    progress = _make_progress(len(image_paths), "OCR extraction")
    total_ocr = 0
    batch_ids: list = []
    batch_vecs: list = []
    batch_metas: list = []

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
                "ocr_text": text[:500],
            }
            batch_ids.append(ocr_id)
            batch_vecs.append(emb)
            batch_metas.append(meta)

            # Flush to ChromaDB periodically
            if len(batch_ids) >= _CHUNK_SIZE:
                total_ocr += _flush_batch(index, batch_ids, batch_vecs, batch_metas)

        except Exception as e:
            logger.error(f"OCR failed for {path}: {e}")

        if progress:
            progress.update(1)

    # Flush remaining
    total_ocr += _flush_batch(index, batch_ids, batch_vecs, batch_metas)

    if progress:
        progress.close()

    print(f"Done: {total_ocr} OCR embeddings from {len(image_paths)} images")
    return total_ocr