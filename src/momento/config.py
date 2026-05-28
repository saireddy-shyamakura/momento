"""
config.py — Central configuration for Momento.

Supports:
- Default settings defined here
- TOML config file loaded from ~/.config/momento/config.toml
- Environment variable overrides (MOMENTO_DEVICE)
- CLI flag overrides (passed as kwargs to load_config)
- 'momento config show' / 'momento config set' subcommands
"""

import os
import platformdirs
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict

from .logger import get_logger
from .device import DeviceManager, device_manager

# Package-internal reference (for bundled assets only — NOT for user data)
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# User-facing data directory: ~/.local/share/momento  (XDG on Linux/Mac, %APPDATA% on Windows)
_DATA_DIR = platformdirs.user_data_dir("momento", appauthor=False)
CHROMA_DB_DIR = os.path.join(_DATA_DIR, "chroma_db")
LOG_DIR = os.path.join(_DATA_DIR, "logs")

# Embedding cache directory (store per-file embeddings to speed re-indexing)
EMBEDDING_CACHE_DIR = os.path.join(_DATA_DIR, "embedding_cache")

# Keep BASE_DIR as an alias for any code still referencing it (e.g. lock file path)
BASE_DIR = _DATA_DIR

# ── CLIP model ────────────────────────────────────────────────────────
MODEL_NAME = "ViT-B/16"
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
SIMILARITY_THRESHOLD = 0.20

# ── Device (from DeviceManager singleton) ─────────────────────────────
DEVICE = device_manager.device

# ── Feature Toggles (all enabled by default) ─────────────────────────
ENABLE_MULTI_EMBED = True      # Enable multi-embedding via image augmentation
ENABLE_VIDEO_INDEXING = True   # Enable video keyframe extraction
ENABLE_YOLO = True             # Enable YOLO object detection
ENABLE_OCR = True              # Enable OCR text extraction

# ── Multi-embedding augmentation ──────────────────────────────────────
AUGMENTATION_COUNT = 5         # number of augmented views per image

# ── Video ─────────────────────────────────────────────────────────────
SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
VIDEO_FRAME_INTERVAL = 2.0     # seconds between extracted frames
MAX_FRAMES_PER_VIDEO = 50

# ── YOLO ──────────────────────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.35

# ── OCR ───────────────────────────────────────────────────────────────
OCR_LANGUAGES = ["en"]
OCR_MIN_TEXT_LENGTH = 3        # ignore OCR results shorter than this

# ── Composite ID separator ────────────────────────────────────────────
COMPOSITE_SEP = "|||"

# ── Indexing ──────────────────────────────────────────────────────────
INDEXING_BATCH_SIZE = 32       # batch size for feature extraction
MAX_SEARCH_RESULTS = 50        # max results to return from search
PROGRESS_BAR_ENABLED = True    # show progress bars during indexing

# ── Cache ─────────────────────────────────────────────────────────────
CACHE_MAX_SIZE_GB = 5          # max cache size in GB (default: 5 GB)

# ── Logging ───────────────────────────────────────────────────────────
LOG_FORMAT = "text"            # "text" or "json"
LOG_LEVEL = "INFO"             # "DEBUG", "INFO", "WARNING", "ERROR"

# ── Config file path ──────────────────────────────────────────────────
_CONFIG_DIR = platformdirs.user_config_dir("momento", appauthor=False)
CONFIG_FILE = os.path.join(_CONFIG_DIR, "config.toml")

logger = get_logger(__name__)


@dataclass
class MomentoConfig:
    """All Momento configuration as a dataclass.

    Can be serialised to/from a TOML config file.
    """
    # Feature toggles
    enable_multi_embed: bool = ENABLE_MULTI_EMBED
    enable_video_indexing: bool = ENABLE_VIDEO_INDEXING
    enable_yolo: bool = ENABLE_YOLO
    enable_ocr: bool = ENABLE_OCR

    # Similarity
    similarity_threshold: float = SIMILARITY_THRESHOLD
    max_search_results: int = MAX_SEARCH_RESULTS

    # Indexing
    indexing_batch_size: int = INDEXING_BATCH_SIZE
    progress_bar_enabled: bool = PROGRESS_BAR_ENABLED

    # Video
    video_frame_interval: float = VIDEO_FRAME_INTERVAL
    max_frames_per_video: int = MAX_FRAMES_PER_VIDEO

    # YOLO
    yolo_model: str = YOLO_MODEL
    yolo_confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD

    # OCR
    ocr_languages: list = field(default_factory=lambda: ["en"])
    ocr_min_text_length: int = OCR_MIN_TEXT_LENGTH

    # Augmentation
    augmentation_count: int = AUGMENTATION_COUNT

    # Cache
    cache_max_size_gb: int = CACHE_MAX_SIZE_GB

    # Logging
    log_format: str = LOG_FORMAT
    log_level: str = LOG_LEVEL


def _get_config_path() -> str:
    """Return the path to the TOML config file."""
    return CONFIG_FILE


def load_config() -> MomentoConfig:
    """Load configuration from the TOML config file, falling back to defaults.

    The TOML file at ``~/.config/momento/config.toml`` is read if it exists.
    Missing or invalid keys fall back to their default values silently.

    Returns:
        A MomentoConfig dataclass with loaded or default values.
    """
    config = MomentoConfig()
    config_path = _get_config_path()

    if not os.path.exists(config_path):
        return config

    try:
        import tomllib
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        # Map TOML sections to dataclass fields
        for section_name in ("features", "similarity", "indexing", "video",
                             "yolo", "ocr", "augmentation", "cache", "logging"):
            section = data.get(section_name, {})
            if not isinstance(section, dict):
                continue
            for key, value in section.items():
                # Convert TOML key names (e.g., "enable_multi_embed") to dataclass fields
                if hasattr(config, key):
                    setattr(config, key, value)
    except (tomllib.TOMLDecodeError, OSError, ValueError) as e:
        logger.warning(f"Failed to load config file {config_path}: {e}")

    return config


def save_config(config: MomentoConfig, config_path: Optional[str] = None) -> None:
    """Save configuration to a TOML file.

    Args:
        config: The configuration to save.
        config_path: Path to write to. Defaults to CONFIG_FILE.
    """
    path = config_path or _get_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    config_dict = asdict(config)

    # Organise into sections
    sections = {
        "features": {k: config_dict[k] for k in (
            "enable_multi_embed", "enable_video_indexing", "enable_yolo", "enable_ocr"
        ) if k in config_dict},
        "similarity": {k: config_dict[k] for k in (
            "similarity_threshold", "max_search_results"
        ) if k in config_dict},
        "indexing": {k: config_dict[k] for k in (
            "indexing_batch_size", "progress_bar_enabled"
        ) if k in config_dict},
        "video": {k: config_dict[k] for k in (
            "video_frame_interval", "max_frames_per_video"
        ) if k in config_dict},
        "yolo": {k: config_dict[k] for k in (
            "yolo_model", "yolo_confidence_threshold"
        ) if k in config_dict},
        "ocr": {k: config_dict[k] for k in (
            "ocr_languages", "ocr_min_text_length"
        ) if k in config_dict},
        "augmentation": {k: config_dict[k] for k in (
            "augmentation_count",
        ) if k in config_dict},
        "cache": {k: config_dict[k] for k in (
            "cache_max_size_gb",
        ) if k in config_dict},
        "logging": {k: config_dict[k] for k in (
            "log_format", "log_level",
        ) if k in config_dict},
    }

    try:
        with open(path, "w") as f:
            f.write("# Momento Configuration\n")
            f.write("# Generated by `momento config show --save`\n\n")
            for section_name, section_fields in sections.items():
                if section_fields:
                    f.write(f"[{section_name}]\n")
                    for key, value in section_fields.items():
                        if isinstance(value, bool):
                            f.write(f"{key} = {'true' if value else 'false'}\n")
                        elif isinstance(value, list):
                            f.write(f"{key} = {value}\n")
                        elif isinstance(value, str):
                            f.write(f'{key} = "{value}"\n')
                        else:
                            f.write(f"{key} = {value}\n")
                    f.write("\n")
        logger.info(f"Config saved to {path}")
    except OSError as e:
        logger.error(f"Failed to save config to {path}: {e}")
        raise


def apply_config_overrides(cli_args: Optional[Dict[str, Any]] = None) -> None:
    """Apply CLI flag overrides to the global module-level config variables.

    Args:
        cli_args: Dict of config key → value from CLI flags.
    """
    if cli_args is None:
        return

    # Mapping of CLI flag names → module-level config variable names
    _OVERRIDE_MAP = {
        "threshold": "SIMILARITY_THRESHOLD",
        "batch_size": "INDEXING_BATCH_SIZE",
        "max_results": "MAX_SEARCH_RESULTS",
        "log_format": "LOG_FORMAT",
        "log_level": "LOG_LEVEL",
    }

    for cli_key, var_name in _OVERRIDE_MAP.items():
        if cli_key in cli_args and cli_args[cli_key] is not None:
            globals()[var_name] = cli_args[cli_key]


def get_device() -> str:
    """Return the current device string from DeviceManager.

    This helper performs a fresh device detection on each call,
    so it remains reliable even if the global singleton was
    initialized before environment changes or test mocks.
    """
    return DeviceManager().device