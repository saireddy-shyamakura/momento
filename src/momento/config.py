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
SUPPORTED_MODELS = ("ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px", "ConvNeXt-B")
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
    # Embedding model
    model_name: str = MODEL_NAME
    
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
    """Load configuration from TOML file, environment variables, and defaults.

    Priority order (highest to lowest):
    1. Environment variables (MOMENTO_* prefix)
    2. TOML config file at ~/.config/momento/config.toml
    3. Hardcoded defaults

    The TOML file at ``~/.config/momento/config.toml`` is read if it exists.
    Missing or invalid keys fall back to their default values silently.

    Returns:
        A MomentoConfig dataclass with loaded or default values.
    """
    config = MomentoConfig()
    config_path = _get_config_path()

    # Load from TOML file first
    if os.path.exists(config_path):
        try:
            import tomllib
            with open(config_path, "rb") as f:
                data = tomllib.load(f)

            # Map TOML sections to dataclass fields
            for section_name in ("embedding", "features", "similarity", "indexing", "video",
                                 "yolo", "ocr", "augmentation", "cache", "logging"):
                section = data.get(section_name, {})
                if not isinstance(section, dict):
                    continue
                for key, value in section.items():
                    # Convert TOML key names (e.g., "enable_multi_embed") to dataclass fields
                    if hasattr(config, key):
                        setattr(config, key, value)
        except (Exception, OSError, ValueError) as e:
            logger.warning(f"Failed to load config file {config_path}: {e}")

    # Override with environment variables (highest priority)
    _ENV_MAPPING = {
        "MOMENTO_MODEL_NAME": ("model_name", str),
        "MOMENTO_ENABLE_MULTI_EMBED": ("enable_multi_embed", lambda x: x.lower() in ("true", "1", "yes")),
        "MOMENTO_ENABLE_VIDEO_INDEXING": ("enable_video_indexing", lambda x: x.lower() in ("true", "1", "yes")),
        "MOMENTO_ENABLE_YOLO": ("enable_yolo", lambda x: x.lower() in ("true", "1", "yes")),
        "MOMENTO_ENABLE_OCR": ("enable_ocr", lambda x: x.lower() in ("true", "1", "yes")),
        "MOMENTO_DEVICE": ("device", str),
        "MOMENTO_SIMILARITY_THRESHOLD": ("similarity_threshold", float),
        "MOMENTO_MAX_SEARCH_RESULTS": ("max_search_results", int),
        "MOMENTO_YOLO_MODEL": ("yolo_model", str),
        "MOMENTO_CACHE_MAX_SIZE_GB": ("cache_max_size_gb", int),
        "MOMENTO_LOG_LEVEL": ("log_level", str),
        "MOMENTO_LOG_FORMAT": ("log_format", str),
        "MOMENTO_AUGMENTATION_COUNT": ("augmentation_count", int),
        "MOMENTO_VIDEO_FRAME_INTERVAL": ("video_frame_interval", float),
        "MOMENTO_INDEXING_BATCH_SIZE": ("indexing_batch_size", int),
    }

    for env_var, (config_field, converter) in _ENV_MAPPING.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            try:
                if converter == str:
                    setattr(config, config_field, env_value)
                else:
                    setattr(config, config_field, converter(env_value))
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse environment variable {env_var}: {e}")

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
        "embedding": {k: config_dict[k] for k in (
            "model_name",
        ) if k in config_dict},
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


def apply_config_overrides(config: MomentoConfig, cli_args: Optional[Dict[str, Any]] = None) -> MomentoConfig:
    """Apply CLI flag overrides to a MomentoConfig dataclass.

    Returns a new config with overrides applied, leaving module-level
    globals untouched.  Thread-safe and type-checker friendly.

    Args:
        config: Base configuration to apply overrides on top of.
        cli_args: Dict of config key → value from CLI flags.

    Returns:
        Updated MomentoConfig (same instance, modified in-place).
    """
    if cli_args is None:
        return config

    # Mapping of CLI flag names → MomentoConfig field names
    _OVERRIDE_MAP = {
        "threshold": "similarity_threshold",
        "batch_size": "indexing_batch_size",
        "max_results": "max_search_results",
        "log_format": "log_format",
        "log_level": "log_level",
    }

    for cli_key, field_name in _OVERRIDE_MAP.items():
        if cli_key in cli_args and cli_args[cli_key] is not None:
            if hasattr(config, field_name):
                setattr(config, field_name, cli_args[cli_key])

    return config


def get_device() -> str:
    """Return the current device string from DeviceManager.

    This helper performs a fresh device detection on each call,
    so it remains reliable even if the global singleton was
    initialized before environment changes or test mocks.
    """
    return DeviceManager().device