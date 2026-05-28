import os
from pathlib import Path
from typing import Tuple
from .config import SUPPORTED_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_image_path(path: str) -> Tuple[bool, str]:
    """
    Validate that a path is a valid, readable image file.
    
    Args:
        path: Path to validate
        
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    # Check not empty
    if not path or not path.strip():
        return False, "Image path cannot be empty"
    
    path = path.strip()
    
    # Check exists
    if not os.path.exists(path):
        return False, f"File not found: {path}"
    
    # Check is file not directory
    if not os.path.isfile(path):
        return False, f"Path is not a file: {path}"
    
    # Check is readable
    if not os.access(path, os.R_OK):
        return False, f"File is not readable: {path}"
    
    # Check file extension
    if not path.lower().endswith(SUPPORTED_EXTENSIONS):
        return False, f"Invalid image format. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    
    # Check file size (avoid extremely large files)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    if file_size_mb > 500:
        return False, f"File too large ({file_size_mb:.1f}MB). Max: 500MB"
    
    return True, ""


def validate_video_path(path: str) -> Tuple[bool, str]:
    """
    Validate that a path is a valid, readable video file.

    Args:
        path: Path to validate

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not path or not path.strip():
        return False, "Video path cannot be empty"

    path = path.strip()

    if not os.path.exists(path):
        return False, f"File not found: {path}"

    if not os.path.isfile(path):
        return False, f"Path is not a file: {path}"

    if not os.access(path, os.R_OK):
        return False, f"File is not readable: {path}"

    if not path.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
        return False, f"Invalid video format. Supported: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    if file_size_mb > 5000:
        return False, f"File too large ({file_size_mb:.1f}MB). Max: 5GB"

    return True, ""


def validate_media_path(path: str) -> Tuple[bool, str]:
    """Validate an image or video path."""
    img_ok, _ = validate_image_path(path)
    if img_ok:
        return True, ""
    vid_ok, _ = validate_video_path(path)
    if vid_ok:
        return True, ""
    return False, f"Unsupported file. Supported images: {', '.join(SUPPORTED_EXTENSIONS)}, videos: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"


def validate_text_query(text: str) -> Tuple[bool, str]:
    """
    Validate text query for search.
    
    Args:
        text: Text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Text query cannot be empty"
    
    if len(text.strip()) > 1000:
        return False, "Text query too long (max 1000 characters)"
    
    return True, ""


def validate_folder_path(path: str) -> Tuple[bool, str]:
    """
    Validate that a path is a valid, accessible folder.
    
    Args:
        path: Path to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not path or not path.strip():
        return False, "Folder path cannot be empty"
    
    path = path.strip()
    
    if not os.path.exists(path):
        return False, f"Folder not found: {path}"
    
    if not os.path.isdir(path):
        return False, f"Path is not a directory: {path}"
    
    if not os.access(path, os.R_OK):
        return False, f"Folder is not readable: {path}"
    
    return True, ""


def validate_positive_int(value: int, name: str = "value") -> Tuple[bool, str]:
    """
    Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, int):
        return False, f"{name} must be an integer"
    
    if value <= 0:
        return False, f"{name} must be positive (got {value})"
    
    if value > 100:
        return False, f"{name} exceeds maximum (100)"
    
    return True, ""


def is_path_safe(file_path: str, base_folder: str) -> Tuple[bool, str]:
    """Check that a file path does not escape its base folder via symlinks.

    Resolves symlinks with os.path.realpath() and verifies the resolved
    path is inside base_folder. This prevents path traversal attacks
    where a symlink like ``Pictures/../../etc/passwd`` would otherwise
    be followed.

    Args:
        file_path: Absolute path to the candidate file.
        base_folder: Absolute path to the allowed parent folder.

    Returns:
        Tuple of (is_safe, error_message).
    """
    try:
        resolved_file = os.path.realpath(file_path)
        resolved_base = os.path.realpath(base_folder)

        if not resolved_file.startswith(resolved_base + os.sep):
            return (False,
                    f"Path resolves outside the indexed folder: "
                    f"{file_path} → {resolved_file}")
        return True, ""
    except (OSError, ValueError) as e:
        return False, f"Cannot resolve path {file_path}: {e}"


def validate_choice(choice: str, valid_options: list) -> Tuple[bool, str]:
    """
    Validate that a choice is in valid options.
    
    Args:
        choice: User choice
        valid_options: List of valid choices
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not choice or not choice.strip():
        return False, "Choice cannot be empty"
    
    choice = choice.strip()
    
    if choice not in valid_options:
        options_str = ", ".join(str(o) for o in valid_options)
        return False, f"Invalid choice. Valid options: {options_str}"
    
    return True, ""
