import os
from pathlib import Path
from typing import Tuple
from config import SUPPORTED_EXTENSIONS


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
