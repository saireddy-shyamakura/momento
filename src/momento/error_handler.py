"""
error_handler.py — Centralized error handling for Momento.

Provides consistent error handling, logging, and user-friendly messages.
"""

from typing import List, Optional
import sys

from .logger import get_logger

logger = get_logger(__name__)


class FeatureError(Exception):
    """Base exception for feature-specific errors."""
    pass


class ValidationError(FeatureError):
    """Raised when validation fails."""
    pass


class IndexingError(FeatureError):
    """Raised when indexing fails."""
    pass


class SearchError(FeatureError):
    """Raised when search fails."""
    pass


class IndexingErrorHandler:
    """Handles errors during indexing with graceful recovery."""
    
    def __init__(self):
        """Initialize error handler."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def add_error(self, feature_name: str, error: Exception) -> str:
        """Add and format an error message.
        
        Args:
            feature_name: Name of the feature that failed
            error: The exception that occurred
            
        Returns:
            Formatted error message
        """
        error_msg = f"{feature_name}: {str(error)}"
        self.errors.append(error_msg)
        logger.error(error_msg)
        return error_msg
    
    def add_warning(self, message: str) -> None:
        """Add a warning message.
        
        Args:
            message: Warning message
        """
        self.warnings.append(message)
        logger.warning(message)
    
    def should_continue_indexing(self) -> bool:
        """Determine if indexing should continue after an error.
        
        Returns:
            True to continue, False to stop
        """
        # Always continue - individual feature failures shouldn't stop indexing
        return True
    
    def get_error_summary(self) -> List[str]:
        """Get summary of all errors.
        
        Returns:
            List of error messages
        """
        return self.errors.copy()
    
    def get_warning_summary(self) -> List[str]:
        """Get summary of all warnings.
        
        Returns:
            List of warning messages
        """
        return self.warnings.copy()
    
    def has_errors(self) -> bool:
        """Check if any errors occurred.
        
        Returns:
            True if errors present
        """
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if any warnings occurred.
        
        Returns:
            True if warnings present
        """
        return len(self.warnings) > 0
    
    def print_summary(self) -> None:
        """Print error and warning summary."""
        if self.has_errors():
            print("\n⚠️  Errors encountered:")
            for error in self.errors[:5]:
                print(f"  - {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")
        
        if self.has_warnings():
            print("\nℹ️  Warnings:")
            for warning in self.warnings[:3]:
                print(f"  - {warning}")
            if len(self.warnings) > 3:
                print(f"  ... and {len(self.warnings) - 3} more")
    
    def reset(self) -> None:
        """Clear all error and warning messages."""
        self.errors.clear()
        self.warnings.clear()


def handle_fatal_error(message: str, exit_code: int = 1) -> None:
    """Handle a fatal error and exit.
    
    Args:
        message: Error message to display
        exit_code: Exit code to use
    """
    print(f"\n❌ Fatal Error: {message}", file=sys.stderr)
    logger.error(f"Fatal error: {message}")
    sys.exit(exit_code)


def handle_validation_error(message: str) -> None:
    """Handle a validation error.
    
    Args:
        message: Error message to display
        
    Raises:
        ValidationError
    """
    print(f"❌ Validation Error: {message}")
    logger.error(f"Validation error: {message}")
    raise ValidationError(message)


def handle_user_interrupt() -> None:
    """Handle user interrupt (Ctrl+C)."""
    print("\n\n👋 Goodbye!")
    logger.info("User interrupted execution")
    sys.exit(0)
