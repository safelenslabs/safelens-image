"""
Configuration package for SafeLens.
Consolidates settings, constants, and logging configuration.
"""

from .logger import setup_logging, get_logger
from .settings import Settings, get_settings
from .constants import (
    # Model Settings
    DETECTION_MODEL,
    IMAGEN_MODEL,
    # Detection Settings
    MIN_FACE_CONFIDENCE,
    MIN_TEXT_CONFIDENCE,
    # Default Anonymization Methods
    DEFAULT_FACE_METHOD,
    DEFAULT_TEXT_METHOD,
    # Image Generator Settings
    MASK_PADDING,
    # Image Quality Settings
    THUMBNAIL_MAX_WIDTH,
    # S3 Folder Structure
    S3_IMAGES_PREFIX,
    S3_DEBUG_PREFIX,
    S3_DEBUG_MASKED_PREFIX,
    S3_DEBUG_GEN_PREFIX,
)

__all__ = [
    # Logger
    "setup_logging",
    "get_logger",
    # Settings
    "Settings",
    "get_settings",
    # Model Settings
    "DETECTION_MODEL",
    "IMAGEN_MODEL",
    # Detection Settings
    "MIN_FACE_CONFIDENCE",
    "MIN_TEXT_CONFIDENCE",
    # Default Methods
    "DEFAULT_FACE_METHOD",
    "DEFAULT_TEXT_METHOD",
    # Image Settings
    "MASK_PADDING",
    "THUMBNAIL_MAX_WIDTH",
    # S3 Settings
    "S3_IMAGES_PREFIX",
    "S3_DEBUG_PREFIX",
    "S3_DEBUG_MASKED_PREFIX",
    "S3_DEBUG_GEN_PREFIX",
]
