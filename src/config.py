"""
Configuration constants for SafeLens.
"""

import os
from .models import ReplacementMethod

# Model Settings
DETECTION_MODEL = "gemini-3-flash-preview"  # Model for PII and face detection
# IMAGEN_MODEL = "gemini-2.5-flash-image"  # Model for image generation
IMAGEN_MODEL = "gemini-3-pro-image-preview"  # Model for image generation

# Detection Settings
MIN_FACE_CONFIDENCE = 0.7  # Minimum confidence for face detection
MIN_TEXT_CONFIDENCE = 0.8  # Minimum confidence for text PII detection

# Default Anonymization Methods
DEFAULT_FACE_METHOD = ReplacementMethod.BLUR
DEFAULT_TEXT_METHOD = ReplacementMethod.GENERATE

# Image Generator Settings
MASK_PADDING = 10  # Padding around masked regions in pixels

# Image Quality Settings
THUMBNAIL_MAX_WIDTH = (
    400  # Maximum width for low-quality thumbnails (height scales proportionally)
)

# S3 Storage Settings
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")  # S3 bucket name
S3_REGION_NAME = os.getenv("S3_REGION_NAME", "us-east-1")  # S3 region

# S3 Folder Structure
S3_UPLOADS_PREFIX = "uploads/"  # Prefix for uploaded images
S3_OUTPUTS_PREFIX = "outputs/"  # Prefix for anonymized images
S3_DEBUG_PREFIX = "debug/"  # Prefix for temporary debug images
S3_DEBUG_MASKED_PREFIX = "debug/masked/"  # Prefix for masked debug images
S3_DEBUG_GEN_PREFIX = "debug/gen/"  # Prefix for generated debug images
S3_ANONYMIZED_PREFIX = "outputs/anonymized/"  # Prefix for anonymized images
