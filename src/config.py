"""
Configuration constants for SaveLens.
"""

from .models import ReplacementMethod

# Detection Settings
MIN_CONFIDENCE = 0.8

# Default Anonymization Methods
DEFAULT_FACE_METHOD = ReplacementMethod.BLUR
DEFAULT_TEXT_METHOD = ReplacementMethod.GENERATE

# Image Generator Settings
MASK_PADDING = 10  # Padding around masked regions in pixels
