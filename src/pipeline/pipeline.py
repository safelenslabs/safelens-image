"""
Pipeline orchestrator that coordinates detection and anonymization.
"""

import uuid
from typing import Optional
from PIL import Image
from ..config import (
    DEFAULT_FACE_METHOD,
    DEFAULT_TEXT_METHOD,
    MIN_FACE_CONFIDENCE,
    MIN_TEXT_CONFIDENCE,
    DETECTION_MODEL,
    IMAGEN_MODEL,
    S3_IMAGES_PREFIX,
)
from ..models import (
    DetectionResult,
    ReplacementMethod,
)
from .gemini_detector import GeminiDetector
from .image_generator import ImageGenerator
from .anonymizer import Anonymizer
from ..s3_storage import S3Storage
from ..settings import Settings


class PrivacyPipeline:
    """
    Main pipeline for image privacy sanitization.

    This orchestrates the entire workflow:
    1. Image upload
    2. Detection (PII text + faces) using Gemini Vision API
    3. User selection
    4. Anonymization
    """

    def __init__(
        self,
        settings: Settings,
        min_face_confidence: float = MIN_FACE_CONFIDENCE,
        min_text_confidence: float = MIN_TEXT_CONFIDENCE,
        detection_model: str = DETECTION_MODEL,
        imagen_model: str = IMAGEN_MODEL,
        default_face_method: ReplacementMethod = DEFAULT_FACE_METHOD,
        default_text_method: ReplacementMethod = DEFAULT_TEXT_METHOD,
    ):
        """
        Initialize the pipeline.

        Args:
            settings: Settings instance with API keys and S3 configuration
            min_face_confidence: Minimum confidence for face detections
            min_text_confidence: Minimum confidence for text PII detections
            detection_model: Gemini model to use for detection
            imagen_model: Imagen model to use for image generation
            default_face_method: Default anonymization for faces (BLUR or GENERATE)
            default_text_method: Default anonymization for text PII (GENERATE or BLACK_BOX)
        """
        # Initialize S3 storage
        self.s3_storage = S3Storage(settings)
        print(f"[Pipeline] S3 storage initialized")

        # Initialize Gemini detector
        self.detector = GeminiDetector(
            api_key=settings.gemini_api_key,
            min_face_confidence=min_face_confidence,
            min_text_confidence=min_text_confidence,
            detection_model=detection_model,
        )

        # Initialize Image Generator with S3 storage
        self.generator = ImageGenerator(
            api_key=settings.gemini_api_key,
            imagen_model=imagen_model,
            s3_storage=self.s3_storage,
        )

        self.anonymizer = Anonymizer(
            generator=self.generator,
            default_face_method=default_face_method,
            default_text_method=default_text_method,
        )

    def detect(
        self, image: Image.Image, image_id: Optional[str] = None
    ) -> DetectionResult:
        """
        Run detection pipeline on an image using Gemini Vision API.

        Args:
            image: PIL Image object
            image_id: Optional ID for the image (generated if not provided)

        Returns:
            DetectionResult with all detected PII and faces
        """
        if image_id is None:
            image_id = str(uuid.uuid4())

        # Save original image to S3
        image_key = f"{S3_IMAGES_PREFIX}{image_id}.png"
        self.s3_storage.upload_image(image, image_key)

        # Run Gemini detection (unified PII + face detection)
        pii_detections, face_detections = self.detector.detect(image)

        # Create result
        result = DetectionResult(
            image_id=image_id,
            pii_detections=pii_detections,
            face_detections=face_detections,
            image_width=image.width,
            image_height=image.height,
        )

        return result

    def anonymize(
        self,
        image_id: str,
        replacements: list,
    ) -> tuple[Image.Image, str]:
        """
        Apply anonymization to image regions.

        Args:
            image_id: ID of the image to anonymize
            replacements: List of (bbox, method, custom_data, label) tuples

        Returns:
            Tuple of (anonymized_image, output_key)
        """
        # Download image from S3
        image_key = f"{S3_IMAGES_PREFIX}{image_id}.png"
        image = self.s3_storage.download_image(image_key)

        if image is None:
            raise ValueError(f"Image not found: {image_id}")

        # Apply anonymization
        anonymized_image = self.anonymizer.anonymize_image(image, replacements)

        # Generate new UUID for anonymized image
        anonymized_id = str(uuid.uuid4())
        output_key = f"{S3_IMAGES_PREFIX}{anonymized_id}.png"
        self.s3_storage.upload_image(anonymized_image, output_key, format="PNG")

        return anonymized_image, output_key
