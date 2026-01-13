"""
Pipeline orchestrator that coordinates detection and anonymization.
"""

import uuid
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
from .config import (
    DEFAULT_FACE_METHOD,
    DEFAULT_TEXT_METHOD,
    MIN_FACE_CONFIDENCE,
    MIN_TEXT_CONFIDENCE,
    DETECTION_MODEL,
    IMAGEN_MODEL,
    S3_UPLOADS_PREFIX,
    S3_OUTPUTS_PREFIX,
    S3_ANONYMIZED_PREFIX,
)
from .models import (
    DetectionResult,
    PIIDetection,
    FaceDetection,
    AnonymizationRequest,
    AnonymizationResult,
    ReplacementMethod,
)
from .gemini_detector import GeminiDetector
from .image_generator import ImageGenerator
from .anonymizer import Anonymizer
from .s3_storage import S3Storage


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
        gemini_api_key: Optional[str] = None,
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
            gemini_api_key: Google Gemini API key (if None, will read from env)
            min_face_confidence: Minimum confidence for face detections
            min_text_confidence: Minimum confidence for text PII detections
            detection_model: Gemini model to use for detection
            imagen_model: Imagen model to use for image generation
            default_face_method: Default anonymization for faces (BLUR or GENERATE)
            default_text_method: Default anonymization for text PII (GENERATE or BLACK_BOX)
        """
        # Initialize S3 storage
        self.s3_storage = S3Storage()
        print(f"[Pipeline] S3 storage initialized")

        # Initialize Gemini detector
        self.detector = GeminiDetector(
            api_key=gemini_api_key,
            min_face_confidence=min_face_confidence,
            min_text_confidence=min_text_confidence,
            detection_model=detection_model,
        )

        # Initialize Image Generator with S3 storage
        self.generator = ImageGenerator(
            api_key=gemini_api_key,
            imagen_model=imagen_model,
            s3_storage=self.s3_storage,
        )

        self.anonymizer = Anonymizer(
            generator=self.generator,
            default_face_method=default_face_method,
            default_text_method=default_text_method,
        )

        # In-memory storage for images (for quick access)
        self._image_cache: Dict[str, Image.Image] = {}
        self._detection_cache: Dict[str, DetectionResult] = {}

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

        # Store image in memory for quick access
        self._image_cache[image_id] = image.copy()

        # Save original image to S3
        image_key = f"{S3_UPLOADS_PREFIX}{image_id}.png"
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

        # Cache the detection result
        self._detection_cache[image_id] = result

        return result

    def anonymize(
        self, request: AnonymizationRequest
    ) -> tuple[Image.Image, AnonymizationResult]:
        """
        Apply anonymization to selected detections.

        Args:
            request: AnonymizationRequest with selected replacements

        Returns:
            Tuple of (anonymized_image, result_info)
        """
        # Retrieve cached image and detections
        image = self._image_cache.get(request.image_id)
        detections = self._detection_cache.get(request.image_id)

        if image is None:
            raise ValueError(f"Image not found: {request.image_id}")

        if detections is None:
            raise ValueError(f"Detections not found for image: {request.image_id}")

        # Build a map of detection_id -> detection
        detection_map = {}
        for pii in detections.pii_detections:
            detection_map[pii.detection_id] = pii
        for face in detections.face_detections:
            detection_map[face.detection_id] = face

        # Prepare replacements for anonymizer
        replacements = []
        applied_ids = []

        for replacement_req in request.replacements:
            detection = detection_map.get(replacement_req.detection_id)
            if detection is None:
                continue

            bbox = detection.bbox
            method = replacement_req.method
            custom_data = replacement_req.custom_text

            label = None
            if isinstance(detection, PIIDetection):
                label = detection.pii_type
            elif isinstance(detection, FaceDetection):
                label = "face"

            replacements.append((bbox, method, custom_data, label))
            applied_ids.append(replacement_req.detection_id)

        # Apply anonymization
        anonymized_image = self.anonymizer.anonymize_image(image, replacements)

        # Save anonymized image to S3
        output_key = f"{S3_ANONYMIZED_PREFIX}{request.image_id}.png"
        self.s3_storage.upload_image(anonymized_image, output_key)
        message = f"Successfully anonymized {len(applied_ids)} regions. Saved to S3: {output_key}"

        # Create result
        result = AnonymizationResult(
            image_id=request.image_id,
            applied_replacements=applied_ids,
            output_format=request.output_format,
            message=message,
        )

        return anonymized_image, result

    def create_preview(
        self, image_id: str, show_labels: bool = True
    ) -> Optional[Image.Image]:
        """
        Create a preview image with all detections highlighted.

        Args:
            image_id: ID of the image
            show_labels: Whether to show labels on bounding boxes

        Returns:
            Preview image or None if not found
        """
        image = self._image_cache.get(image_id)
        detections = self._detection_cache.get(image_id)

        if image is None or detections is None:
            return None

        return self.anonymizer.create_preview_with_boxes(
            image,
            detections.pii_detections,
            detections.face_detections,
            show_labels=show_labels,
        )
