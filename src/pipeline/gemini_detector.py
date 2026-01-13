"""
Gemini-based unified detector for PII and faces.
Uses Google Gemini Vision API for detection.
"""

import uuid
import os
import json
from typing import List
from PIL import Image
import io
import base64
from google import genai
from ..config import (
    DETECTION_MODEL,
    MIN_FACE_CONFIDENCE,
    MIN_TEXT_CONFIDENCE,
    get_logger,
)

logger = get_logger(__name__)

from ..models import (
    PIIDetection,
    FaceDetection,
    BoundingBox,
    PIIType,
    DetectionType,
)


# System prompt for Gemini detection
DETECTION_PROMPT_TEMPLATE = """You are an AI system performing privacy-safe image redaction and replacement.

PHASE 1 — DETECTION (NO MODIFICATION)

Analyze the provided image (Resolution: {width}x{height}) and DETECT ONLY the following:

1. **Text containing personal information (PII)** - ONLY detect if text is CLEARLY VISIBLE and LEGIBLE:
   - Phone numbers (with digits visible)
   - Street addresses (with numbers and street names)
   - Personal names (when clearly readable)
   - Email addresses (with @ symbol visible)
   - Vehicle license plates / car number plates (with plate number visible)
   - QR codes (any visible QR code pattern)
   - Barcodes (any visible barcode pattern)
   - Signboards (store signs, shop names, business signs with text)
   
   DO NOT detect:
   - Blurry or illegible text
   - Generic logos without text
   - General labels or descriptions without personal info
   - Random patterns that look like text

2. **Human faces** - ONLY detect clear, recognizable human faces:
   - Must show eyes, nose, and mouth
   - Must be a real human face (not drawings, logos, or objects)
   
   DO NOT detect:
   - Partial faces or side profiles without clear features
   - Objects that vaguely resemble faces
   - Mannequins or statues unless very realistic

For EACH detected item, output a JSON object with:
- id: unique string identifier
- type: "text_pii" or "face"
- label: for text_pii: "phone" | "address" | "name" | "email" | "license_plate" | "qrcode" | "barcode" | "signboard" ; for face: "face"
- bbox: bounding box as [ymin, xmin, ymax, xmax] where:
  * VALUES MUST BE INTEGERS BETWEEN 0 AND 1000 (NORMALIZED COORDINATES)
  * 0 represents top/left edge, 1000 represents bottom/right edge
  * The image resolution is {width}x{height}, but you MUST return normalized 0-1000 coordinates relative to this full size.
  * [ymin, xmin, ymax, xmax] order (Standard Gemini format)
- confidence: number between 0 and 1

CRITICAL COORDINATE RULES:
- Coordinates are NORMALIZED to 0-1000 scale relative to the full {width}x{height} image
- x values range from 0 to 1000
- y values range from 0 to 1000
- Format: [ymin, xmin, ymax, xmax] where ymin < ymax and xmin < xmax
- DO NOT use pixel values, use 0-1000 scale

IMPORTANT:
- Detection ONLY in this phase
- Do NOT anonymize, blur, or modify anything
- Do NOT guess identities
- Face recognition or identity inference is NOT allowed
- Return ONLY valid JSON array of detections

Output format:
[
  {{
    "id": "uuid-1",
    "type": "text_pii",
    "label": "phone",
    "bbox": [200, 100, 250, 300],
    "confidence": 0.95
  }},
  {{
    "id": "uuid-2",
    "type": "face",
    "label": "face",
    "bbox": [600, 500, 900, 800],
    "confidence": 0.98
  }}
]

Return ONLY the JSON array, no additional text."""


class GeminiDetector:
    """Unified detector using Gemini Vision API."""

    def __init__(
        self,
        api_key: str = None,
        detection_model: str = DETECTION_MODEL,
        min_face_confidence: float = MIN_FACE_CONFIDENCE,
        min_text_confidence: float = MIN_TEXT_CONFIDENCE,
    ):
        """
        Initialize Gemini detector.

        Args:
            api_key: Google AI API key (if None, reads from GEMINI_API_KEY env var)
            detection_model: Gemini model to use for detection
            min_face_confidence: Minimum confidence threshold for face detections (0.0-1.0)
            min_text_confidence: Minimum confidence threshold for text PII detections (0.0-1.0)
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = genai.Client(api_key=api_key)
        self.detection_model = detection_model
        self.min_face_confidence = min_face_confidence
        self.min_text_confidence = min_text_confidence

    def detect(
        self, image: Image.Image
    ) -> tuple[List[PIIDetection], List[FaceDetection]]:
        """
        Detect PII and faces in an image using Gemini.

        Args:
            image: PIL Image object

        Returns:
            Tuple of (pii_detections, face_detections)
        """

        # Get original image dimensions
        orig_width, orig_height = image.size
        logger.info(f"Processing image: {orig_width}x{orig_height} pixels")

        # Skip resizing to ensure bbox coordinates map directly to original image
        scale_factor = 1.0
        processing_image = image
        proc_width, proc_height = orig_width, orig_height
        logger.info(f"Using original image size: {proc_width}x{proc_height}")

        # Format prompt with processing image dimensions
        detection_prompt = DETECTION_PROMPT_TEMPLATE.format(
            width=proc_width, height=proc_height
        )

        # Convert image to bytes for Gemini
        img_byte_arr = io.BytesIO()
        processing_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        # Call Gemini API with new client
        response = self.client.models.generate_content(
            model=self.detection_model,
            contents=[
                detection_prompt,
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(img_byte_arr.getvalue()).decode(),
                    }
                },
            ],
        )

        # Parse response
        try:
            # Extract JSON from response
            response_text = response.text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            detections = json.loads(response_text)

            if not isinstance(detections, list):
                raise ValueError("Response is not a list")

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.error(f"Response text: {response.text}")
            return [], []

        # Convert to our models
        pii_detections = []
        face_detections = []

        for det in detections:
            detection_id = det.get("id", str(uuid.uuid4()))
            detection_type = det.get("type")
            label = det.get("label")
            bbox_coords = det.get("bbox", [])
            confidence = det.get("confidence", 0.0)

            # Debug: Print original coordinates from Gemini
            logger.debug(f"Gemini returned {detection_type} ({label}):")
            logger.debug(f"  Original bbox: {bbox_coords}")
            logger.debug(f"  Processing size: {proc_width} x {proc_height}")

            # Filter by confidence threshold (different for faces vs text)
            if detection_type == "face" and confidence < self.min_face_confidence:
                logger.info(
                    f"Face confidence {confidence} below threshold {self.min_face_confidence}, skipping"
                )
                continue
            elif detection_type == "text_pii" and confidence < self.min_text_confidence:
                logger.info(
                    f"Text PII confidence {confidence} below threshold {self.min_text_confidence}, skipping"
                )
                continue

            # Parse bbox [ymin, xmin, ymax, xmax]
            if len(bbox_coords) != 4:
                logger.warning("Invalid bbox format, skipping")
                continue

            y_min, x_min, y_max, x_max = bbox_coords

            # Handle 0-1000 normalized coordinates (preferred)
            if all(0 <= c <= 1000 for c in bbox_coords) and any(
                c > 1 for c in bbox_coords
            ):
                logger.info("Detected 0-1000 normalized coords, converting to pixels")
                # Convert 0-1000 -> processing_image pixels
                x_min_proc = (x_min / 1000.0) * proc_width
                x_max_proc = (x_max / 1000.0) * proc_width
                y_min_proc = (y_min / 1000.0) * proc_height
                y_max_proc = (y_max / 1000.0) * proc_height

                # Map processing_image pixels -> original image pixels
                x_min = int(x_min_proc / scale_factor)
                x_max = int(x_max_proc / scale_factor)
                y_min = int(y_min_proc / scale_factor)
                y_max = int(y_max_proc / scale_factor)

            # Handle 0-1 normalized coordinates (fallback)
            elif all(0 <= c <= 1.0 for c in bbox_coords):
                logger.info("Detected 0-1 normalized coords, converting to pixels")
                x_min = int(x_min * orig_width)
                x_max = int(x_max * orig_width)
                y_min = int(y_min * orig_height)
                y_max = int(y_max * orig_height)

            # Handle absolute pixel coordinates (fallback)
            else:
                logger.info("Detected absolute pixel coords")
                # If we resized, we need to scale back?
                # But if model used proc_width/height, we need to know.
                # Assuming model followed instructions and used 0-1000, this block shouldn't be hit often.
                # If it returns pixels relative to resized image:
                if scale_factor != 1.0 and x_max <= proc_width and y_max <= proc_height:
                    logger.info("Scaling up from processing size")
                    x_min = int(x_min / scale_factor)
                    x_max = int(x_max / scale_factor)
                    y_min = int(y_min / scale_factor)
                    y_max = int(y_max / scale_factor)

            # Validate bbox
            if x_min >= x_max or y_min >= y_max:
                logger.warning(
                    "Invalid bbox: x_min >= x_max or y_min >= y_max, skipping"
                )
                continue

            # Clamp coordinates to original image bounds
            x_min = max(0, min(x_min, orig_width))
            x_max = max(0, min(x_max, orig_width))
            y_min = max(0, min(y_min, orig_height))
            y_max = max(0, min(y_max, orig_height))

            # Ensure bbox still valid after clamping
            if x_min >= x_max or y_min >= y_max:
                logger.warning("bbox became invalid after clamping, skipping")
                continue

            logger.debug(f"Final pixel bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")
            logger.debug(f"Bbox size: {x_max - x_min} x {y_max - y_min} pixels")

            # Create BoundingBox object
            bbox = BoundingBox(
                x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min
            )

            if detection_type == "text_pii":
                # Map label to PIIType
                pii_type_map = {
                    "phone": PIIType.PHONE,
                    "email": PIIType.EMAIL,
                    "address": PIIType.ADDRESS,
                    "name": PIIType.NAME,
                    "license_plate": PIIType.LICENSE_PLATE,
                    "qrcode": PIIType.QRCODE,
                    "barcode": PIIType.BARCODE,
                    "signboard": PIIType.SIGNBOARD,
                }
                pii_type = pii_type_map.get(label, PIIType.OTHER)

                pii_det = PIIDetection(
                    detection_id=detection_id,
                    detection_type=DetectionType.TEXT_PII,
                    pii_type=pii_type,
                    text=det.get("text", ""),  # Gemini might provide text
                    bbox=bbox,
                    confidence=confidence,
                )
                pii_detections.append(pii_det)

            elif detection_type == "face":
                face_det = FaceDetection(
                    detection_id=detection_id,
                    detection_type=DetectionType.FACE,
                    bbox=bbox,
                    confidence=confidence,
                )
                face_detections.append(face_det)

        return pii_detections, face_detections
