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

try:
    from google import genai
except ImportError:
    raise ImportError("Please install google-genai: uv add google-genai")

from .models import (
    PIIDetection,
    FaceDetection,
    Polygon,
    PIIType,
    DetectionType,
)


# System prompt for Gemini detection
DETECTION_PROMPT = """You are an AI system performing privacy-safe image redaction and replacement.

PHASE 1 — DETECTION (NO MODIFICATION)

Analyze the provided image and DETECT the following:

1. Any visible text that may contain personal information (PII), including:
   - phone numbers
   - addresses
   - personal names
   - email addresses

2. Any human faces visible in the image.

For EACH detected item, output a JSON object with:
- id: unique string identifier
- type: "text_pii" or "face"
- label: for text_pii: "phone" | "address" | "name" | "email" ; for face: "face"
- polygon: list of [x, y] coordinates describing the region boundary (in image pixel space)
- confidence: number between 0 and 1

IMPORTANT:
- Detection ONLY in this phase
- Do NOT anonymize, blur, or modify anything
- Do NOT guess identities
- Face recognition or identity inference is NOT allowed
- Return ONLY valid JSON array of detections
- Polygon coordinates must be in image pixel coordinates

Output format:
[
  {
    "id": "uuid-1",
    "type": "text_pii",
    "label": "phone",
    "polygon": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "confidence": 0.95
  },
  {
    "id": "uuid-2",
    "type": "face",
    "label": "face",
    "polygon": [[x1, y1], [x2, y2], ...],
    "confidence": 0.98
  }
]

Return ONLY the JSON array, no additional text."""


class GeminiDetector:
    """Unified detector using Gemini Vision API."""
    
    def __init__(
        self, 
        api_key: str = None, 
        model_name: str = "gemini-2.0-flash-exp",
        min_confidence: float = 0.7
    ):
        """
        Initialize Gemini detector.
        
        Args:
            api_key: Google AI API key (if None, reads from GEMINI_API_KEY env var)
            model_name: Gemini model to use
            min_confidence: Minimum confidence threshold for detections (0.0-1.0)
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.min_confidence = min_confidence
    
    def detect(self, image: Image.Image) -> tuple[List[PIIDetection], List[FaceDetection]]:
        """
        Detect PII and faces in an image using Gemini.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (pii_detections, face_detections)
        """
        # Convert image to bytes for Gemini
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Call Gemini API with new client
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                DETECTION_PROMPT,
                {"inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(img_byte_arr.getvalue()).decode()
                }}
            ]
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
            print(f"Failed to parse Gemini response: {e}")
            print(f"Response text: {response.text}")
            return [], []
        
        # Convert to our models
        pii_detections = []
        face_detections = []
        
        for det in detections:
            detection_id = det.get("id", str(uuid.uuid4()))
            detection_type = det.get("type")
            label = det.get("label")
            polygon_coords = det.get("polygon", [])
            confidence = det.get("confidence", 0.0)
            
            # Filter by confidence threshold
            if confidence < self.min_confidence:
                continue
            
            # Convert polygon coordinates to tuples
            polygon_points = [(int(p[0]), int(p[1])) for p in polygon_coords]
            polygon = Polygon(points=polygon_points)
            
            if detection_type == "text_pii":
                # Map label to PIIType
                pii_type_map = {
                    "phone": PIIType.PHONE,
                    "email": PIIType.EMAIL,
                    "address": PIIType.ADDRESS,
                    "name": PIIType.NAME,
                }
                pii_type = pii_type_map.get(label, PIIType.OTHER)
                
                pii_det = PIIDetection(
                    detection_id=detection_id,
                    detection_type=DetectionType.TEXT_PII,
                    pii_type=pii_type,
                    text=det.get("text", ""),  # Gemini might provide text
                    polygon=polygon,
                    confidence=confidence
                )
                pii_detections.append(pii_det)
                
            elif detection_type == "face":
                face_det = FaceDetection(
                    detection_id=detection_id,
                    detection_type=DetectionType.FACE,
                    polygon=polygon,
                    confidence=confidence
                )
                face_detections.append(face_det)
        
        return pii_detections, face_detections
