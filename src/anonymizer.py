"""
Anonymization module for applying various replacement methods to detected regions.
Supports both polygon and bounding box regions.
"""
import io
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import cv2
from .models import (
    BoundingBox,
    ReplacementMethod,
    PIIDetection,
    FaceDetection,
    ReplacementRequest,
)


class Anonymizer:
    """Applies anonymization techniques to image regions."""
    
    def __init__(
        self,
        generator=None,
        default_face_method: ReplacementMethod = ReplacementMethod.BLUR,
        default_text_method: ReplacementMethod = ReplacementMethod.BLUR
    ):
        """Initialize anonymizer with default settings.
        
        Args:
            generator: Object with generate_replacement method (e.g. GeminiDetector)
            default_face_method: Default method for face anonymization (BLUR or INPAINT recommended)
            default_text_method: Default method for text PII anonymization (MASK or REDACT recommended)
        """
        self.generator = generator
        self.default_face_method = default_face_method
        self.default_text_method = default_text_method
    
    def anonymize_image(
        self,
        image: Image.Image,
        replacements: List[Tuple[BoundingBox, ReplacementMethod, Optional[str], Optional[str]]]
    ) -> Image.Image:
        """
        Apply multiple anonymization replacements to an image.
        
        Args:
            image: Original PIL Image
            replacements: List of (bbox, method, custom_data, label) tuples
            
        Returns:
            Anonymized PIL Image
        """
        # Work on a copy
        result = image.copy()
        
        for item in replacements:
            if len(item) == 4:
                region, method, custom_data, label = item
            else:
                region, method, custom_data = item
                label = None
            result = self._apply_replacement(result, region, method, custom_data, label)
        
        return result
    
    def _apply_replacement(
        self,
        image: Image.Image,
        region: BoundingBox,
        method: ReplacementMethod,
        custom_data: Optional[str] = None,
        label: Optional[str] = None
    ) -> Image.Image:
        """Apply a single replacement to an image region."""
        
        if method == ReplacementMethod.GENERATE:
            return self._generate_region(image, region, label)
        elif method == ReplacementMethod.BLUR:
            return self._blur_region(image, region)
        elif method == ReplacementMethod.BLACK_BOX:
            return self._black_box(image, region)
        else:
            # Default to black box for unknown methods
            return self._black_box(image, region)
    
    def _get_region_mask(self, image: Image.Image, region: BoundingBox) -> Image.Image:
        """Create a mask for the given bounding box region."""
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        x1, y1, x2, y2 = region.to_xyxy()
        draw.rectangle([x1, y1, x2, y2], fill=255)
        return mask
    
    def _generate_region(self, image: Image.Image, region: BoundingBox, label: str = None) -> Image.Image:
        """Use generative AI to fill the region."""
        if not self.generator:
            print("Warning: No generator provided for GENERATE method. Falling back to BLUR.")
            return self._blur_region(image, region)
        
        try:
            # Call generator to get a patch
            patch = self.generator.generate_replacement(image, region, label)
            if patch:
                result = image.copy()
                x1, y1, x2, y2 = region.to_xyxy()
                # Resize patch to fit region if needed
                patch = patch.resize((region.width, region.height))
                result.paste(patch, (x1, y1))
                return result
            else:
                return self._blur_region(image, region)
        except Exception as e:
            print(f"Error in generation: {e}")
            return self._blur_region(image, region)

    def _black_box(self, image: Image.Image, region: BoundingBox) -> Image.Image:
        """Draw a black rectangle over the region."""
        result = image.copy()
        draw = ImageDraw.Draw(result)
        x1, y1, x2, y2 = region.to_xyxy()
        draw.rectangle([x1, y1, x2, y2], fill='black')
        return result
    
    def _blur_region(
        self, 
        image: Image.Image, 
        region: BoundingBox,
        blur_radius: int = 15
    ) -> Image.Image:
        """Apply Gaussian blur to the region."""
        result = image.copy()
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = region.to_xyxy()
        
        # Extract and blur region
        region_crop = result.crop((x1, y1, x2, y2))
        blurred = region_crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Paste back
        result.paste(blurred, (x1, y1))
        
        return result
    
    def create_preview_with_boxes(
        self,
        image: Image.Image,
        pii_detections: List[PIIDetection],
        face_detections: List[FaceDetection],
        show_labels: bool = True
    ) -> Image.Image:
        """
        Create a preview image with polygons/boxes drawn around detections.
        
        Useful for showing users what was detected before anonymization.
        
        Args:
            image: Original image
            pii_detections: List of PII detections
            face_detections: List of face detections
            show_labels: Whether to show labels on boxes
            
        Returns:
            Preview image with regions drawn
        """
        result = image.copy()
        draw = ImageDraw.Draw(result)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Draw PII regions in red
        for pii in pii_detections:
            bbox = pii.bbox
            x1, y1, x2, y2 = bbox.to_xyxy()
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            
            if show_labels:
                label = f"{pii.pii_type.value} ({pii.confidence:.2f})"
                draw.text((x1, y1 - 15), label, fill='red', font=font)
        
        # Draw face regions in blue
        for face in face_detections:
            bbox = face.bbox
            x1, y1, x2, y2 = bbox.to_xyxy()
            draw.rectangle([x1, y1, x2, y2], outline='blue', width=2)
            
            if show_labels:
                label = f"Face ({face.confidence:.2f})"
                draw.text((x1, y1 - 15), label, fill='blue', font=font)
        
        return result
    
    def anonymize_detections(
        self,
        image: Image.Image,
        pii_detections: List[PIIDetection],
        face_detections: List[FaceDetection],
        face_method: Optional[ReplacementMethod] = None,
        text_method: Optional[ReplacementMethod] = None,
        custom_data: Optional[str] = None
    ) -> Image.Image:
        """Convenience method to anonymize all detections with specified methods.
        
        Args:
            image: Original image
            pii_detections: List of PII detections to anonymize
            face_detections: List of face detections to anonymize
            face_method: Method for faces (default: BLUR, can use INPAINT for generative)
            text_method: Method for text PII (default: MASK)
            custom_data: Optional custom data for certain methods (e.g., emoji type)
            
        Returns:
            Anonymized image
        """
        # Use default methods if not specified
        face_method = face_method or self.default_face_method
        text_method = text_method or self.default_text_method
        
        # Build replacements list
        replacements = []
        
        # Add PII replacements
        for pii in pii_detections:
            replacements.append((pii.polygon, text_method, custom_data))
        
        # Add face replacements
        for face in face_detections:
            replacements.append((face.polygon, face_method, custom_data))
        
        # Apply all replacements
        return self.anonymize_image(image, replacements)
