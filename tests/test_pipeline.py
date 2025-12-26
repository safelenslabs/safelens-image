"""
Basic tests for the SaveLens pipeline components.
Run with: pytest tests/
"""
import pytest
from PIL import Image
import numpy as np

from src.models import BoundingBox, PIIType, DetectionType, ReplacementMethod
from src.pipeline import PrivacyPipeline


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    # Create a 400x300 white image
    img = Image.new('RGB', (400, 300), color='white')
    return img


@pytest.fixture
def pipeline():
    """Create a pipeline instance for testing."""
    return PrivacyPipeline(
        use_enhanced_ner=False,
        face_min_confidence=0.9,
        languages=['en']
    )


class TestBoundingBox:
    """Test BoundingBox model."""
    
    def test_to_xyxy(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.to_xyxy() == (10, 20, 110, 70)
    
    def test_to_xywh(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.to_xywh() == (10, 20, 100, 50)


class TestPipeline:
    """Test the main pipeline."""
    
    def test_pipeline_initialization(self, pipeline):
        assert pipeline is not None
        assert pipeline.text_detector is not None
        assert pipeline.face_detector is not None
        assert pipeline.anonymizer is not None
    
    def test_detect_empty_image(self, pipeline, sample_image):
        """Test detection on an empty white image."""
        result = pipeline.detect(sample_image)
        
        assert result.image_id is not None
        assert result.image_width == 400
        assert result.image_height == 300
        assert isinstance(result.pii_detections, list)
        assert isinstance(result.face_detections, list)
    
    def test_cache_management(self, pipeline, sample_image):
        """Test cache storage and clearing."""
        result = pipeline.detect(sample_image)
        image_id = result.image_id
        
        # Check image is cached
        assert image_id in pipeline._image_cache
        assert image_id in pipeline._detection_cache
        
        # Clear cache
        pipeline.clear_cache(image_id)
        assert image_id not in pipeline._image_cache
        assert image_id not in pipeline._detection_cache
    
    def test_preview_generation(self, pipeline, sample_image):
        """Test preview image generation."""
        result = pipeline.detect(sample_image)
        preview = pipeline.create_preview(result.image_id, show_labels=True)
        
        assert preview is not None
        assert preview.size == sample_image.size


class TestModels:
    """Test data models."""
    
    def test_pii_type_enum(self):
        assert PIIType.PHONE == "phone"
        assert PIIType.EMAIL == "email"
        assert PIIType.NAME == "name"
    
    def test_detection_type_enum(self):
        assert DetectionType.TEXT_PII == "text_pii"
        assert DetectionType.FACE == "face"
    
    def test_replacement_method_enum(self):
        assert ReplacementMethod.MASK == "mask"
        assert ReplacementMethod.BLUR == "blur"
        assert ReplacementMethod.EMOJI == "emoji"


class TestTextDetector:
    """Test text detection and PII classification."""
    
    def test_phone_pattern_matching(self):
        """Test phone number pattern matching."""
        from src.text_detector import TextPIIDetector
        import re
        
        detector = TextPIIDetector()
        patterns = detector.patterns[PIIType.PHONE]
        
        test_numbers = [
            "555-123-4567",
            "555.123.4567",
            "555 123 4567",
            "+1-555-123-4567",
        ]
        
        for number in test_numbers:
            matched = any(re.search(pattern, number) for pattern in patterns)
            assert matched, f"Failed to match phone: {number}"
    
    def test_email_pattern_matching(self):
        """Test email pattern matching."""
        from src.text_detector import TextPIIDetector
        import re
        
        detector = TextPIIDetector()
        patterns = detector.patterns[PIIType.EMAIL]
        
        test_emails = [
            "test@example.com",
            "user.name+tag@example.co.uk",
            "test_email@subdomain.example.com",
        ]
        
        for email in test_emails:
            matched = any(re.search(pattern, email) for pattern in patterns)
            assert matched, f"Failed to match email: {email}"


class TestAnonymizer:
    """Test anonymization methods."""
    
    def test_black_box(self, sample_image):
        """Test black box anonymization."""
        from src.anonymizer import Anonymizer
        
        anonymizer = Anonymizer()
        bbox = BoundingBox(x=50, y=50, width=100, height=50)
        
        result = anonymizer._black_box(sample_image, bbox)
        
        assert result is not None
        assert result.size == sample_image.size
        
        # Check that the region is black
        pixels = np.array(result)
        region = pixels[50:100, 50:150]
        assert np.all(region == 0), "Region should be black"
    
    def test_blur_region(self, sample_image):
        """Test blur anonymization."""
        from src.anonymizer import Anonymizer
        
        anonymizer = Anonymizer()
        bbox = BoundingBox(x=50, y=50, width=100, height=50)
        
        result = anonymizer._blur_region(sample_image, bbox)
        
        assert result is not None
        assert result.size == sample_image.size


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
