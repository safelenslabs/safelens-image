"""
FastAPI service for image privacy sanitization.

This service provides REST API endpoints for:
1. Upload and detect PII/faces in images using Gemini Vision API
2. Get detection results
3. Apply selective anonymization
4. Download anonymized images
"""

import io
import os
import logging
from typing import Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from src.pipeline import PrivacyPipeline
from src.models import (
    DetectionResult,
    AnonymizationRequest,
    AnonymizationResult,
    ReplacementMethod,
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global pipeline instance
pipeline: Optional[PrivacyPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the pipeline."""
    global pipeline
    logger.info("Initializing Privacy Pipeline with Gemini Vision API...")

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set in environment. Pipeline may fail.")

    # Get default methods from environment (optional)
    default_face_method = os.getenv("DEFAULT_FACE_METHOD", "blur").upper()
    default_text_method = os.getenv("DEFAULT_TEXT_METHOD", "mask").upper()

    # Map to enum
    face_method = getattr(
        ReplacementMethod, default_face_method, ReplacementMethod.BLUR
    )
    text_method = getattr(
        ReplacementMethod, default_text_method, ReplacementMethod.MASK
    )

    # Initialize with Gemini detector
    pipeline = PrivacyPipeline(
        gemini_api_key=api_key,
        default_face_method=face_method,
        default_text_method=text_method,
    )

    logger.info("Pipeline initialized successfully")
    yield

    # Cleanup
    logger.info("Shutting down...")
    if pipeline:
        pipeline.clear_cache()


# Create FastAPI app
app = FastAPI(
    title="SaveLens Image Privacy Sanitization API",
    description="Privacy-safe image sanitization with PII detection and face detection",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "SaveLens Image Privacy Sanitization API",
        "version": "0.1.0",
        "status": "healthy",
    }


@app.post("/api/v1/detect", response_model=DetectionResult)
async def detect_pii_and_faces(
    file: UploadFile = File(..., description="Image file to analyze"),
) -> DetectionResult:
    """
    Upload an image and detect PII and faces.

    This endpoint:
    1. Accepts an image file
    2. Runs OCR to extract text
    3. Classifies PII in the text
    4. Detects faces (no identity recognition)
    5. Returns all detections with bounding boxes

    Returns:
        DetectionResult with image_id and all detected regions
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        logger.info(f"Processing image: {file.filename}, size: {image.size}")

        # Run detection pipeline
        result = pipeline.detect(image)

        logger.info(
            f"Detection complete for {result.image_id}: "
            f"{len(result.pii_detections)} PII, {len(result.face_detections)} faces"
        )

        return result

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/api/v1/anonymize", response_model=AnonymizationResult)
async def anonymize_image(request: AnonymizationRequest) -> AnonymizationResult:
    """
    Apply anonymization to selected detections.

    This endpoint:
    1. Takes a list of detection IDs and replacement methods
    2. Applies the specified anonymization techniques
    3. Returns metadata about the operation

    Use GET /api/v1/download/{image_id} to download the anonymized image.

    Returns:
        AnonymizationResult with applied replacement IDs
    """
    try:
        logger.info(
            f"Anonymizing image {request.image_id} with {len(request.replacements)} replacements"
        )

        # Apply anonymization
        anonymized_image, result = pipeline.anonymize(request)

        # Store the anonymized image back in cache
        # This allows downloading it later
        pipeline._image_cache[f"{request.image_id}_anonymized"] = anonymized_image

        logger.info(f"Anonymization complete for {request.image_id}")

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error anonymizing image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error anonymizing image: {str(e)}"
        )


@app.get("/api/v1/preview/{image_id}")
async def get_preview(image_id: str, show_labels: bool = True) -> Response:
    """
    Get a preview image with detection bounding boxes drawn.

    Args:
        image_id: ID of the image from detection step
        show_labels: Whether to show labels on bounding boxes

    Returns:
        PNG image with bounding boxes
    """
    try:
        preview = pipeline.create_preview(image_id, show_labels=show_labels)

        if preview is None:
            raise HTTPException(status_code=404, detail="Image not found")

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        preview.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating preview: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating preview: {str(e)}"
        )


@app.get("/api/v1/download/{image_id}")
async def download_anonymized_image(image_id: str, format: str = "png") -> Response:
    """
    Download the anonymized image.

    Args:
        image_id: ID of the image (from detection step)
        format: Output format (png, jpg, webp)

    Returns:
        Anonymized image file
    """
    try:
        # Look for the anonymized version
        anonymized_key = f"{image_id}_anonymized"
        image = pipeline._image_cache.get(anonymized_key)

        if image is None:
            # Fallback to original if not anonymized yet
            image = pipeline._image_cache.get(image_id)

        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")

        # Validate format
        format = format.lower()
        if format not in ["png", "jpg", "jpeg", "webp"]:
            raise HTTPException(status_code=400, detail="Invalid format")

        # Convert format
        if format in ["jpg", "jpeg"]:
            format = "JPEG"
            media_type = "image/jpeg"
            # Convert RGBA to RGB for JPEG
            if image.mode == "RGBA":
                image = image.convert("RGB")
        elif format == "webp":
            format = "WEBP"
            media_type = "image/webp"
        else:
            format = "PNG"
            media_type = "image/png"

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format, quality=95)
        img_byte_arr.seek(0)

        return StreamingResponse(
            img_byte_arr,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=anonymized_{image_id}.{format.lower()}"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error downloading image: {str(e)}"
        )


@app.delete("/api/v1/clear/{image_id}")
async def clear_image_cache(image_id: str) -> dict:
    """
    Clear cached image and detection data.

    Args:
        image_id: ID of the image to clear

    Returns:
        Success message
    """
    try:
        pipeline.clear_cache(image_id)
        # Also clear anonymized version
        pipeline.clear_cache(f"{image_id}_anonymized")

        return {"message": f"Cache cleared for image {image_id}"}

    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


@app.delete("/api/v1/clear")
async def clear_all_cache() -> dict:
    """
    Clear all cached images and detections.

    Returns:
        Success message
    """
    try:
        pipeline.clear_cache()
        return {"message": "All cache cleared"}

    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
