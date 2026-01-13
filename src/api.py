"""
FastAPI service for image privacy sanitization.

This service provides REST API endpoints for:
1. Upload and detect PII/faces in images using Gemini Vision API
2. Get detection results
3. Apply selective anonymization
4. Download anonymized images
"""

import io
import logging
import uuid
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
    ReplacementMethod,
    BoundingBox,
)
from src.config import (
    THUMBNAIL_MAX_WIDTH,
    S3_IMAGES_PREFIX,
    get_settings,
    setup_logging,
    get_logger,
)
from pydantic import BaseModel
from typing import List

# Load environment variables from .env file
load_dotenv()

# Configure logging
setup_logging(level=logging.INFO)
logger = get_logger(__name__)


# Global pipeline instance
pipeline: Optional[PrivacyPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the pipeline."""
    global pipeline
    logger.info("Initializing Privacy Pipeline with Gemini Vision API...")

    # Load settings
    settings = get_settings()

    # Initialize pipeline with settings
    pipeline = PrivacyPipeline(
        settings=settings,
    )

    logger.info("Pipeline initialized successfully")
    yield

    # Cleanup
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="SafeLens Image Privacy Sanitization API",
    description="Privacy-safe image sanitization with PII detection and face detection",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "chrome-extension://*",
    ],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint for Docker and load balancers."""
    return {"status": "healthy", "service": "safelens-image", "version": "0.1.0"}


# ===== API Endpoints =====


class UploadResponse(BaseModel):
    """Response model for image upload."""

    image_id: str
    message: str


class BboxWithLabel(BaseModel):
    """Bounding box with optional PII type label."""

    bbox: BoundingBox
    pii_type: Optional[str] = None  # PIIType value or "face"


class BboxAnonymizeRequest(BaseModel):
    """Request model for bbox-based anonymization."""

    image_id: str
    regions: List[BboxWithLabel]  # Changed from bboxes to regions with labels
    method: ReplacementMethod = ReplacementMethod.GENERATE


class AnonymizeResponse(BaseModel):
    """Response model for anonymization."""

    original_image_id: str
    anonymized_image_id: str
    success: bool
    message: str
    processed_count: int


@app.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(..., description="Image file to upload"),
) -> UploadResponse:
    """
    1. Upload an image and get a UUID.

    This endpoint:
    - Accepts an image file
    - Generates a unique UUID
    - Stores the image in cache
    - Returns the UUID for later processing

    Returns:
        UploadResponse with image_id (UUID)
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        logger.info(f"Uploading image: {file.filename}, size: {image.size}")

        # Generate UUID
        image_id = str(uuid.uuid4())

        # Generate thumbnail (low quality)
        thumbnail = image.copy()
        if thumbnail.width > THUMBNAIL_MAX_WIDTH:
            ratio = THUMBNAIL_MAX_WIDTH / thumbnail.width
            new_height = int(thumbnail.height * ratio)
            thumbnail = thumbnail.resize(
                (THUMBNAIL_MAX_WIDTH, new_height), Image.Resampling.LANCZOS
            )

        # Save to S3
        image_key = f"{S3_IMAGES_PREFIX}{image_id}.png"
        pipeline.s3_storage.upload_image(image, image_key, format="PNG")

        # Save thumbnail
        thumbnail_key = f"{S3_IMAGES_PREFIX}{image_id}_low.png"
        pipeline.s3_storage.upload_image(thumbnail, thumbnail_key, format="PNG")

        logger.info(f"Image uploaded to S3 with ID: {image_id}")

        return UploadResponse(image_id=image_id, message="Image uploaded successfully")

    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")


@app.post("/detect/{image_id}", response_model=DetectionResult)
async def detect_from_uuid(image_id: str) -> DetectionResult:
    """
    2. Detect PII and faces from uploaded image UUID.

    This endpoint:
    - Takes an image_id (UUID) from upload step
    - Runs detection on the stored image
    - Returns all detected PII and faces with bounding boxes

    Returns:
        DetectionResult with all detections
    """
    try:
        # Download image from S3
        image_key = f"{S3_IMAGES_PREFIX}{image_id}.png"
        image = pipeline.s3_storage.download_image(image_key)

        if image is None:
            raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

        logger.info(f"Running detection on image: {image_id}")

        # Run detection
        result = pipeline.detect(image, image_id=image_id)

        logger.info(
            f"Detection complete for {image_id}: "
            f"{len(result.pii_detections)} PII, {len(result.face_detections)} faces"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error detecting image: {str(e)}")


@app.post("/anonymize", response_model=AnonymizeResponse)
async def anonymize_with_bboxes(request: BboxAnonymizeRequest) -> AnonymizeResponse:
    """
    3. Anonymize image regions using bounding box coordinates and PII types.

    This endpoint:
    - Takes image_id (UUID) and list of regions with bounding boxes and PII types
    - Applies anonymization (GENERATE or BLUR) to each region
    - Returns original and anonymized image UUIDs

    Returns:
        AnonymizeResponse with success status and both image UUIDs
    """
    try:
        logger.info(
            f"Anonymizing image {request.image_id} with {len(request.regions)} regions"
        )

        # Prepare replacements from regions
        replacements = []
        for region in request.regions:
            label = region.pii_type
            replacements.append((region.bbox, request.method, None, label))

        # Apply anonymization (downloads from S3 internally)
        anonymized_image, output_key = pipeline.anonymize(
            request.image_id, replacements
        )

        # Generate UUID for anonymized image
        anonymized_image_id = str(uuid.uuid4())

        # Generate thumbnail (low quality)
        thumbnail = anonymized_image.copy()
        if thumbnail.width > THUMBNAIL_MAX_WIDTH:
            ratio = THUMBNAIL_MAX_WIDTH / thumbnail.width
            new_height = int(thumbnail.height * ratio)
            thumbnail = thumbnail.resize(
                (THUMBNAIL_MAX_WIDTH, new_height), Image.Resampling.LANCZOS
            )

        # Save with new UUID
        final_output_key = f"{S3_IMAGES_PREFIX}{anonymized_image_id}.png"
        pipeline.s3_storage.upload_image(
            anonymized_image, final_output_key, format="PNG"
        )

        # Save thumbnail
        thumbnail_key = f"{S3_IMAGES_PREFIX}{anonymized_image_id}_low.png"
        pipeline.s3_storage.upload_image(thumbnail, thumbnail_key, format="PNG")

        logger.info(
            f"Anonymization complete: {request.image_id} -> {anonymized_image_id}"
        )

        return AnonymizeResponse(
            original_image_id=request.image_id,
            anonymized_image_id=anonymized_image_id,
            success=True,
            message=f"Successfully anonymized {len(request.regions)} regions",
            processed_count=len(request.regions),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error anonymizing image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error anonymizing image: {str(e)}"
        )


@app.get("/download/{image_id}")
async def download_image(image_id: str, quality: str = "high") -> Response:
    """
    4. Download image by UUID.

    Args:
        image_id: UUID of the image (original or anonymized)
        quality: Image quality - "high" (original) or "low" (thumbnail, 512px width) (default: "high")

    Returns:
        Image file
    """
    try:
        # Validate quality parameter
        if quality not in ["high", "low"]:
            raise HTTPException(
                status_code=400, detail="Invalid quality. Use 'high' or 'low'"
            )

        # Determine image key based on quality
        if quality == "low":
            image_key = f"{S3_IMAGES_PREFIX}{image_id}_low.png"
        else:
            image_key = f"{S3_IMAGES_PREFIX}{image_id}.png"

        # Download image
        image = pipeline.s3_storage.download_image(image_key)

        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")

        # Convert to PNG bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={image_id}.png"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error downloading image: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
