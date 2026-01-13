"""
Example usage script demonstrating the SafeLens pipeline with real image.
Uses the FastAPI endpoints to upload, detect, and anonymize images.
"""

from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import requests
import io

# Load environment variables from .env file
load_dotenv()

# Configuration
IMAGE_PATH = "./figures/example1.jpg"  # Change this to use a different image
API_BASE_URL = "http://localhost:8000"  # FastAPI server URL


def main():
    """Run the example pipeline with the specified image."""
    print("=" * 60)
    print("SafeLens Image Privacy Sanitization - Example Usage")
    print("=" * 60)

    # Load example image
    print("\n1. Loading example image...")
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        print(f"❌ Example image not found: {image_path}")
        print(f"   Please place an image at {image_path}")
        return

    image = Image.open(image_path)
    print(f"✓ Loaded image: {image_path}")
    print(f"   Size: {image.width}x{image.height} pixels")

    # Step 1: Upload image
    print("\n2. Uploading image to server...")
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return
    
    upload_result = response.json()
    image_id = upload_result["image_id"]
    print(f"✓ Image uploaded successfully!")
    print(f"   Image ID: {image_id}")

    # Step 2: Run detection
    print("\n3. Running detection with Gemini...")
    response = requests.post(f"{API_BASE_URL}/detect/{image_id}")
    
    if response.status_code != 200:
        print(f"❌ Detection failed: {response.text}")
        return
    
    result = response.json()
    print("✓ Detection complete!")
    print(f"   - Found {len(result['pii_detections'])} PII instances")
    print(f"   - Found {len(result['face_detections'])} faces")

    # Show detected items
    if result["pii_detections"]:
        print("\n4. Detected PII:")
        for i, pii in enumerate(result["pii_detections"], 1):
            print(f"   {i}. Type: {pii['pii_type']}")
            print(f"      Text: {pii.get('text', '(detected)')}")
            print(f"      Confidence: {pii['confidence']:.2f}")
            bbox = pii['bbox']
            print(f"      BBox: x={bbox['x']}, y={bbox['y']}, w={bbox['width']}, h={bbox['height']}")
    else:
        print("\n4. No PII detected")

    if result["face_detections"]:
        print("\n   Detected Faces:")
        for i, face in enumerate(result["face_detections"], 1):
            print(f"   {i}. Confidence: {face['confidence']:.2f}")
            bbox = face['bbox']
            print(f"      BBox: x={bbox['x']}, y={bbox['y']}, w={bbox['width']}, h={bbox['height']}")

    # Step 3: Prepare anonymization request
    print("\n5. Preparing anonymization request...")
    regions = []
    
    # Add all PII detections with GENERATE method
    for pii in result["pii_detections"]:
        regions.append({
            "bbox": pii["bbox"],
            "pii_type": pii["pii_type"]
        })
    
    # Add all face detections with GENERATE method
    for face in result["face_detections"]:
        regions.append({
            "bbox": face["bbox"],
            "pii_type": "face"
        })
    
    print(f"✓ Prepared {len(regions)} regions for anonymization")

    # Step 4: Apply anonymization
    print("\n6. Applying anonymization...")
    anonymize_request = {
        "image_id": image_id,
        "regions": regions,
        "method": "generate"
    }
    
    response = requests.post(f"{API_BASE_URL}/anonymize", json=anonymize_request)
    
    if response.status_code != 200:
        print(f"❌ Anonymization failed: {response.text}")
        return
    
    anon_result = response.json()
    anonymized_image_id = anon_result["anonymized_image_id"]
    print("✓ Anonymization complete!")
    print(f"   {anon_result['message']}")
    print(f"   Anonymized Image ID: {anonymized_image_id}")

    # Step 5: Download anonymized image
    print("\n7. Downloading anonymized image...")
    
    # Download high quality
    response = requests.get(f"{API_BASE_URL}/download/{anonymized_image_id}?quality=high")
    if response.status_code == 200:
        output_path = Path("debug") / f"{anonymized_image_id}_anonymized.png"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✓ Anonymized image saved as '{output_path}'")
    
    # Download original (low quality preview)
    response = requests.get(f"{API_BASE_URL}/download/{image_id}?quality=low")
    if response.status_code == 200:
        preview_path = Path("debug") / f"{image_id}_preview.png"
        with open(preview_path, "wb") as f:
            f.write(response.content)
        print(f"✓ Original preview saved as '{preview_path}'")

    print("\n" + "=" * 60)
    print("Example complete! Generated files:")
    print(f"  - debug/{image_id}_preview.png: Original (low quality)")
    print(f"  - debug/{anonymized_image_id}_anonymized.png: Anonymized")
    print("=" * 60)
    print("\nNote: All images are stored in S3 and can be accessed via API")
    print(f"      Original ID: {image_id}")
    print(f"      Anonymized ID: {anonymized_image_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
