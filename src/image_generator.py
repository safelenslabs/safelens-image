"""
Image generator module using Gemini and Imagen.
"""

import io
import os
import uuid
from typing import Optional
from PIL import Image, ImageDraw
from google import genai
from .models import BoundingBox, PIIType, PII_REPLACEMENT_VALUES
from .config import IMAGEN_MODEL, MASK_PADDING


class ImageGenerator:
    """Generates image patches using Imagen Inpainting."""

    def __init__(
        self,
        api_key: str = None,
        imagen_model: str = IMAGEN_MODEL,
        mask_padding: int = MASK_PADDING,
    ):
        """
        Initialize Image Generator.

        Args:
            api_key: Google AI API key
            imagen_model: Model to use for image generation (default: gemini-2.5-flash-image)
            mask_padding: Padding (in pixels) to add around the bbox for better context (default: 10)
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = genai.Client(api_key=api_key)
        self.imagen_model = imagen_model
        self.mask_padding = mask_padding

    def generate_replacement(
        self, image: Image.Image, region: BoundingBox, label: str = None
    ) -> Optional[Image.Image]:
        """
        Generate a replacement patch for the given region using Gemini.

        Args:
            image: Original image
            region: Bounding box of the region to replace
            label: Label of the object being replaced (PIIType value or "face")

        Returns:
            Full modified image (not a patch)
        """
        try:
            # 1. Create a copy of the image with the region blacked out
            masked_image = image.copy()
            draw = ImageDraw.Draw(masked_image)
            x1, y1, x2, y2 = region.to_xyxy()

            # Add padding for better context
            mask_x1 = max(0, x1 - self.mask_padding)
            mask_y1 = max(0, y1 - self.mask_padding)
            mask_x2 = min(image.width, x2 + self.mask_padding)
            mask_y2 = min(image.height, y2 + self.mask_padding)

            # Draw black rectangle on the area to be replaced
            draw.rectangle([mask_x1, mask_y1, mask_x2, mask_y2], fill="black")

            print(f"[DEBUG] Original bbox: ({x1}, {y1}) to ({x2}, {y2})")
            print(
                f"[DEBUG] Blacked area: ({mask_x1}, {mask_y1}) to ({mask_x2}, {mask_y2}) [padding: {self.mask_padding}px]"
            )

            # Debug: Save masked image
            mask_uuid = uuid.uuid4()
            masked_path = f"temp/debug_masked_{mask_uuid}.png"
            masked_image.save(masked_path)
            print(f"[DEBUG] Saved masked image to {masked_path}")

            # 2. Call Gemini Generate Content
            try:
                from google.genai import types
                import io

                prompt_text = (
                    "You are an expert image editor. I am providing you with images:\n"
                    "1. The ORIGINAL image (first image)\n"
                    "2. A MASKED version where certain areas are blacked out (second image)\n"
                )

                # Check if we need to include a reference image for QR/barcode
                reference_image = None
                if label:
                    label_str = (
                        label.value
                        if isinstance(label, PIIType)
                        else str(label).lower()
                    )
                    replacement_value = None

                    if isinstance(label, PIIType):
                        replacement_value = PII_REPLACEMENT_VALUES.get(label)
                    elif isinstance(label, str):
                        try:
                            pii_type = PIIType(label)
                            replacement_value = PII_REPLACEMENT_VALUES.get(pii_type)
                        except (ValueError, KeyError):
                            pass

                    # Load reference image if it's a path to QR/barcode
                    if (
                        replacement_value
                        and isinstance(replacement_value, str)
                        and (
                            replacement_value.endswith(".png")
                            or replacement_value.endswith(".jpg")
                        )
                    ):
                        try:
                            ref_path = (
                                replacement_value
                                if os.path.isabs(replacement_value)
                                else os.path.join(
                                    os.path.dirname(os.path.dirname(__file__)),
                                    replacement_value,
                                )
                            )
                            if os.path.exists(ref_path):
                                reference_image = Image.open(ref_path)
                                prompt_text += "3. A REFERENCE image showing the desired QR code or barcode design (third image)\n\n"
                                print(f"[DEBUG] Loaded reference image from {ref_path}")
                        except Exception as e:
                            print(f"[WARNING] Failed to load reference image: {e}")

                if not reference_image:
                    prompt_text += "\n"

                prompt_text += (
                    "Your task: Use the original image as reference, and fill in the black areas in the masked image naturally and seamlessly. "
                    "The filled areas should match the surrounding context, lighting, texture, and overall composition perfectly. "
                    "Do not modify any other parts of the image - only fill in the black regions in the masked image. "
                    "Return the complete restored image with the same dimensions as the input."
                )

                if label:
                    # Get replacement value from predefined constants
                    replacement_value = None
                    label_str = None

                    # Handle both PIIType enum and string labels
                    if isinstance(label, PIIType):
                        replacement_value = PII_REPLACEMENT_VALUES.get(label)
                        label_str = label.value
                    elif isinstance(label, str):
                        try:
                            pii_type = PIIType(label)
                            replacement_value = PII_REPLACEMENT_VALUES.get(pii_type)
                            label_str = label
                        except (ValueError, KeyError):
                            label_str = label
                    else:
                        label_str = str(label)

                    label_lower = label_str.lower()

                    if replacement_value:
                        # Use predefined replacement value
                        if "license" in label_lower or "plate" in label_lower:
                            prompt_text += f"\n\nIMPORTANT: The black area contained a license plate. Fill it with a realistic license plate showing '{replacement_value}', maintaining natural appearance and style."
                        elif (
                            "phone" in label_lower
                            or "telephone" in label_lower
                            or "mobile" in label_lower
                        ):
                            prompt_text += f"\n\nIMPORTANT: The black area contained a phone number. Fill it with the phone number '{replacement_value}' in the same format and style."
                        elif "id" in label_lower or "card" in label_lower:
                            prompt_text += f"\n\nIMPORTANT: The black area contained identification information. Fill it with '{replacement_value}' in the same format and style."
                        elif "address" in label_lower or "postal" in label_lower:
                            prompt_text += f"\n\nIMPORTANT: The black area contained address information. Fill it with '{replacement_value}' in the same format and style."
                        elif "email" in label_lower:
                            prompt_text += f"\n\nIMPORTANT: The black area contained an email address. Fill it with '{replacement_value}' in the same format and style."
                        elif "name" in label_lower:
                            prompt_text += f"\n\nIMPORTANT: The black area contained a name. Fill it with '{replacement_value}' in the same format and style."
                        elif "birth" in label_lower or "date" in label_lower:
                            prompt_text += f"\n\nIMPORTANT: The black area contained a date. Fill it with '{replacement_value}' in the same format and style."
                        elif "qr" in label_lower or "qrcode" in label_lower:
                            if reference_image:
                                prompt_text += "\n\nIMPORTANT: The black area contained a QR code. Fill it with the QR code shown in the REFERENCE image (third image). Copy the reference QR code EXACTLY as it appears - same design, colors, background style, border, pattern, and size. Do not modify or regenerate it, just replicate it precisely."
                            else:
                                prompt_text += "\n\nIMPORTANT: The black area contained a QR code. Fill it with a QR code that EXACTLY matches the original design - same colors, background style, size, border, pattern, and visual appearance. Maintain everything as it was."
                        elif "barcode" in label_lower:
                            if reference_image:
                                prompt_text += "\n\nIMPORTANT: The black area contained a barcode. Fill it with the barcode shown in the REFERENCE image (third image). Copy the reference barcode EXACTLY as it appears - same barcode type, colors, background, pattern, numbers, and size. Do not modify or regenerate it, just replicate it precisely."
                            else:
                                prompt_text += "\n\nIMPORTANT: The black area contained a barcode. Fill it with a barcode that EXACTLY matches the original design - same barcode type, colors, background, size, pattern, and style. Maintain everything as it was."
                        elif "sign" in label_lower or "signboard" in label_lower:
                            prompt_text += f"\n\nIMPORTANT: The black area contained a signboard or store sign. Fill it with a signboard displaying '{replacement_value}' in the same style and format."
                        else:
                            prompt_text += f"\n\nIMPORTANT: The black area contained '{label_str}'. Fill it with '{replacement_value}' maintaining natural appearance."
                    else:
                        # No predefined value, use natural fill
                        if "license" in label_lower or "plate" in label_lower:
                            prompt_text += "\n\nIMPORTANT: The black area contained a license plate. Fill it with a realistic license plate with different random numbers and letters."
                        elif (
                            "phone" in label_lower
                            or "telephone" in label_lower
                            or "mobile" in label_lower
                        ):
                            prompt_text += "\n\nIMPORTANT: The black area contained a phone number. Fill it with a different random phone number in similar format."
                        elif "face" in label_lower:
                            prompt_text += (
                                "\n\nIMPORTANT: The black area contained a face. Replace it with a COMPLETELY DIFFERENT realistic human face of a different person. "
                                "Requirements:\n"
                                "- Maintain SIMILAR gender (if male, use male; if female, use female)\n"
                                "- Maintain SIMILAR ethnicity/race (if Asian, use Asian; if Caucasian, use Caucasian, etc.)\n"
                                "- Maintain SIMILAR age range (if young adult, use young adult; if elderly, use elderly, etc.)\n"
                                "- Match the EXACT eye gaze direction and where the person is looking\n"
                                "- Match the EXACT pose, head angle, and facial direction of the original\n"
                                "- Make the face size SLIGHTLY SMALLER and proportional to body size for natural harmony\n"
                                "- Match the LIGHTING conditions (direction, intensity, color temperature) from the surrounding environment\n"
                                "- Match the BACKGROUND blur/focus and depth of field\n"
                                "- Ensure the face integrates naturally with shadows, reflections, and environmental lighting\n"
                                "- The result must look photorealistic and indistinguishable from the original photo style"
                            )
                        else:
                            prompt_text += f"\n\nIMPORTANT: The black area contained {label_str}. Fill it with natural background that matches the context."

                print(f"[DEBUG] Prompt: {prompt_text}")

                # Prepare contents list
                contents = [
                    prompt_text,
                    image,  # First: Original image
                    masked_image,  # Second: Masked image with black areas
                ]

                # Add reference image if available (for QR/barcode)
                if reference_image:
                    contents.append(reference_image)  # Third: Reference image
                    print(f"[DEBUG] Including reference image in API call")

                response = self.client.models.generate_content(
                    model=self.imagen_model,
                    contents=contents,
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
                )

                if response.parts:
                    for part in response.parts:
                        if part.inline_data:
                            gen_img = Image.open(io.BytesIO(part.inline_data.data))

                            # Debug: Save generated image
                            debug_path = f"temp/debug_gen_{uuid.uuid4()}.png"
                            gen_img.save(debug_path)
                            print(f"[DEBUG] Saved generated image to {debug_path}")
                            print(
                                f"[DEBUG] Original size: {image.size}, Generated size: {gen_img.size}"
                            )

                            # Check if generated image size matches original
                            if gen_img.size != image.size:
                                print(
                                    f"[WARNING] Size mismatch! Resizing {gen_img.size} -> {image.size}"
                                )
                                gen_img = gen_img.resize(
                                    image.size, Image.Resampling.LANCZOS
                                )

                            # Return the full generated image (no cropping needed)
                            return gen_img

                print("[WARNING] No image part found in response.")
                return None

            except Exception as e:
                print(f"[WARNING] Gemini generation failed: {e}")
                return None

        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return None

    def generate_replacements_batch(
        self,
        image: Image.Image,
        regions: list[BoundingBox],
        labels: list[Optional[str]] = None,
    ) -> Optional[Image.Image]:
        """
        Generate replacements for multiple regions in a single API call.
        Creates one masked image with all regions blacked out.

        Args:
            image: Original image
            regions: List of bounding boxes to replace
            labels: List of labels for each region (same length as regions)

        Returns:
            Full modified image with all regions filled
        """
        if not regions:
            return image

        try:
            # Create a single masked image with ALL regions blacked out
            masked_image = image.copy()
            draw = ImageDraw.Draw(masked_image)

            region_infos = []
            for i, region in enumerate(regions):
                x1, y1, x2, y2 = region.to_xyxy()

                # Add padding
                mask_x1 = max(0, x1 - self.mask_padding)
                mask_y1 = max(0, y1 - self.mask_padding)
                mask_x2 = min(image.width, x2 + self.mask_padding)
                mask_y2 = min(image.height, y2 + self.mask_padding)

                # Draw black rectangle
                draw.rectangle([mask_x1, mask_y1, mask_x2, mask_y2], fill="black")

                label = labels[i] if labels and i < len(labels) else None
                region_infos.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "masked_bbox": (mask_x1, mask_y1, mask_x2, mask_y2),
                        "label": label,
                    }
                )

            print(f"[DEBUG] Batch processing {len(regions)} regions in single API call")
            for i, info in enumerate(region_infos):
                print(
                    f"[DEBUG]   Region {i + 1}: bbox={info['bbox']}, masked={info['masked_bbox']}, label={info['label']}"
                )

            # Debug: Save masked image
            mask_uuid = uuid.uuid4()
            masked_path = f"temp/debug_batch_masked_{mask_uuid}.png"
            masked_image.save(masked_path)
            print(f"[DEBUG] Saved batch masked image to {masked_path}")

            # Build prompt with all region information
            try:
                from google.genai import types

                # Check if any region needs QR/barcode reference images
                reference_images = []
                reference_image_info = []  # Track which regions use which reference

                for i, info in enumerate(region_infos):
                    label = info["label"]
                    if label:
                        label_str = (
                            label.value
                            if isinstance(label, PIIType)
                            else str(label).lower()
                        )
                        replacement_value = None

                        if isinstance(label, PIIType):
                            replacement_value = PII_REPLACEMENT_VALUES.get(label)
                        elif isinstance(label, str):
                            try:
                                pii_type = PIIType(label)
                                replacement_value = PII_REPLACEMENT_VALUES.get(pii_type)
                            except (ValueError, KeyError):
                                pass

                        # Check if it's a reference image path
                        if (
                            replacement_value
                            and isinstance(replacement_value, str)
                            and (
                                replacement_value.endswith(".png")
                                or replacement_value.endswith(".jpg")
                            )
                        ):
                            try:
                                ref_path = (
                                    replacement_value
                                    if os.path.isabs(replacement_value)
                                    else os.path.join(
                                        os.path.dirname(os.path.dirname(__file__)),
                                        replacement_value,
                                    )
                                )
                                if os.path.exists(ref_path):
                                    ref_img = Image.open(ref_path)
                                    reference_images.append(ref_img)
                                    reference_image_info.append(
                                        (i, len(reference_images))
                                    )
                                    print(
                                        f"[DEBUG] Loaded reference image from {ref_path} for region {i + 1}"
                                    )
                            except Exception as e:
                                print(
                                    f"[WARNING] Failed to load reference image for region {i + 1}: {e}"
                                )

                prompt_text = (
                    "You are an expert image editor. I am providing you with images:\n"
                    "1. The ORIGINAL image (first image)\n"
                    "2. A MASKED version where multiple areas are blacked out (second image)\n"
                )

                if reference_images:
                    for idx, _ in enumerate(reference_images, 1):
                        prompt_text += f"{idx + 2}. A REFERENCE image for QR code or barcode design (image {idx + 2})\n"

                prompt_text += (
                    "\nYour task: Use the original image as reference, and fill in ALL the black areas in the masked image naturally and seamlessly. "
                    "The filled areas should match the surrounding context, lighting, texture, and overall composition perfectly. "
                    "Do not modify any other parts of the image - only fill in the black regions. "
                    "Return the complete restored image with the same dimensions as the input.\n\n"
                )

                # Add information about each region
                prompt_text += f"There are {len(region_infos)} areas to fill:\n"
                for i, info in enumerate(region_infos, 1):
                    label = info["label"]
                    if label:
                        # Get replacement value
                        replacement_value = None
                        label_str = None

                        # Handle both PIIType enum and string labels
                        if isinstance(label, PIIType):
                            replacement_value = PII_REPLACEMENT_VALUES.get(label)
                            label_str = label.value
                        elif isinstance(label, str):
                            try:
                                pii_type = PIIType(label)
                                replacement_value = PII_REPLACEMENT_VALUES.get(pii_type)
                                label_str = label
                            except (ValueError, KeyError):
                                label_str = label
                        else:
                            label_str = str(label)

                        label_lower = label_str.lower()

                        if replacement_value:
                            if "license" in label_lower or "plate" in label_lower:
                                prompt_text += f"{i}. License plate - fill with '{replacement_value}'\n"
                            elif "phone" in label_lower:
                                prompt_text += f"{i}. Phone number - fill with '{replacement_value}'\n"
                            elif "email" in label_lower:
                                prompt_text += (
                                    f"{i}. Email - fill with '{replacement_value}'\n"
                                )
                            elif "address" in label_lower:
                                prompt_text += (
                                    f"{i}. Address - fill with '{replacement_value}'\n"
                                )
                            elif "name" in label_lower:
                                prompt_text += (
                                    f"{i}. Name - fill with '{replacement_value}'\n"
                                )
                            elif "qr" in label_lower or "qrcode" in label_lower:
                                # Check if this region has a reference image
                                ref_idx = next(
                                    (
                                        ref_num
                                        for reg_i, ref_num in reference_image_info
                                        if reg_i == i - 1
                                    ),
                                    None,
                                )
                                if ref_idx:
                                    prompt_text += f"{i}. QR code - copy REFERENCE image {ref_idx + 2} EXACTLY as shown (same design, colors, pattern, size). Do not modify it\n"
                                else:
                                    prompt_text += f"{i}. QR code - maintain exact original QR code as it was (design, colors, pattern, size)\n"
                            elif "barcode" in label_lower:
                                # Check if this region has a reference image
                                ref_idx = next(
                                    (
                                        ref_num
                                        for reg_i, ref_num in reference_image_info
                                        if reg_i == i - 1
                                    ),
                                    None,
                                )
                                if ref_idx:
                                    prompt_text += f"{i}. Barcode - copy REFERENCE image {ref_idx + 2} EXACTLY as shown (same barcode type, colors, pattern, numbers, size). Do not modify it\n"
                                else:
                                    prompt_text += f"{i}. Barcode - maintain exact original barcode as it was (type, colors, pattern, numbers, size)\n"
                            elif "sign" in label_lower or "signboard" in label_lower:
                                prompt_text += f"{i}. Signboard - fill with sign displaying '{replacement_value}'\n"
                            else:
                                prompt_text += f"{i}. {label_str} - fill with '{replacement_value}'\n"
                        else:
                            if "face" in label_lower:
                                prompt_text += (
                                    f"{i}. Face - replace with DIFFERENT person maintaining: "
                                    f"same gender, ethnicity, age range, eye gaze direction, pose, lighting. "
                                    f"Make face SLIGHTLY SMALLER and proportional to body. Must be photorealistic.\n"
                                )
                            else:
                                prompt_text += (
                                    f"{i}. {label_str} - fill with natural background\n"
                                )
                    else:
                        prompt_text += f"{i}. Unknown content - fill naturally\n"

                prompt_text += "\nFill all areas maintaining consistent style and natural appearance."

                print(f"[DEBUG] Batch prompt: {prompt_text}")

                # Prepare contents list
                contents = [
                    prompt_text,
                    image,  # Original image
                    masked_image,  # Masked image with all black areas
                ]

                # Add reference images if available
                if reference_images:
                    contents.extend(reference_images)
                    print(
                        f"[DEBUG] Including {len(reference_images)} reference image(s) in batch API call"
                    )

                response = self.client.models.generate_content(
                    model=self.imagen_model,
                    contents=contents,
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
                )

                if response.parts:
                    for part in response.parts:
                        if part.inline_data:
                            gen_img = Image.open(io.BytesIO(part.inline_data.data))

                            # Debug: Save generated image
                            debug_path = f"temp/debug_batch_gen_{uuid.uuid4()}.png"
                            gen_img.save(debug_path)
                            print(
                                f"[DEBUG] Saved batch generated image to {debug_path}"
                            )
                            print(
                                f"[DEBUG] Original size: {image.size}, Generated size: {gen_img.size}"
                            )

                            # Resize if needed
                            if gen_img.size != image.size:
                                print(
                                    f"[WARNING] Size mismatch! Resizing {gen_img.size} -> {image.size}"
                                )
                                gen_img = gen_img.resize(
                                    image.size, Image.Resampling.LANCZOS
                                )

                            return gen_img

                print("[WARNING] No image part found in batch response.")
                return None

            except Exception as e:
                print(f"[WARNING] Batch generation failed: {e}")
                return None

        except Exception as e:
            print(f"[ERROR] Batch generation failed: {e}")
            return None
