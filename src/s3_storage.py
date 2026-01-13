"""
S3 storage utility for image upload/download.
"""

import io
from typing import Optional
from PIL import Image
import boto3
from botocore.exceptions import ClientError
from .settings import Settings


class S3Storage:
    """S3 storage handler for images."""

    def __init__(self, settings: Settings):
        """
        Initialize S3 storage client.

        Args:
            settings: Settings instance with S3 configuration
        """
        self.bucket_name = settings.s3_bucket_name
        region = settings.s3_region_name
        endpoint_url = settings.s3_endpoint_url
        access_key = settings.aws_access_key_id
        secret_key = settings.aws_secret_access_key

        # Initialize S3 client
        session_kwargs = {"region_name": region}
        if endpoint_url:
            session_kwargs["endpoint_url"] = endpoint_url
        if access_key and secret_key:
            session_kwargs["aws_access_key_id"] = access_key
            session_kwargs["aws_secret_access_key"] = secret_key

        self.s3_client = boto3.client("s3", **session_kwargs)
        print(f"[S3] Initialized S3 client for bucket: {self.bucket_name}")

    def upload_image(
        self,
        image: Image.Image,
        key: str,
        format: str = "PNG",
        quality: int = 95,
    ) -> str:
        """
        Upload PIL Image to S3.

        Args:
            image: PIL Image object
            key: S3 object key (path in bucket)
            format: Image format (PNG, JPEG, WEBP)
            quality: Image quality (1-100)

        Returns:
            S3 object URL
        """
        try:
            # Convert image to bytes
            img_byte_arr = io.BytesIO()

            # Convert RGBA to RGB for JPEG
            if format.upper() in ["JPG", "JPEG"] and image.mode == "RGBA":
                image = image.convert("RGB")

            image.save(img_byte_arr, format=format.upper(), quality=quality)
            img_byte_arr.seek(0)

            # Determine content type
            content_type_map = {
                "PNG": "image/png",
                "JPEG": "image/jpeg",
                "JPG": "image/jpeg",
                "WEBP": "image/webp",
            }
            content_type = content_type_map.get(format.upper(), "image/png")

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=img_byte_arr.getvalue(),
                ContentType=content_type,
            )

            print(f"[S3] Uploaded image to s3://{self.bucket_name}/{key}")
            return f"s3://{self.bucket_name}/{key}"

        except ClientError as e:
            print(f"[S3] Error uploading image: {e}")
            raise

    def download_image(self, key: str) -> Optional[Image.Image]:
        """
        Download image from S3.

        Args:
            key: S3 object key

        Returns:
            PIL Image object or None if not found
        """
        try:
            # Download from S3
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            img_data = response["Body"].read()

            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            print(f"[S3] Downloaded image from s3://{self.bucket_name}/{key}")
            return image

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                print(f"[S3] Image not found: {key}")
                return None
            print(f"[S3] Error downloading image: {e}")
            raise

    def delete_image(self, key: str) -> bool:
        """
        Delete image from S3.

        Args:
            key: S3 object key

        Returns:
            True if deleted, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            print(f"[S3] Deleted image: s3://{self.bucket_name}/{key}")
            return True

        except ClientError as e:
            print(f"[S3] Error deleting image: {e}")
            return False

    def get_presigned_url(self, key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for temporary access to an S3 object.

        Args:
            key: S3 object key
            expiration: URL expiration time in seconds (default: 1 hour)

        Returns:
            Presigned URL or None if error
        """
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": key},
                ExpiresIn=expiration,
            )
            return url

        except ClientError as e:
            print(f"[S3] Error generating presigned URL: {e}")
            return None

    def list_images(self, prefix: str = "") -> list[str]:
        """
        List all images in S3 bucket with given prefix.

        Args:
            prefix: S3 key prefix (folder path)

        Returns:
            List of S3 object keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=prefix
            )

            if "Contents" not in response:
                return []

            keys = [obj["Key"] for obj in response["Contents"]]
            return keys

        except ClientError as e:
            print(f"[S3] Error listing images: {e}")
            return []
