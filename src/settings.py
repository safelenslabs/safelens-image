"""
Settings module for environment configuration using Pydantic.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Gemini API
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")

    # S3 Storage
    s3_bucket_name: str = Field(..., alias="S3_BUCKET_NAME")
    s3_region_name: str = Field(default="us-east-1", alias="S3_REGION_NAME")
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(
        default=None, alias="AWS_SECRET_ACCESS_KEY"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get or create settings instance.

    Returns:
        Settings instance loaded from environment variables
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
