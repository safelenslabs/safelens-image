# SafeLens Image Server

> A Image server for automatic PII and face anonymization in images using Google Gemini Vision API.

## Overview

SafeLens Image is a FastAPI-based web service that automatically detects and anonymizes personally identifiable information (PII) and faces in images using Google Gemini Vision API.

## Key Features

- **Automatic Detection**: Detect text PII and faces in images with Gemini Vision API
- **Selective Anonymization**: Choose which detected items to anonymize
- **Multiple Anonymization Methods**: Blur, mosaic, pixelate, or replace with AI-generated images
- **REST API**: Easy-to-use web interface

## Quick Start

### 1. Environment Setup

```bash
# Requires Python 3.11 or higher
$ uv sync
```

### 2. API Key Configuration

Create a `.env` file:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
S3_BUCKET_NAME=your-bucket-name
S3_REGION_NAME=us-east-1
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
```

### Storage

SafeLens Image uses Amazon S3 for image storage:
- All images are uploaded to the specified S3 bucket
- Images are organized with prefixes:
  - `uploads/` - Original uploaded images
  - `outputs/` - Anonymized images
  - `temp/` - Temporary debug images
- Requires AWS credentials with S3 read/write permissions

### 3. Run Server

```bash
$ uv run main.py
```

## Dependencies

- FastAPI - HTTP server
- Google Gemini Vision API - PII/Face detection
- OpenCV & Pillow - Image processing
- Boto3 - AWS S3 integration
