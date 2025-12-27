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
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run Server

```bash
$ uv run main.py
```

## Dependencies

- FastAPI - HTTP server
- Google Gemini Vision API - PII/Face detection
- OpenCV & Pillow - Image processing
