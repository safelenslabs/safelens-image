# Installation Guide

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional, but recommended for better performance)
- 4GB+ RAM
- 2GB+ disk space for models

## Step 1: Clone or Download

```bash
cd savelens_image
```

## Step 2: Create Virtual Environment

### Using venv (standard)

```bash
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### Using conda

```bash
conda create -n savelens python=3.11
conda activate savelens
```

## Step 3: Install Dependencies

### Option A: Using pip (recommended)

```bash
pip install -e .
```

### Option B: Using requirements.txt

```bash
pip install -r requirements.txt
```

### Option C: Using uv (faster)

```bash
# Install uv first
pip install uv

# Install dependencies
uv pip install -e .
```

## Step 4: Download Additional Models

### EasyOCR Models

EasyOCR will automatically download models on first use. To pre-download:

```python
import easyocr
reader = easyocr.Reader(['en', 'ko'])  # Downloads English and Korean models
```

### spaCy Models (for enhanced NER)

```bash
python -m spacy download en_core_web_sm
```

## Step 5: Verify Installation

```bash
python -c "from src.pipeline import PrivacyPipeline; print('Installation successful!')"
```

## Step 6: Run Example

```bash
python example.py
```

This will create sample images and demonstrate the full pipeline.

## Step 7: Start API Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for interactive API documentation.

## Troubleshooting

### Issue: CUDA not available

If you don't have a CUDA-capable GPU:

```python
# Edit src/face_detector.py and src/text_detector.py
# Change device='cuda' to device='cpu'
```

Or set environment variable:

```bash
export CUDA_VISIBLE_DEVICES=""
```

### Issue: EasyOCR download fails

Manually download models from [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR) and place in `~/.EasyOCR/model/`

### Issue: Memory error

Reduce batch size or use CPU-only mode:

```python
pipeline = PrivacyPipeline(use_enhanced_ner=False)
```

### Issue: Import errors

Make sure you're in the project root and virtual environment is activated:

```bash
cd savelens_image
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```

## Performance Tips

1. **Use GPU**: Install CUDA and PyTorch with CUDA support for 10x speed improvement
2. **Disable enhanced NER**: Set `use_enhanced_ner=False` for faster startup
3. **Reduce image size**: Resize large images before processing
4. **Batch processing**: Process multiple images in sequence without clearing cache

## Docker Setup (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t savelens .
docker run -p 8000:8000 savelens
```

## Next Steps

- Read [README.md](README.md) for API usage
- Check [example.py](example.py) for code examples
- Run tests with `pytest tests/`
- Explore interactive docs at http://localhost:8000/docs
