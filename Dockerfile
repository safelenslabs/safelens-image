# Builder stage
FROM python:3.11-slim AS builder

# Build-time deps (for numpy/opencv/pillow C-extension fallback)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Final runtime stage
FROM python:3.11-slim

# Runtime deps (OpenCV headless + Pillow)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY --from=builder /usr/local /usr/local
COPY --chown=appuser:appuser . .

RUN mkdir -p uploads outputs temp public && \
    chown -R appuser:appuser uploads outputs temp public

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
