# Builder stage
FROM python:3.11-slim AS builder

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

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY --from=builder /usr/local /usr/local
COPY --chown=appuser:appuser . /app

RUN mkdir -p /app/uploads /app/outputs /app/temp /app/public && \
    chown -R appuser:appuser /app/uploads /app/outputs /app/temp /app/public && \
    chmod -R 775 /app/uploads /app/outputs /app/temp /app/public

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
