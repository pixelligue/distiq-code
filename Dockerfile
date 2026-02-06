# Multi-stage build for distiq-code

FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/distiq-code /usr/local/bin/distiq-code
COPY --from=builder /build/src /app/src

# Create config directory
RUN mkdir -p /root/.distiq-code

# Expose proxy port
EXPOSE 11434

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import requests; requests.get('http://localhost:11434/health')"]

# Run proxy server
CMD ["distiq-code", "start"]
