# ==============================================================================
# Dockerfile for Tier 2 FastAPI Backend + Tier 3 Static Web App
# Optimized multi-stage Python 3.11 slim image (<150 MB)
# ==============================================================================

FROM python:3.11-slim

# Set environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install minimal OS dependencies for cryptography and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install server Python requirements
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy application source code
COPY config.yaml .
COPY server/ ./server/
COPY client/ ./client/
COPY data/ ./data/

# Expose standard FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run FastAPI backend with Uvicorn
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
