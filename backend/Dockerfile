# ─── Backend Dockerfile ──────────────────────────────────
# Python 3.11 — required for scikit-learn 1.7.2 + xgboost 3.2.0 + tensorflow 2.17
FROM python:3.11-slim

WORKDIR /app

# Install system deps needed for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies — pinned to match the versions models were trained with
COPY backend/requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application code and models
# WORKDIR is /app, so models go to /models (one level up = MODELS_DIR=../models)
COPY backend/ .
COPY models/ /models/

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001/health')"

# Use $PORT env var (Render sets this automatically)
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --access-logfile - --error-logfile - api:app
