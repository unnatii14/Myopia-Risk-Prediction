# ─── Backend Dockerfile ──────────────────────────────────
FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn python-dotenv

# Copy application
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001/health')"

# Run with gunicorn in production
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--timeout", "60", "--access-logfile", "-", "--error-logfile", "-", "api:app"]
