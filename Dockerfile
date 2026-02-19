# Multi-stage build for optimized image size

# Stage 1: Builder
FROM python:3.13-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.13-slim as runner

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable:
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p instance/temp temp vector_store

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV WORKERS=4
ENV THREADS=2
ENV TIMEOUT=120

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run with gunicorn for production
CMD gunicorn --bind 0.0.0.0:5000 \
    --workers ${WORKERS} \
    --threads ${THREADS} \
    --timeout ${TIMEOUT} \
    --worker-class sync \
    --access-logfile - \
    --error-logfile - \
    "app:create_app()"
