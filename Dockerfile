# Multi-stage build for Luxia Worker Service
FROM python:3.13-slim as builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies to a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first (cached layer)
RUN pip install --upgrade pip

# Install dependencies (Docker caches this layer unless requirements change)
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Clean up in separate layer (only runs if above changes)
RUN rm -rf /root/.cache/pip /tmp/* && \
    find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Final stage
FROM python:3.13-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment to use the venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:$PYTHONPATH \
    REDIS_URL=redis://redis:6379 \
    LOG_DB_PATH=/app/logs.db \
    TRANSFORMERS_CACHE=/app/model_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/model_cache

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application code
COPY . .

# Create volume for persistent SQLite logs
VOLUME ["/app/logs.db"]

# Expose FastAPI port
EXPOSE 9000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:9000/admin/logs?limit=1 || exit 1

# Start FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
