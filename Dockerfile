# Multi-stage build for Luxia Worker Service
FROM python:3.13-slim as builder

WORKDIR /app

# Install system dependencies for building (including llama-cpp-python requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies to a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first (cached layer)
RUN pip install --upgrade pip

# Install dependencies (Docker caches this layer unless requirements change)
COPY requirements-docker.txt .

# Set environment variables for llama-cpp-python CPU build
ENV CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_CUDA=OFF"

RUN pip install -r requirements-docker.txt

# Clean up in separate layer (only runs if above changes)
RUN rm -rf /root/.cache/pip /tmp/* && \
    find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Model download stage
FROM python:3.13-slim as model-downloader

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Download Qwen2-0.5B-Instruct GGUF model (~350MB, good for RAG tasks on CPU)
RUN mkdir -p /models && \
    curl -L -o /models/qwen2-0_5b-instruct-q4_k_m.gguf \
    "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_k_m.gguf"

# Final stage
FROM python:3.13-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy model from model-downloader
COPY --from=model-downloader /models /app/models

# Set environment to use the venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:$PYTHONPATH \
    REDIS_URL=redis://redis:6379 \
    LOG_DB_PATH=/app/logs.db \
    TRANSFORMERS_CACHE=/app/model_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/model_cache \
    LOCAL_LLM_MODEL_PATH=/app/models/qwen2-0_5b-instruct-q4_k_m.gguf \
    LOCAL_LLM_CONTEXT_SIZE=2048 \
    LOCAL_LLM_THREADS=4 \
    LOCAL_LLM_MAX_TOKENS=512

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
