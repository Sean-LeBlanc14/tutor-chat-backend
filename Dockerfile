# Production Dockerfile - Updated for vLLM + GPU optimization
FROM nvidia/cuda:12.2-devel-ubuntu22.04 as builder

# Set build arguments
ARG ENVIRONMENT=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (including vLLM)
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download models to cache them
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Production stage
FROM nvidia/cuda:12.2-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symlinks for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create cache directories with proper permissions
RUN mkdir -p /home/appuser/.cache /tmp/huggingface_cache /app/logs /app/data && \
    chown -R appuser:appuser /home/appuser/.cache /app && \
    chmod -R 755 /home/appuser/.cache && \
    chmod -R 777 /tmp/huggingface_cache

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Make sure FAISS files are in the right place (you'll add these after GitHub commit)
# These files should be copied in: chunk_index.faiss, chunk_metadata.json, chunks.jsonl

# Make sure scripts are executable
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Set environment variables for GPU and vLLM
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV PORT=8080
ENV HF_HOME=/tmp/huggingface_cache
ENV CUDA_VISIBLE_DEVICES=0
ENV VLLM_USE_MODELSCOPE=False

# Switch to non-root user
USER appuser

# Health check with longer timeout for model loading
HEALTHCHECK --interval=60s --timeout=60s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Expose port
EXPOSE 8080

# Use uvicorn directly (single worker for GPU)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
