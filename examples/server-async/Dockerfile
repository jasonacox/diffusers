# Simple container for the async Diffusers server
# Default base is a slim Python image (CPU). For GPU, run with --gpus all and consider a CUDA-enabled base.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    MODEL_PATH="black-forest-labs/FLUX.1-schnell" \
    SERVICE_URL="" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

WORKDIR /app

# System deps (git for pip VCS, curl for health/debug, minimal build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy only the server directory
COPY . /app

# Install Python deps
# By default, install CUDA 12.1-enabled wheels for torch/vision/audio for NVIDIA GPUs.
# If you need CPU-only, you can override TORCH_INDEX to the CPU index or install specific CPU wheels.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121
RUN python -m pip install --upgrade pip && \
    python -m pip install --extra-index-url ${TORCH_INDEX} torch torchvision torchaudio && \
    # Try xformers (optional). If it fails, proceed without it.
    (python -m pip install xformers || true) && \
    # Install remaining requirements (drop torch/torchvision lines to avoid conflicts)
    (grep -v -E '^(torch|torchvision|torchaudio)\b' requirements.txt > /tmp/reqs.txt) && \
    (python -m pip install -r /tmp/reqs.txt || true) && \
    python -m pip install diffusers

EXPOSE 8500

# Health check for /api/status
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:8500/api/status || exit 1

# Start the server
CMD ["uvicorn", "serverasync:app", "--host", "0.0.0.0", "--port", "8500"]
