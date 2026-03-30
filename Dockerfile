# ── RunPod Serverless Dockerfile for Wan2.1 T2V + I2V ─────────────────────
# Pre-downloads T2V 1.3B (~2.6GB) at build time for instant cold starts.
# I2V 14B (~28GB) is lazy-downloaded on first I2V request to container disk.
# Requires containerDiskInGb >= 60 for I2V support.
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/models \
    WAN_MODEL_SIZE=1.3B \
    WAN_MODE=both

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r requirements.txt runpod

# ── Pre-download T2V 1.3B model (~2.6GB) ──────────────────────────────────
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
print('[build] Downloading T2V 1.3B...'); \
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B-Diffusers'); \
print('[build] T2V 1.3B cached!')"

# NOTE: I2V 14B (~28GB) is NOT pre-baked — too large for GitHub builder.
# It downloads lazily on first I2V request using the 100GB container disk.

COPY handler_runpod.py .
CMD ["python3", "handler_runpod.py"]
