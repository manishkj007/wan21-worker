# ── RunPod Serverless Dockerfile for Wan2.1 T2V + I2V ─────────────────────
# Pre-downloads BOTH models at build time:
#   - T2V 1.3B (~2.6GB)  — text-to-video
#   - I2V 14B  (~28GB)   — image-to-video (the good stuff)
#
# Total image size: ~35-40GB. Build takes ~20-30 min on first run.
# Workers start instantly with no runtime downloads.
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

# ── Pre-download I2V 14B model (~28GB) ────────────────────────────────────
# This is the large download — cached as a Docker layer so it only downloads once.
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
print('[build] Downloading I2V 14B-480P...'); \
snapshot_download('Wan-AI/Wan2.1-I2V-14B-480P-Diffusers'); \
print('[build] I2V 14B cached!')"

COPY handler_runpod.py .
CMD ["python3", "handler_runpod.py"]
