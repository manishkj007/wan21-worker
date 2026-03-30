# ── RunPod Serverless Dockerfile for Wan2.1 T2V + I2V ─────────────────────
# SLIM image — models stored on RunPod Network Volume (/runpod-volume).
# Image is ~2-3GB (code + deps only). Cold start ~2 min instead of ~25 min.
#
# Models on volume:
#   /runpod-volume/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers  (~2.6GB)
#   /runpod-volume/models/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers (~28GB)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/models \
    WAN_MODEL_SIZE=1.3B \
    WAN_MODE=both

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r requirements.txt runpod

COPY handler_runpod.py .
CMD ["python3", "handler_runpod.py"]
