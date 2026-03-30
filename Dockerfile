# ── RunPod Serverless Dockerfile for Wan2.1 T2V + I2V ─────────────────────
# Pre-downloads the 1.3B T2V model at build time so workers start instantly
# with no runtime downloads (avoids "no space left on device" on worker disks).
#
# The 14B I2V model is too large to bake in (~28GB). I2V requests will attempt
# a lazy download; if the worker has sufficient disk/RAM, it works.
# Otherwise the pipeline falls back to T2V automatically.
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

# Pre-download the 1.3B T2V model during build (~2.6GB, cached in image layer)
RUN python3 -c "\
from diffusers import WanPipeline; \
import torch; \
WanPipeline.from_pretrained('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', torch_dtype=torch.float16); \
print('[build] T2V 1.3B model cached successfully')"

COPY handler_runpod.py .
CMD ["python3", "handler_runpod.py"]
