# ── RunPod Serverless Dockerfile for Wan2.1 T2V + I2V ─────────────────────
# SLIM image — models & weights stored on RunPod Network Volume (/runpod-volume).
# Image is ~1.5GB (code + deps only). Cold start ~1 min.
#
# On network volume:
#   /runpod-volume/models/   — Wan2.1 T2V/I2V model checkpoints
#   /runpod-volume/weights/  — Real-ESRGAN + Wav2Lip weights (~420MB total)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/models \
    WAN_MODEL_SIZE=1.3B \
    WAN_MODE=both

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip ffmpeg git wget && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r requirements.txt runpod

# ── Real-ESRGAN + Wav2Lip pip deps (code only, weights on volume) ────
RUN pip3 install --no-cache-dir realesrgan basicsr opencv-python-headless && \
    pip3 install --no-cache-dir librosa face-alignment && \
    git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git /app/Wav2Lip || true

# NO weight downloads — they live on /runpod-volume/weights/
# Upload once via: handler action "download_weights"

COPY handler_runpod.py .
CMD ["python3", "handler_runpod.py"]
