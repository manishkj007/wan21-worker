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
    apt-get install -y --no-install-recommends python3 python3-pip ffmpeg git wget && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r requirements.txt runpod

# ── Real-ESRGAN for 4x upscaling ────────────────────────────────────
RUN pip3 install --no-cache-dir realesrgan basicsr opencv-python-headless

# ── Wav2Lip for lip-sync (optional, may not work on cartoon faces) ──
RUN pip3 install --no-cache-dir librosa face-alignment && \
    git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git /app/Wav2Lip || true

# Pre-download Real-ESRGAN anime weights (~17 MB)
RUN mkdir -p /app/weights && \
    wget -q -O /app/weights/RealESRGAN_x4plus_anime_6B.pth \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" || \
    echo "[Upscale] Weight download failed — will retry at runtime"

# Pre-download Wav2Lip GAN weights (~400 MB)
RUN wget -q -O /app/weights/wav2lip_gan.pth \
    "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEbfuuzJLiQ_QRxSxRfkBw_9Mf_huhA3IRVFhfZSjpg?download=1" || \
    wget -q -O /app/weights/wav2lip_gan.pth \
    "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth" || \
    echo "[Wav2Lip] Weight download failed — will retry at runtime"

COPY handler_runpod.py .
CMD ["python3", "handler_runpod.py"]
