FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev gcc g++ \
    ffmpeg git wget libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install PyTorch from official pre-built CUDA 12.1 wheels (no compilation)
RUN pip3 install --no-cache-dir \
    "torch>=2.1.0" "torchvision>=0.16.0" \
    --index-url https://download.pytorch.org/whl/cu121

# 2) Pin numpy<2 + setuptools for basicsr compatibility
RUN pip3 install --no-cache-dir "setuptools<70" "numpy<2" cython

# 3) Install basicsr/realesrgan WITHOUT building CUDA extensions
#    (the runtime image has no nvcc; inference still uses CUDA via PyTorch wheels)
COPY requirements.txt .
ENV BASICSR_EXT=False
RUN pip3 install --no-cache-dir -r requirements.txt runpod opencv-python-headless

# ── Wav2Lip for lip-sync ─────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git /app/Wav2Lip && \
    pip3 install --no-cache-dir librosa==0.9.2 face-detection numba==0.59.1

COPY . .

# ── Model config ──────────────────────────────────────────────────────────
# Override with 14B for higher quality (needs ≥24 GB VRAM)
ENV WAN_MODEL_SIZE=1.3B
ENV HF_HOME=/app/models

# Pre-download the model weights at build time so cold starts are fast.
# This makes the image ~5-8 GB but avoids downloading on every pod start.
RUN python3 -c "from diffusers import WanPipeline; WanPipeline.from_pretrained('Wan-AI/Wan2.1-T2V-1.3B')"

# Pre-download Real-ESRGAN anime upscaler weights (~17 MB)
RUN mkdir -p /app/weights && \
    wget -q -O /app/weights/RealESRGAN_x4plus_anime_6B.pth \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"

# Pre-download Wav2Lip GAN weights (~400 MB)
RUN wget -q -O /app/weights/wav2lip_gan.pth \
    "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth" || \
    echo "[Wav2Lip] Auto-download may fail — will retry at runtime"

# ── Entry point ───────────────────────────────────────────────────────────
# RunPod Serverless: uses handler_runpod.py (default)
# Self-hosted / Vast.ai: override CMD to "python3 server.py --host 0.0.0.0"
CMD ["python3", "handler_runpod.py"]
