# ── Lean RunPod Serverless Dockerfile for Wan2.1 ──────────────────────────
# Uses RunPod's PyTorch base (already has torch + CUDA + python + build tools)
# Model downloads lazily on first request → small image, fast build
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models
ENV WAN_MODEL_SIZE=1.3B

WORKDIR /app

# Install only light deps (no basicsr/realesrgan — too heavy, not needed for MVP)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt runpod

COPY handler_runpod.py .

CMD ["python", "handler_runpod.py"]
