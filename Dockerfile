# ── RunPod Serverless Dockerfile for Wan2.1 T2V + I2V ─────────────────────
# Models are lazy-downloaded on first request to /app/models (container disk).
# Set containerDiskInGb >= 100 for I2V 14B support.
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

COPY handler_runpod.py .
CMD ["python3", "handler_runpod.py"]
