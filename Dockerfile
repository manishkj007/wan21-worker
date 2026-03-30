# ── Lean RunPod Serverless Dockerfile for Wan2.1 T2V + I2V ────────────────
# Supports both text-to-video (1.3B/14B) and image-to-video (14B-480P)
# Models download lazily on first request via HuggingFace cache
#
# Deploy modes (set WAN_MODE env var):
#   WAN_MODE=t2v   → T2V only    (AMPERE_24 GPU OK for 1.3B, ~$0.31/hr)
#   WAN_MODE=i2v   → I2V only    (needs AMPERE_48, ~$0.47/hr)
#   WAN_MODE=both  → both models (needs AMPERE_48, ~$0.47/hr)
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
