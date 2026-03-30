"""RunPod Serverless handler — Wan2.1 Text-to-Video + Image-to-Video.
Lazy model load on first request. Supports both T2V and I2V via 'action' field.

Actions:
  - "t2v" (default): text-to-video using WanPipeline (1.3B or 14B)
  - "i2v": image-to-video using WanImageToVideoPipeline (14B-480P)
    Requires: image_base64 (PNG/JPG as base64)

Deploy with WAN_MODE env var:
  - "t2v"  → only loads T2V pipeline (small GPU OK, 1.3B ≈ 8GB VRAM)
  - "i2v"  → only loads I2V pipeline (needs 48GB GPU, 14B ≈ 28GB VRAM)
  - "both" → loads whichever is requested first (needs 48GB GPU)
"""

import os, uuid, time, base64, io

# Model cache: use the pre-baked /app/models from Docker build.
# For temp files (mp4 encoding), use /dev/shm (RAM) if available.
MODEL_DIR = "/app/models"
if os.path.isdir("/dev/shm"):
    TMPDIR = "/dev/shm"
else:
    TMPDIR = "/tmp"

# Set HF cache so from_pretrained() finds the pre-downloaded model
os.environ["HF_HOME"] = MODEL_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_DIR
os.environ["TMPDIR"] = TMPDIR

import runpod

WAN_MODE = os.environ.get("WAN_MODE", "both")  # "t2v", "i2v", or "both"

t2v_pipe = None
i2v_pipe = None
device = "cuda"
_torch = None

def ensure_torch():
    global _torch, device
    if _torch is not None:
        return _torch
    import torch
    _torch = torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    return torch

def load_t2v():
    global t2v_pipe
    if t2v_pipe is not None:
        return t2v_pipe
    torch = ensure_torch()
    from diffusers import WanPipeline
    mid = f"Wan-AI/Wan2.1-T2V-{os.environ.get('WAN_MODEL_SIZE','1.3B')}-Diffusers"
    print(f"[wan-t2v] loading {mid}")
    t2v_pipe = WanPipeline.from_pretrained(mid, torch_dtype=torch.float16).to(device)
    try: t2v_pipe.enable_vae_slicing()
    except: pass
    print(f"[wan-t2v] ready on {device}")
    return t2v_pipe

def load_i2v():
    global i2v_pipe
    if i2v_pipe is not None:
        return i2v_pipe
    torch = ensure_torch()
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionModel
    mid = os.environ.get("WAN_I2V_MODEL", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    print(f"[wan-i2v] loading {mid}")
    image_encoder = CLIPVisionModel.from_pretrained(
        mid, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        mid, subfolder="vae", torch_dtype=torch.float32
    )
    i2v_pipe = WanImageToVideoPipeline.from_pretrained(
        mid, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
    )
    # Use CPU offload instead of .to(device) — 14B model needs ~50GB,
    # this keeps components on CPU and moves to GPU only during forward pass.
    i2v_pipe.enable_model_cpu_offload()
    try: i2v_pipe.enable_vae_slicing()
    except: pass
    print(f"[wan-i2v] ready (cpu_offload, {device})")
    return i2v_pipe

def frames_to_mp4(frames, fps=16):
    """Encode list of PIL/ndarray frames to base64 MP4."""
    import imageio.v3 as iio, numpy as np
    mp4 = f"{TMPDIR}/{uuid.uuid4()}.mp4"
    iio.imwrite(mp4, [np.array(f) for f in frames], fps=fps, codec="libx264")
    with open(mp4, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    sz = os.path.getsize(mp4)
    try: os.remove(mp4)
    except: pass
    return b64, sz

def handle_t2v(inp):
    """Text-to-video generation."""
    prompt = inp.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}
    pipe = load_t2v()
    nf = min(max(int(inp.get("num_frames", 81)), 17), 121)
    w = int(inp.get("width", 832))
    h = int(inp.get("height", 480))
    gs = float(inp.get("guidance_scale", 5.0))
    steps = min(max(int(inp.get("num_inference_steps", 25)), 10), 50)
    t0 = time.time()
    print(f"[wan-t2v] gen {nf}f {w}x{h} {steps}s")
    out = pipe(prompt=prompt, num_frames=nf, width=w, height=h,
               guidance_scale=gs, num_inference_steps=steps)
    frames = out.frames[0]
    b64, sz = frames_to_mp4(frames)
    dt = time.time() - t0
    print(f"[wan-t2v] done {dt:.1f}s {sz}B")
    return {"video_base64": b64, "duration_seconds": round(nf / 16, 2),
            "inference_time": round(dt, 1), "file_size_bytes": sz}

def handle_i2v(inp):
    """Image-to-video generation."""
    import numpy as np
    from PIL import Image

    prompt = inp.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required for i2v"}
    image_b64 = inp.get("image_base64", "")
    if not image_b64:
        return {"error": "image_base64 is required for i2v"}

    # Decode input image
    img_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    pipe = load_i2v()

    nf = min(max(int(inp.get("num_frames", 81)), 17), 121)
    gs = float(inp.get("guidance_scale", 5.0))
    steps = min(max(int(inp.get("num_inference_steps", 25)), 10), 50)

    # Auto-resize image to match VAE constraints (480P target)
    max_area = int(inp.get("max_area", 480 * 832))
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    h = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    w = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((w, h))

    t0 = time.time()
    print(f"[wan-i2v] gen {nf}f {w}x{h} {steps}s")
    out = pipe(image=image, prompt=prompt, height=h, width=w,
               num_frames=nf, guidance_scale=gs, num_inference_steps=steps)
    frames = out.frames[0]
    b64, sz = frames_to_mp4(frames)
    dt = time.time() - t0
    print(f"[wan-i2v] done {dt:.1f}s {sz}B")
    return {"video_base64": b64, "duration_seconds": round(nf / 16, 2),
            "inference_time": round(dt, 1), "file_size_bytes": sz,
            "resolution": f"{w}x{h}"}

def handler(event):
    try:
        inp = event.get("input", {})
        action = inp.get("action", "t2v")

        if action == "i2v":
            if WAN_MODE == "t2v":
                return {"error": "This endpoint is T2V-only. Deploy with WAN_MODE=i2v or WAN_MODE=both"}
            return handle_i2v(inp)
        else:
            if WAN_MODE == "i2v":
                return {"error": "This endpoint is I2V-only. Deploy with WAN_MODE=t2v or WAN_MODE=both"}
            return handle_t2v(inp)
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
