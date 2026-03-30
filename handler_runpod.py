"""RunPod Serverless handler — Wan2.1 Text-to-Video.
Lazy model load on first request. Tiny handler = fast worker start."""

import os, uuid, time, base64

# Force all caches and temp files to /dev/shm (RAM-backed tmpfs).
# Network volume (/runpod-volume) exists but may have quota issues;
# /dev/shm is proven to work reliably for the 1.3B model.
# Endpoint is pinned to EU-CZ-1 (same DC as the network volume).
if os.path.isdir("/dev/shm"):
    os.environ["HF_HOME"] = "/dev/shm/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/dev/shm/hf"
    os.environ["XDG_CACHE_HOME"] = "/dev/shm/cache"
    os.environ["TMPDIR"] = "/dev/shm"
    TMPDIR = "/dev/shm"
else:
    TMPDIR = "/tmp"

import runpod

pipe = None
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

def load_model():
    global pipe
    if pipe is not None:
        return
    torch = ensure_torch()
    from diffusers import WanPipeline
    mid = f"Wan-AI/Wan2.1-T2V-{os.environ.get('WAN_MODEL_SIZE','1.3B')}-Diffusers"
    print(f"[wan] loading {mid}")
    pipe = WanPipeline.from_pretrained(mid, torch_dtype=torch.float16).to(device)
    try: pipe.enable_vae_slicing()
    except: pass
    print(f"[wan] ready on {device}")

def handler(event):
    try:
        inp = event.get("input", {})
        prompt = inp.get("prompt", "")
        if not prompt:
            return {"error": "prompt is required"}
        load_model()
        nf = min(max(int(inp.get("num_frames", 81)), 17), 121)
        w = int(inp.get("width", 832))
        h = int(inp.get("height", 480))
        gs = float(inp.get("guidance_scale", 5.0))
        steps = min(max(int(inp.get("num_inference_steps", 25)), 10), 50)
        t0 = time.time()
        print(f"[wan] gen {nf}f {w}x{h} {steps}s")
        out = pipe(prompt=prompt, num_frames=nf, width=w, height=h,
                   guidance_scale=gs, num_inference_steps=steps)
        frames = out.frames[0]
        mp4 = f"{TMPDIR}/{uuid.uuid4()}.mp4"
        import imageio.v3 as iio, numpy as np
        iio.imwrite(mp4, [np.array(f) for f in frames], fps=16, codec="libx264")
        dt = time.time() - t0
        with open(mp4, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        sz = os.path.getsize(mp4)
        try: os.remove(mp4)
        except: pass
        print(f"[wan] done {dt:.1f}s {sz}B")
        return {"video_base64": b64, "duration_seconds": round(nf/16, 2),
                "inference_time": round(dt, 1), "file_size_bytes": sz}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
