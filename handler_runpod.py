"""
RunPod Serverless handler for Wan2.1 Text-to-Video.
Lean version — just text-to-video, no upscaling/lip-sync overhead.
Model downloads lazily on first request.
"""

import os
import uuid
import time
import base64

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
    print(f"[Init] torch loaded, device={device}, CUDA={torch.cuda.is_available()}")
    return torch


def load_model():
    global pipe
    if pipe is not None:
        return

    torch = ensure_torch()
    from diffusers import WanPipeline

    model_size = os.environ.get("WAN_MODEL_SIZE", "1.3B")
    model_id = f"Wan-AI/Wan2.1-T2V-{model_size}"
    print(f"[Wan2.1] Loading {model_id} …")

    pipe = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    print(f"[Wan2.1] Model loaded on {device}")


def handler(event):
    """RunPod handler: text prompt -> video (base64 mp4)."""
    try:
        inp = event.get("input", {})
        prompt = inp.get("prompt", "")
        if not prompt:
            return {"error": "prompt is required"}

        # Lazy-load model on first request
        load_model()

        num_frames = min(max(int(inp.get("num_frames", 81)), 17), 121)
        width = int(inp.get("width", 832))
        height = int(inp.get("height", 480))
        guidance_scale = float(inp.get("guidance_scale", 5.0))
        num_inference_steps = min(max(int(inp.get("num_inference_steps", 25)), 10), 50)

        start = time.time()
        print(f"[Wan2.1] Generating: {num_frames} frames, {width}x{height}, {num_inference_steps} steps")

        output = pipe(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        frames = output.frames[0]
        raw_path = f"/tmp/{uuid.uuid4()}.mp4"

        import imageio.v3 as iio
        import numpy as np

        frame_arrays = [np.array(f) for f in frames]
        iio.imwrite(raw_path, frame_arrays, fps=16, codec="libx264")

        gen_time = time.time() - start

        # Read file and base64 encode
        with open(raw_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        file_size = os.path.getsize(raw_path)

        # Cleanup
        try:
            os.remove(raw_path)
        except OSError:
            pass

        print(f"[Wan2.1] Done in {gen_time:.1f}s, {file_size} bytes")

        return {
            "video_base64": video_b64,
            "duration_seconds": round(num_frames / 16.0, 2),
            "inference_time": round(gen_time, 1),
            "total_time": round(gen_time, 1),
            "file_size_bytes": file_size,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
