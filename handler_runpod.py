"""RunPod Serverless handler — Wan2.1 Text-to-Video + Image-to-Video.
Lazy model load on first request. Supports T2V, I2V, lip_sync, and upscale.

Actions:
  - "t2v" (default): text-to-video using WanPipeline (1.3B or 14B)
  - "i2v": image-to-video using WanImageToVideoPipeline (14B-480P)
    Requires: image_base64 (PNG/JPG as base64)
  - "lip_sync": video_base64 + audio_base64 → Wav2Lip lip-synced video
    Optional: upscale=true → also 4x upscale after lip-sync
    Optional: output_name → save to /runpod-volume/output/<name>.mp4
  - "upscale": video_base64 → Real-ESRGAN 4x upscaled video
    Optional: output_name → save to /runpod-volume/output/<name>.mp4
  - "list_outputs" / "get_output" / "clean_outputs": manage network volume files

Deploy with WAN_MODE env var:
  - "t2v"  → only loads T2V pipeline (small GPU OK, 1.3B ≈ 8GB VRAM)
  - "i2v"  → only loads I2V pipeline (needs 48GB GPU, 14B ≈ 28GB VRAM)
  - "both" → loads whichever is requested first (needs 48GB GPU)
"""

import os, uuid, time, base64, io, subprocess, glob, sys

# Model cache: use network volume at /runpod-volume/models.
# Falls back to /app/models if volume not mounted (legacy).
# For temp files (mp4 encoding), use /dev/shm (RAM) if available.
VOLUME_MODEL_DIR = "/runpod-volume/models"
LEGACY_MODEL_DIR = "/app/models"
MODEL_DIR = VOLUME_MODEL_DIR if os.path.isdir(VOLUME_MODEL_DIR) else LEGACY_MODEL_DIR
if os.path.isdir("/dev/shm"):
    TMPDIR = "/dev/shm"
else:
    TMPDIR = "/tmp"

# Set HF cache so from_pretrained() finds the pre-downloaded model
os.environ["HF_HOME"] = MODEL_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_DIR
os.environ["TMPDIR"] = TMPDIR

print(f"[init] MODEL_DIR={MODEL_DIR} (volume={'yes' if MODEL_DIR==VOLUME_MODEL_DIR else 'no'})")

import runpod

WAN_MODE = os.environ.get("WAN_MODE", "both")  # "t2v", "i2v", or "both"

t2v_pipe = None
i2v_pipe = None
_upsampler = None
_wav2lip_model = None
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

OUTPUT_DIR = "/runpod-volume/output"
WEIGHTS_DIR = "/runpod-volume/weights"  # weights on network volume for fast cold starts
WEIGHTS_FALLBACK = "/app/weights"      # fallback if volume not mounted

WAV2LIP_URLS = [
    # Primary mirror (GitHub release — no auth required)
    "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth",
    # Alternate mirror (HuggingFace — public repo)
    "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth",
]
S3FD_PATH = "/root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth"
S3FD_URLS = [
    "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
]


def _download_first(urls, dst_path, label):
    import urllib.request
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    last_err = None
    for url in urls:
        try:
            print(f"[{label}] Downloading from {url}")
            urllib.request.urlretrieve(url, dst_path)
            if os.path.exists(dst_path) and os.path.getsize(dst_path) > 1024:
                return dst_path
        except Exception as e:
            last_err = e
    raise RuntimeError(f"{label} download failed: {last_err}")


def _ensure_s3fd_weights():
    if os.path.exists(S3FD_PATH):
        return S3FD_PATH
    return _download_first(S3FD_URLS, S3FD_PATH, "S3FD")


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

def save_to_volume(b64_data, output_name):
    """Save base64 video to network volume for overnight persistence."""
    if not output_name:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{output_name}.mp4")
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    print(f"[save] {out_path} ({os.path.getsize(out_path)} bytes)")


def handle_list_outputs(inp):
    """List generated videos stored on the network volume."""
    if not os.path.isdir(OUTPUT_DIR):
        return {"files": [], "total_bytes": 0}
    files = []
    total = 0
    for name in sorted(os.listdir(OUTPUT_DIR)):
        if name.endswith(".mp4"):
            sz = os.path.getsize(os.path.join(OUTPUT_DIR, name))
            files.append({"name": name, "size_bytes": sz})
            total += sz
    return {"files": files, "total_bytes": total, "count": len(files)}


def handle_get_output(inp):
    """Retrieve a generated video from the network volume as base64."""
    filename = inp.get("filename", "")
    if not filename:
        return {"error": "filename is required"}
    # Sanitize: only allow simple filenames, no path traversal
    safe_name = os.path.basename(filename)
    fpath = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.isfile(fpath):
        return {"error": f"file not found: {safe_name}"}
    with open(fpath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    sz = os.path.getsize(fpath)
    return {"filename": safe_name, "video_base64": b64, "size_bytes": sz}


def handle_clean_outputs(inp):
    """Remove downloaded videos from the network volume to free space."""
    filenames = inp.get("filenames", [])
    if not filenames:
        # Clean all
        if os.path.isdir(OUTPUT_DIR):
            removed = 0
            for name in os.listdir(OUTPUT_DIR):
                if name.endswith(".mp4"):
                    os.remove(os.path.join(OUTPUT_DIR, name))
                    removed += 1
            return {"removed": removed}
        return {"removed": 0}
    removed = 0
    for fn in filenames:
        safe = os.path.basename(fn)
        fpath = os.path.join(OUTPUT_DIR, safe)
        if os.path.isfile(fpath):
            os.remove(fpath)
            removed += 1
    return {"removed": removed}


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
    result = {"video_base64": b64, "duration_seconds": round(nf / 16, 2),
              "inference_time": round(dt, 1), "file_size_bytes": sz}
    # Persist to volume if output_name given
    output_name = inp.get("output_name")
    if output_name:
        save_to_volume(b64, output_name)
        result["saved_to_volume"] = f"{output_name}.mp4"
    return result

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
    result = {"video_base64": b64, "duration_seconds": round(nf / 16, 2),
              "inference_time": round(dt, 1), "file_size_bytes": sz,
              "resolution": f"{w}x{h}"}
    output_name = inp.get("output_name")
    if output_name:
        save_to_volume(b64, output_name)
        result["saved_to_volume"] = f"{output_name}.mp4"
    return result


# ── Real-ESRGAN Upscaler ─────────────────────────────────────────────

def load_upscaler():
    """Load Real-ESRGAN 4x anime upscaler (lazy, first call only)."""
    global _upsampler
    if _upsampler is not None:
        return _upsampler
    torch = ensure_torch()
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        # Check volume first, then fallback
        model_path = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus_anime_6B.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(WEIGHTS_FALLBACK, "RealESRGAN_x4plus_anime_6B.pth")
        if not os.path.exists(model_path):
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
            model_path = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus_anime_6B.pth")
            import urllib.request
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            print("[Upscale] Downloading to volume …")
            urllib.request.urlretrieve(url, model_path)

        _upsampler = RealESRGANer(
            scale=4, model_path=model_path, model=model,
            tile=256, tile_pad=10, half=True, device=device,
        )
        print("[Upscale] Real-ESRGAN loaded")
    except Exception as e:
        print(f"[Upscale] Failed to load: {e}")
        _upsampler = None
    return _upsampler


def upscale_video(input_path, output_path, target_height=1080):
    """Upscale video frames with Real-ESRGAN, then reassemble."""
    import cv2
    import numpy as np

    upscale = load_upscaler()
    if upscale is None:
        print("[Upscale] Skipping — upscaler not available")
        return input_path

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return input_path

    print(f"[Upscale] Processing {len(frames)} frames …")
    upscaled = []
    for frame in frames:
        output, _ = upscale.enhance(frame, outscale=4)
        h, w = output.shape[:2]
        if h > target_height:
            scale = target_height / h
            new_w = int(w * scale)
            output = cv2.resize(output, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4)
        upscaled.append(output)

    h, w = upscaled[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in upscaled:
        writer.write(f)
    writer.release()

    # Re-encode with libx264 for compatibility
    final = output_path.replace(".mp4", "_h264.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", output_path, "-c:v", "libx264",
                    "-preset", "fast", "-crf", "18", final], capture_output=True)
    if os.path.exists(final):
        os.replace(final, output_path)

    print(f"[Upscale] Done — {w}x{h}")
    return output_path


def handle_upscale(inp):
    """Standalone upscale: takes video_base64, returns 4x upscaled video."""
    video_b64 = inp.get("video_base64", "")
    if not video_b64:
        return {"error": "upscale requires video_base64"}

    target_height = int(inp.get("target_height", 1080))
    output_name = inp.get("output_name", "")
    job_id = str(uuid.uuid4())
    video_path = f"{TMPDIR}/{job_id}_video.mp4"
    out_path = f"{TMPDIR}/{job_id}_upscaled.mp4"

    with open(video_path, "wb") as f:
        f.write(base64.b64decode(video_b64))

    t0 = time.time()
    result_path = upscale_video(video_path, out_path, target_height=target_height)
    dt = time.time() - t0

    with open(result_path, "rb") as f:
        result_bytes = f.read()
    result_b64 = base64.b64encode(result_bytes).decode("utf-8")
    file_size = len(result_bytes)

    saved = ""
    if output_name:
        save_to_volume(result_bytes, output_name)
        saved = f"{output_name}.mp4"

    for p in glob.glob(f"{TMPDIR}/{job_id}*"):
        try: os.remove(p)
        except: pass

    return {"video_base64": result_b64, "upscale_time": round(dt, 1),
            "file_size_bytes": file_size, "saved_to_volume": saved}


# ── Wav2Lip Lip-Sync ─────────────────────────────────────────────────

def load_wav2lip():
    """Load Wav2Lip model (lazy, first call only)."""
    global _wav2lip_model
    if _wav2lip_model is not None:
        return _wav2lip_model
    torch = ensure_torch()
    try:
        wav2lip_dir = "/app/Wav2Lip"
        if os.path.isdir(wav2lip_dir) and wav2lip_dir not in sys.path:
            sys.path.insert(0, wav2lip_dir)

        print("[Wav2Lip] Importing model class...")
        from models import Wav2Lip as W2LModel

        # Check volume first, then fallback
        model_path = os.path.join(WEIGHTS_DIR, "wav2lip_gan.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(WEIGHTS_FALLBACK, "wav2lip_gan.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(WEIGHTS_DIR, "wav2lip_gan.pth")
            print(f"[Wav2Lip] Downloading weights to {model_path}...")
            _download_first(WAV2LIP_URLS, model_path, "Wav2Lip")
        print(f"[Wav2Lip] Loading weights from {model_path} ({os.path.getsize(model_path)} bytes)...")

        model = W2LModel()
        ckpt = torch.load(model_path, map_location=device)
        s = ckpt.get("state_dict", ckpt)
        model.load_state_dict({k.replace("module.", ""): v for k, v in s.items()})
        model = model.to(device).eval()
        _wav2lip_model = model
        print("[Wav2Lip] Model loaded successfully")
    except Exception as e:
        import traceback
        print(f"[Wav2Lip] Failed to load: {e}")
        traceback.print_exc()
        _wav2lip_model = None
    return _wav2lip_model


def apply_lip_sync(video_path, audio_path, output_path):
    """Audio-driven mouth morph for cartoon lip-sync.

    Uses smooth radial warp (cv2.remap) to gently open/close the mouth
    area based on audio energy.  No neural network, no blur, no hard edges.
    """
    import cv2
    import numpy as np
    import librosa

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_path
        ], capture_output=True)
        return output_path, {"lipsync_mode": "fallback", "fallback_error": "no_frames"}

    # ── Audio energy per video frame ──
    sr = 22050
    wav, _ = librosa.load(audio_path, sr=sr)
    hop = int(sr / fps)
    n_frames = len(frames)
    energy = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(wav))
        if start < len(wav):
            chunk = wav[start:end]
            energy[i] = np.sqrt(np.mean(chunk ** 2))

    if energy.max() > 0:
        energy = energy / energy.max()
    # Temporal smoothing (5-frame moving average for smoother motion)
    kernel = np.ones(5) / 5.0
    energy = np.convolve(energy, kernel, mode="same").astype(np.float32)
    energy = np.clip(energy, 0.0, 1.0)

    # ── Smooth radial warp parameters ──
    h_frame, w_frame = frames[0].shape[:2]
    mouth_cx = w_frame // 2
    mouth_cy = int(h_frame * 0.58)           # mouth centre (slightly above mid-lower)
    warp_rx  = int(w_frame * 0.12)            # horizontal radius of warp zone
    warp_ry  = int(h_frame * 0.08)            # vertical radius of warp zone
    max_disp = max(3, int(h_frame * 0.025))   # max outward displacement in pixels

    # Pre-build normalised displacement field (vectorised, computed once)
    ys = np.arange(h_frame, dtype=np.float32)[:, np.newaxis]   # (H,1)
    xs = np.arange(w_frame, dtype=np.float32)[np.newaxis, :]   # (1,W)
    dx_norm = (xs - mouth_cx) / max(warp_rx, 1)                # (1,W) → (H,W)
    dy_norm = (ys - mouth_cy) / max(warp_ry, 1)                # (H,1) → (H,W)
    r2 = dx_norm ** 2 + dy_norm ** 2                            # (H,W)

    # Cosine falloff inside ellipse → 1 at centre, 0 at boundary
    inside = r2 < 1.0
    strength = np.where(inside,
                        0.5 * (1.0 + np.cos(np.pi * np.sqrt(np.clip(r2, 0, 1)))),
                        0.0).astype(np.float32)

    # Displacement directions: push outward from mouth centre
    # Vertical push is dominant; horizontal is subtle
    unit_disp_y = -(dy_norm * strength)       # above → up, below → down
    unit_disp_x = -(dx_norm * strength * 0.3) # slight horizontal

    # Inner darkening mask (smaller ellipse for "mouth interior" hint)
    dark_r2 = (dx_norm * 1.8) ** 2 + (dy_norm * 1.3) ** 2     # tighter ellipse
    dark_mask = np.where(dark_r2 < 1.0,
                         0.5 * (1.0 + np.cos(np.pi * np.sqrt(np.clip(dark_r2, 0, 1)))),
                         0.0).astype(np.float32)

    # Base remap grids
    map_x_base = xs.repeat(h_frame, axis=0)   # (H,W)
    map_y_base = ys.repeat(1, w_frame)         # (H,W)

    result_frames = []
    for idx, frame in enumerate(frames):
        e = float(energy[idx]) if idx < len(energy) else 0.0
        if e < 0.05:
            result_frames.append(frame)
            continue

        d = e * max_disp
        map_x = (map_x_base + unit_disp_x * d).astype(np.float32)
        map_y = (map_y_base + unit_disp_y * d).astype(np.float32)
        warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)

        # Subtle darkening at the centre to hint at open mouth interior
        darken = 1.0 - dark_mask * e * 0.35   # at most 35% darker at peak
        warped = (warped.astype(np.float32) * darken[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)

        result_frames.append(warped)

    # Write output video
    h_out, w_out = result_frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_out, h_out))
    for f in result_frames:
        writer.write(f)
    writer.release()

    # Re-encode + merge audio
    final = output_path.replace(".mp4", "_final.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", output_path, "-i", audio_path,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                    "-shortest", final], capture_output=True)
    if os.path.exists(final):
        os.replace(final, output_path)

    processed = sum(1 for e in energy if e >= 0.05)
    print(f"[MouthMorph] Done — {len(result_frames)} frames, {processed} morphed")
    return output_path, {
        "lipsync_mode": "mouth_morph",
        "face_count": 1,
        "processed_frames": int(processed),
    }


def handle_lip_sync(inp):
    """Lip-sync: video + audio → synced video. Optional upscale after."""
    video_b64 = inp.get("video_base64", "")
    audio_b64 = inp.get("audio_base64", "")
    if not video_b64 or not audio_b64:
        return {"error": "lip_sync requires video_base64 and audio_base64"}

    do_upscale = inp.get("upscale", False)
    target_height = int(inp.get("target_height", 1080))
    output_name = inp.get("output_name", "")

    job_id = str(uuid.uuid4())
    video_path = f"{TMPDIR}/{job_id}_video.mp4"
    audio_path = f"{TMPDIR}/{job_id}_audio.mp3"
    sync_path = f"{TMPDIR}/{job_id}_synced.mp4"

    with open(video_path, "wb") as f:
        f.write(base64.b64decode(video_b64))
    with open(audio_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))

    t0 = time.time()
    result_path, lipsync_meta = apply_lip_sync(video_path, audio_path, sync_path)
    lipsync_time = time.time() - t0

    current_path = result_path
    upscale_time = 0

    if do_upscale:
        t1 = time.time()
        up_path = f"{TMPDIR}/{job_id}_upscaled.mp4"
        up_result = upscale_video(current_path, up_path, target_height=target_height)
        if up_result != current_path:
            # Re-merge audio onto upscaled video
            merged = f"{TMPDIR}/{job_id}_up_audio.mp4"
            subprocess.run(["ffmpeg", "-y", "-i", up_result, "-i", audio_path,
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                            "-map", "0:v:0", "-map", "1:a:0", "-shortest", merged],
                           capture_output=True)
            current_path = merged if os.path.exists(merged) else up_result
        upscale_time = time.time() - t1

    with open(current_path, "rb") as f:
        result_bytes = f.read()
    result_b64 = base64.b64encode(result_bytes).decode("utf-8")
    file_size = len(result_bytes)
    elapsed = time.time() - t0

    saved = ""
    if output_name:
        save_to_volume(result_bytes, output_name)
        saved = f"{output_name}.mp4"

    for p in glob.glob(f"{TMPDIR}/{job_id}*"):
        try: os.remove(p)
        except: pass

    return {"video_base64": result_b64, "lip_sync_time": round(lipsync_time, 1),
            "upscale_time": round(upscale_time, 1), "total_time": round(elapsed, 1),
            "file_size_bytes": file_size, "upscaled": do_upscale,
            "saved_to_volume": saved,
            "lipsync_mode": lipsync_meta.get("lipsync_mode", "unknown"),
            "face_count": lipsync_meta.get("face_count", 0),
            "processed_frames": lipsync_meta.get("processed_frames", 0),
            "fallback_error": lipsync_meta.get("fallback_error", "")}


def handle_download_weights(inp):
    """Download weights to network volume (run once to seed volume)."""
    import urllib.request
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    results = {}

    esrgan_path = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus_anime_6B.pth")
    if not os.path.exists(esrgan_path):
        print("[Weights] Downloading Real-ESRGAN …")
        urllib.request.urlretrieve(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            esrgan_path)
        results["realesrgan"] = f"downloaded ({os.path.getsize(esrgan_path)} bytes)"
    else:
        results["realesrgan"] = f"exists ({os.path.getsize(esrgan_path)} bytes)"

    wav2lip_path = os.path.join(WEIGHTS_DIR, "wav2lip_gan.pth")
    if not os.path.exists(wav2lip_path):
        print("[Weights] Downloading Wav2Lip …")
        try:
            _download_first(WAV2LIP_URLS, wav2lip_path, "Wav2Lip")
            results["wav2lip"] = f"downloaded ({os.path.getsize(wav2lip_path)} bytes)"
        except Exception as e:
            results["wav2lip"] = f"failed: {e}"
    else:
        results["wav2lip"] = f"exists ({os.path.getsize(wav2lip_path)} bytes)"

    if not os.path.exists(S3FD_PATH):
        print("[Weights] Downloading S3FD …")
        try:
            _download_first(S3FD_URLS, S3FD_PATH, "S3FD")
            results["s3fd"] = f"downloaded ({os.path.getsize(S3FD_PATH)} bytes)"
        except Exception as e:
            results["s3fd"] = f"failed: {e}"
    else:
        results["s3fd"] = f"exists ({os.path.getsize(S3FD_PATH)} bytes)"

    return results


def handle_check_deps(inp):
    """Diagnostic: check all dependencies for lip-sync pipeline."""
    result = {}

    # Wav2Lip repo
    wav2lip_dir = "/app/Wav2Lip"
    result["wav2lip_dir_exists"] = os.path.isdir(wav2lip_dir)
    if os.path.isdir(wav2lip_dir):
        result["wav2lip_files"] = os.listdir(wav2lip_dir)[:20]

    # Model import
    try:
        if os.path.isdir(wav2lip_dir) and wav2lip_dir not in sys.path:
            sys.path.insert(0, wav2lip_dir)
        from models import Wav2Lip as W2LModel
        result["wav2lip_import"] = "ok"
    except Exception as e:
        result["wav2lip_import"] = str(e)

    # Audio module
    try:
        from audio import load_wav, melspectrogram
        result["audio_import"] = "ok"
    except Exception as e:
        result["audio_import"] = str(e)

    # face_alignment
    try:
        import face_alignment
        result["face_alignment"] = "ok"
    except Exception as e:
        result["face_alignment"] = str(e)

    # Weights
    vol_w = os.path.join(WEIGHTS_DIR, "wav2lip_gan.pth")
    app_w = os.path.join(WEIGHTS_FALLBACK, "wav2lip_gan.pth")
    result["wav2lip_weights_volume"] = f"{os.path.getsize(vol_w)}B" if os.path.exists(vol_w) else "missing"
    result["wav2lip_weights_fallback"] = f"{os.path.getsize(app_w)}B" if os.path.exists(app_w) else "missing"
    result["s3fd_weights"] = f"{os.path.getsize(S3FD_PATH)}B" if os.path.exists(S3FD_PATH) else "missing"

    # GPU info
    try:
        torch = ensure_torch()
        result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        result["vram_mb"] = round(torch.cuda.get_device_properties(0).total_mem / 1024**2) if torch.cuda.is_available() else 0
    except Exception as e:
        result["gpu"] = str(e)

    return result


def handler(event):
    try:
        inp = event.get("input", {})
        action = inp.get("action", "t2v")

        # Storage/utility actions (no GPU needed — fast)
        if action == "list_outputs":
            return handle_list_outputs(inp)
        if action == "get_output":
            return handle_get_output(inp)
        if action == "clean_outputs":
            return handle_clean_outputs(inp)
        if action == "download_weights":
            return handle_download_weights(inp)
        if action == "check_deps":
            return handle_check_deps(inp)

        if action == "i2v":
            if WAN_MODE == "t2v":
                return {"error": "This endpoint is T2V-only. Deploy with WAN_MODE=i2v or WAN_MODE=both"}
            return handle_i2v(inp)
        elif action == "lip_sync":
            return handle_lip_sync(inp)
        elif action == "upscale":
            return handle_upscale(inp)
        else:
            if WAN_MODE == "i2v":
                return {"error": "This endpoint is I2V-only. Deploy with WAN_MODE=t2v or WAN_MODE=both"}
            return handle_t2v(inp)
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
