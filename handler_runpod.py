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


def _detect_cartoon_face(gray, w, h):
    """Detect face region in a cartoon frame using multi-strategy approach.

    Returns list of (x, y, fw, fh) face bounding boxes sorted by area (largest first).
    Falls back to centre-frame heuristic if no faces found.
    """
    faces = []

    # Strategy 1: Haar cascade (works well on cartoon faces with big eyes)
    cascade_paths = [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]
    import cv2
    cascade = None
    for cp in cascade_paths:
        if os.path.exists(cp):
            cascade = cv2.CascadeClassifier(cp)
            break
    if cascade is None:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    if cascade is not None and not cascade.empty():
        dets = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=3,
                                        minSize=(w // 10, h // 10))
        for (fx, fy, fw, fh) in dets:
            faces.append((int(fx), int(fy), int(fw), int(fh)))

    # Strategy 2: If no Haar detections, try edge-density heuristic for cartoon eyes
    if not faces:
        edges = cv2.Canny(gray, 60, 150)
        # Scan for high-density eye regions (cartoon eyes = strong edges, round)
        cell_h, cell_w = h // 6, w // 6
        best_score, best_pos = 0, None
        for gy in range(1, 4):  # scan upper-mid rows
            for gx in range(1, 5):
                patch = edges[gy * cell_h:(gy + 1) * cell_h, gx * cell_w:(gx + 1) * cell_w]
                score = float(np.sum(patch > 0))
                if score > best_score:
                    best_score = score
                    best_pos = (gx * cell_w, gy * cell_h)
        if best_pos and best_score > cell_h * cell_w * 0.08:
            # Build face box around detected eye region
            ex, ey = best_pos
            fw = int(w * 0.45)
            fh = int(h * 0.55)
            fx = max(0, ex - fw // 4)
            fy = max(0, ey - fh // 5)
            faces.append((fx, fy, fw, fh))

    # Sort by area (largest first)
    faces.sort(key=lambda f: f[2] * f[3], reverse=True)
    return faces


def _pick_face(faces, face_hint, w_frame, h_frame):
    """Select the correct face based on hint from caller.

    face_hint: "left", "right", "center", "largest", or "x,y" explicit coords.
    """
    if not faces:
        return None

    if len(faces) == 1:
        return faces[0]

    if face_hint:
        hint = face_hint.strip().lower()
        if "," in hint:
            # Explicit centre coordinates
            parts = hint.split(",")
            tx, ty = float(parts[0]), float(parts[1])
            best, best_d = faces[0], float("inf")
            for f in faces:
                cx = f[0] + f[2] / 2
                cy = f[1] + f[3] / 2
                d = (cx - tx) ** 2 + (cy - ty) ** 2
                if d < best_d:
                    best, best_d = f, d
            return best
        elif hint == "left":
            return min(faces, key=lambda f: f[0] + f[2] / 2)
        elif hint == "right":
            return max(faces, key=lambda f: f[0] + f[2] / 2)
        elif hint == "center":
            return min(faces, key=lambda f: abs(f[0] + f[2] / 2 - w_frame / 2))

    # Default: largest face
    return faces[0]


def _extract_audio_features(audio_path, fps, n_frames):
    """Extract rich audio features: energy, vowel emphasis, consonant emphasis.

    Returns dict with per-frame arrays:
      energy:    overall RMS energy [0..1]
      vowel:     low-frequency emphasis (open mouth) [0..1]
      consonant: high-frequency emphasis (tight mouth) [0..1]
      pitch_var: pitch variation (for expression) [0..1]
    """
    import librosa
    sr = 22050
    wav, _ = librosa.load(audio_path, sr=sr)
    hop = int(sr / fps)

    energy = np.zeros(n_frames, dtype=np.float32)
    vowel = np.zeros(n_frames, dtype=np.float32)
    consonant = np.zeros(n_frames, dtype=np.float32)

    # Mel spectrogram for frequency analysis
    n_fft = min(2048, len(wav))
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop,
                                         n_mels=40, fmax=8000)
    mel_db = librosa.power_to_db(mel + 1e-10, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / max(1, mel_db.max() - mel_db.min())

    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(wav))
        if start < len(wav):
            chunk = wav[start:end]
            energy[i] = np.sqrt(np.mean(chunk ** 2))

        if i < mel_norm.shape[1]:
            col = mel_norm[:, i]
            # Low mels (0-12) = vowels/open mouth
            vowel[i] = float(np.mean(col[:12]))
            # High mels (20-40) = consonants/fricatives
            consonant[i] = float(np.mean(col[20:]))

    # Normalize
    for arr in [energy, vowel, consonant]:
        mx = arr.max()
        if mx > 0:
            arr[:] = arr / mx

    # Temporal smoothing (3-frame for responsive motion)
    kernel = np.ones(3) / 3.0
    energy = np.convolve(energy, kernel, mode="same").astype(np.float32)
    vowel = np.convolve(energy, kernel, mode="same").astype(np.float32)
    energy = np.clip(energy, 0.0, 1.0)
    vowel = np.clip(vowel, 0.0, 1.0)
    consonant = np.clip(consonant, 0.0, 1.0)

    # Pitch variation (for eyebrow + expression)
    try:
        pitches, voiced = librosa.piptrack(y=wav, sr=sr, hop_length=hop, fmin=80, fmax=600)
        pitch_per_frame = np.zeros(n_frames, dtype=np.float32)
        for i in range(min(n_frames, pitches.shape[1])):
            idx = pitches[:, i] > 0
            if idx.any():
                pitch_per_frame[i] = float(np.mean(pitches[idx, i]))
        if pitch_per_frame.max() > 0:
            pitch_per_frame = pitch_per_frame / pitch_per_frame.max()
        # Variation = deviation from running mean
        smooth_pitch = np.convolve(pitch_per_frame, np.ones(7) / 7, mode="same")
        pitch_var = np.abs(pitch_per_frame - smooth_pitch)
        if pitch_var.max() > 0:
            pitch_var = pitch_var / pitch_var.max()
    except Exception:
        pitch_var = np.zeros(n_frames, dtype=np.float32)

    return {
        "energy": energy,
        "vowel": vowel,
        "consonant": consonant,
        "pitch_var": pitch_var.astype(np.float32),
    }


def apply_lip_sync(video_path, audio_path, output_path, face_hint=None):
    """Audio-driven cartoon lip-sync with face detection, expressions, and emotions.

    Enhanced version with:
    - Automatic face detection (Haar + edge fallback)
    - Multi-face support with face_hint selection
    - Per-face jaw warp + mouth cavity
    - Eyebrow micro-lift during speech emphasis
    - Eye squint on loud/emotional moments
    - Micro head-tilt for natural animation feel
    - Spectral audio features (vowel vs consonant shapes)
    """
    import cv2
    import numpy as np

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

    h_frame, w_frame = frames[0].shape[:2]
    n_frames = len(frames)

    # ── Extract rich audio features ──
    audio = _extract_audio_features(audio_path, fps, n_frames)
    energy = audio["energy"]
    vowel = audio["vowel"]
    consonant = audio["consonant"]
    pitch_var = audio["pitch_var"]

    # ── Detect face in first frame ──
    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    all_faces = _detect_cartoon_face(gray0, w_frame, h_frame)

    if all_faces:
        face = _pick_face(all_faces, face_hint, w_frame, h_frame)
        fx, fy, fw, fh = face
        face_cx = fx + fw // 2
        face_cy = fy + fh // 2
        # Mouth sits at ~65% of face height from top
        mouth_cx = face_cx
        mouth_cy = fy + int(fh * 0.65)
        # Scale warp zones relative to detected face size
        jaw_rx = max(12, int(fw * 0.45))
        jaw_ry = max(10, int(fh * 0.35))
        max_jaw_drop = max(6, int(fh * 0.15))
        mouth_oval_rx = max(6, int(fw * 0.15))
        mouth_oval_ry_max = max(4, int(fh * 0.08))
        # Eye region (for expression warp: top 40% of face)
        eye_cy = fy + int(fh * 0.32)
        eye_rx = int(fw * 0.4)
        eye_ry = int(fh * 0.12)
        # Eyebrow region (above eyes)
        brow_cy = fy + int(fh * 0.22)
        brow_rx = int(fw * 0.35)
        brow_ry = int(fh * 0.08)
        face_detected = True
        detection = f"{fx},{fy},{fw},{fh}"
        print(f"[LipSync] Face detected at ({fx},{fy}) {fw}x{fh}, mouth=({mouth_cx},{mouth_cy})")
    else:
        # Fallback: centre-frame heuristic
        mouth_cx = w_frame // 2
        mouth_cy = int(h_frame * 0.60)
        jaw_rx = int(w_frame * 0.18)
        jaw_ry = int(h_frame * 0.14)
        max_jaw_drop = max(6, int(h_frame * 0.07))
        mouth_oval_rx = int(w_frame * 0.06)
        mouth_oval_ry_max = max(4, int(h_frame * 0.035))
        eye_cy = int(h_frame * 0.38)
        eye_rx = int(w_frame * 0.15)
        eye_ry = int(h_frame * 0.06)
        brow_cy = int(h_frame * 0.30)
        brow_rx = int(w_frame * 0.13)
        brow_ry = int(h_frame * 0.05)
        face_detected = False
        detection = "centre_heuristic"
        print(f"[LipSync] No face detected, using centre heuristic mouth=({mouth_cx},{mouth_cy})")

    # ── Track face across frames via template matching ──
    # Extract small patch around mouth region from frame 0 for tracking
    track_pad = max(jaw_rx, jaw_ry)
    tmpl_y1 = max(0, mouth_cy - track_pad)
    tmpl_y2 = min(h_frame, mouth_cy + track_pad)
    tmpl_x1 = max(0, mouth_cx - track_pad)
    tmpl_x2 = min(w_frame, mouth_cx + track_pad)
    face_template = gray0[tmpl_y1:tmpl_y2, tmpl_x1:tmpl_x2].copy()

    # Per-frame mouth positions (tracked)
    mouth_positions = [(mouth_cx, mouth_cy)]
    search_expand = int(track_pad * 0.6)
    prev_cx, prev_cy = mouth_cx, mouth_cy
    for i in range(1, n_frames):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        th, tw = face_template.shape
        sx1 = max(0, prev_cx - track_pad - search_expand)
        sy1 = max(0, prev_cy - track_pad - search_expand)
        sx2 = min(w_frame, prev_cx + track_pad + search_expand)
        sy2 = min(h_frame, prev_cy + track_pad + search_expand)
        region = gray[sy1:sy2, sx1:sx2]
        if region.shape[0] >= th and region.shape[1] >= tw:
            res = cv2.matchTemplate(region, face_template, cv2.TM_CCOEFF_NORMED)
            _, val, _, loc = cv2.minMaxLoc(res)
            if val > 0.3:
                nx = sx1 + loc[0] + tw // 2
                ny = sy1 + loc[1] + th // 2
                # Smooth: 70% new + 30% previous
                cx_t = int(prev_cx * 0.3 + nx * 0.7)
                cy_t = int(prev_cy * 0.3 + ny * 0.7)
                mouth_positions.append((cx_t, cy_t))
                prev_cx, prev_cy = cx_t, cy_t
                continue
        mouth_positions.append((prev_cx, prev_cy))

    # ── Pre-compute base grids ──
    ys = np.arange(h_frame, dtype=np.float32)[:, np.newaxis]
    xs = np.arange(w_frame, dtype=np.float32)[np.newaxis, :]
    map_x_base = np.repeat(xs, h_frame, axis=0).astype(np.float32)
    map_y_base = np.repeat(ys, w_frame, axis=1).astype(np.float32)

    result_frames = []
    active_count = 0

    for idx, frame in enumerate(frames):
        e = float(energy[idx]) if idx < n_frames else 0.0
        v = float(vowel[idx]) if idx < n_frames else 0.0
        c_val = float(consonant[idx]) if idx < n_frames else 0.0
        pv = float(pitch_var[idx]) if idx < n_frames else 0.0

        if e < 0.06:
            result_frames.append(frame)
            continue

        active_count += 1
        mcx, mcy = mouth_positions[idx]

        # ── Jaw zone (per-frame position) ──
        dy_from_mouth = (ys - mcy) / max(jaw_ry, 1)
        dx_from_mouth = (xs - mcx) / max(jaw_rx, 1)
        jaw_r2 = dx_from_mouth ** 2 + dy_from_mouth ** 2

        below_mouth = (ys > mcy).astype(np.float32)
        jaw_inside = (jaw_r2 < 1.0).astype(np.float32)
        cosine_falloff = np.where(
            jaw_r2 < 1.0,
            0.5 * (1.0 + np.cos(np.pi * np.sqrt(np.clip(jaw_r2, 0, 1)))),
            0.0
        ).astype(np.float32)
        jaw_strength = jaw_inside * below_mouth * cosine_falloff

        # Upper lip lift zone
        above_zone = ((ys > mcy - jaw_ry * 0.4) & (ys <= mcy)).astype(np.float32)
        lip_dist = np.abs(ys - mcy) / max(jaw_ry * 0.4, 1)
        lip_strength = above_zone * jaw_inside * np.where(
            lip_dist < 1.0,
            0.5 * (1.0 + np.cos(np.pi * np.clip(lip_dist, 0, 1))),
            0.0
        ).astype(np.float32) * 0.3

        # ── Vowel = wide open jaw; Consonant = tighter, less drop ──
        vowel_factor = max(v, 0.3)  # vowels open wide
        drop = e * max_jaw_drop * (0.6 + 0.4 * vowel_factor)

        disp_y = jaw_strength * drop - lip_strength * drop
        # Horizontal stretch: vowels widen more, consonants narrow
        h_stretch = jaw_strength * e * (0.12 + 0.08 * vowel_factor - 0.05 * c_val)
        disp_x = dx_from_mouth * jaw_rx * h_stretch

        # ── Expression: eyebrow lift on emphasis / pitch changes ──
        brow_amount = max(pv * 0.4, e * 0.15) * max_jaw_drop * 0.3
        if brow_amount > 1.0:
            dy_brow = (ys - brow_cy) / max(brow_ry, 1)
            dx_brow = (xs - mcx) / max(brow_rx, 1)
            brow_r2 = dx_brow ** 2 + dy_brow ** 2
            brow_mask = np.where(
                brow_r2 < 1.0,
                0.5 * (1.0 + np.cos(np.pi * np.sqrt(np.clip(brow_r2, 0, 1)))),
                0.0
            ).astype(np.float32)
            # Only lift pixels above brow centre (eyebrows go UP)
            brow_above = (ys < brow_cy + brow_ry * 0.3).astype(np.float32)
            disp_y = disp_y - brow_mask * brow_above * brow_amount

        # ── Expression: subtle eye squint on loud moments ──
        squint_amount = max(0, e - 0.6) * max_jaw_drop * 0.15
        if squint_amount > 0.5:
            dy_eye = (ys - eye_cy) / max(eye_ry, 1)
            dx_eye = (xs - mcx) / max(eye_rx, 1)
            eye_r2 = dx_eye ** 2 + dy_eye ** 2
            eye_mask = np.where(
                eye_r2 < 1.0,
                0.5 * (1.0 + np.cos(np.pi * np.sqrt(np.clip(eye_r2, 0, 1)))),
                0.0
            ).astype(np.float32)
            # Push lower eyelid UP slightly (squint)
            below_eye = (ys > eye_cy).astype(np.float32)
            disp_y = disp_y - eye_mask * below_eye * squint_amount

        # ── Micro head tilt (subtle sine oscillation for life-like feel) ──
        tilt_phase = idx / max(fps, 1) * 2.5  # slow oscillation
        tilt_amount = e * 0.8  # only when speaking
        micro_tilt = np.sin(tilt_phase) * tilt_amount
        # Tilt = small rotation → left side up, right side down
        tilt_disp = (xs - mcx) / max(w_frame * 0.5, 1) * micro_tilt
        disp_y = disp_y + tilt_disp

        map_y = (map_y_base - disp_y).astype(np.float32)
        map_x = (map_x_base - disp_x).astype(np.float32)

        warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)

        # ── Mouth cavity (dark oval) — shape depends on vowel vs consonant ──
        # Vowels: tall oval (open mouth); Consonants: wide but flat
        oval_ry = max(2, int(mouth_oval_ry_max * e * (0.5 + 0.5 * vowel_factor)))
        oval_rx_frame = max(4, int(mouth_oval_rx * (0.7 + 0.3 * vowel_factor + 0.2 * c_val)))
        if oval_ry > 2:
            m_dx = (xs - mcx) / max(oval_rx_frame, 1)
            m_dy = (ys - mcy) / max(oval_ry, 1)
            m_r2 = m_dx ** 2 + m_dy ** 2
            cavity_mask = np.where(
                m_r2 < 1.0,
                0.5 * (1.0 + np.cos(np.pi * np.sqrt(np.clip(m_r2, 0, 1)))),
                0.0
            ).astype(np.float32)
            cavity_strength = cavity_mask * min(e * 1.2, 0.85)
            # Mouth colour: slightly warmer for vowels
            mouth_color = np.array([30 + int(10 * v), 20, 45 + int(15 * v)],
                                   dtype=np.float32)
            warped = (
                warped.astype(np.float32) * (1.0 - cavity_strength[:, :, np.newaxis])
                + mouth_color[np.newaxis, np.newaxis, :] * cavity_strength[:, :, np.newaxis]
            ).clip(0, 255).astype(np.uint8)

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

    processed = int(active_count)
    print(f"[LipSync] Done — {len(result_frames)} frames, {processed} morphed, "
          f"face={'detected' if face_detected else 'heuristic'} at {detection}")
    return output_path, {
        "lipsync_mode": "face_detect_morph",
        "face_count": len(all_faces) if face_detected else 0,
        "face_detected": face_detected,
        "face_box": detection,
        "processed_frames": processed,
    }


def handle_lip_sync(inp):
    """Lip-sync: video + audio → synced video. Optional upscale after.

    New parameters:
      face_hint: "left", "right", "center", "largest", or "x,y" coords
                 Tells the face picker which character is speaking.
      character_name: Informational — logged for debugging.
    """
    video_b64 = inp.get("video_base64", "")
    audio_b64 = inp.get("audio_base64", "")
    if not video_b64 or not audio_b64:
        return {"error": "lip_sync requires video_base64 and audio_base64"}

    do_upscale = inp.get("upscale", False)
    target_height = int(inp.get("target_height", 1080))
    output_name = inp.get("output_name", "")
    face_hint = inp.get("face_hint", "")
    character_name = inp.get("character_name", "")

    if character_name:
        print(f"[LipSync] Character: {character_name}, face_hint: {face_hint}")

    job_id = str(uuid.uuid4())
    video_path = f"{TMPDIR}/{job_id}_video.mp4"
    audio_path = f"{TMPDIR}/{job_id}_audio.mp3"
    sync_path = f"{TMPDIR}/{job_id}_synced.mp4"

    with open(video_path, "wb") as f:
        f.write(base64.b64decode(video_b64))
    with open(audio_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))

    t0 = time.time()
    result_path, lipsync_meta = apply_lip_sync(video_path, audio_path, sync_path,
                                                face_hint=face_hint or None)
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
            "character_name": character_name,
            "lipsync_mode": lipsync_meta.get("lipsync_mode", "unknown"),
            "face_count": lipsync_meta.get("face_count", 0),
            "face_detected": lipsync_meta.get("face_detected", False),
            "face_box": lipsync_meta.get("face_box", ""),
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
