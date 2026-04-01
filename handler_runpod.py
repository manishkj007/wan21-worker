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
        model_path = "/app/weights/RealESRGAN_x4plus_anime_6B.pth"
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            import urllib.request
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            print("[Upscale] Downloading Real-ESRGAN anime weights …")
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

        from models import Wav2Lip as W2LModel

        model_path = "/app/weights/wav2lip_gan.pth"
        if not os.path.exists(model_path):
            print("[Wav2Lip] Weights not found, downloading …")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(
                "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth",
                model_path)

        model = W2LModel()
        ckpt = torch.load(model_path, map_location=device)
        s = ckpt.get("state_dict", ckpt)
        model.load_state_dict({k.replace("module.", ""): v for k, v in s.items()})
        model = model.to(device).eval()
        _wav2lip_model = model
        print("[Wav2Lip] Model loaded")
    except Exception as e:
        print(f"[Wav2Lip] Failed to load: {e}")
        _wav2lip_model = None
    return _wav2lip_model


def apply_lip_sync(video_path, audio_path, output_path):
    """Run Wav2Lip on video+audio. Falls back to ffmpeg merge if Wav2Lip fails."""
    import cv2
    import numpy as np
    torch = ensure_torch()

    model = load_wav2lip()
    if model is not None:
        try:
            from audio import load_wav, melspectrogram
            import face_alignment

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 16
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
            cap.release()
            if not frames:
                raise RuntimeError("No frames")

            wav = load_wav(audio_path, 16000)
            mel = melspectrogram(wav)

            mel_step = 16
            mel_chunks = [mel[:, i:i+mel_step] for i in range(0, mel.shape[1], mel_step)]

            # Loop video to match audio length
            needed = len(mel_chunks)
            while len(frames) < needed:
                frames.extend(frames[:needed - len(frames)])
            frames = frames[:needed]

            # Face detection
            det = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, flip_input=False, device=str(device))

            face_rects = []
            for frame in frames:
                try:
                    preds = det.get_detections_for_batch(np.array([frame[..., ::-1]]))
                    if preds and preds[0] is not None and len(preds[0]) > 0:
                        box = preds[0][0]
                        face_rects.append((int(max(0, box[1])), int(min(frame.shape[0], box[3])),
                                           int(max(0, box[0])), int(min(frame.shape[1], box[2]))))
                    else:
                        face_rects.append(None)
                except:
                    face_rects.append(None)

            # Check if we found any faces at all
            face_count = sum(1 for f in face_rects if f is not None)
            if face_count == 0:
                raise RuntimeError("No faces detected (likely cartoon — using fallback)")

            # Wav2Lip inference
            img_size = 96
            result_frames = []
            batch_size = 8
            for idx in range(0, len(frames), batch_size):
                bf = frames[idx:idx+batch_size]
                bm = mel_chunks[idx:idx+batch_size]
                bfd = face_rects[idx:idx+batch_size]
                img_batch, mel_batch, coords, origs = [], [], [], []
                for f, m, fd in zip(bf, bm, bfd):
                    if fd is None:
                        result_frames.append(f)
                        continue
                    y1, y2, x1, x2 = fd
                    face = cv2.resize(f[y1:y2, x1:x2], (img_size, img_size))
                    if m.shape[1] < mel_step:
                        m = np.pad(m, ((0, 0), (0, mel_step - m.shape[1])))
                    img_batch.append(face)
                    mel_batch.append(m)
                    coords.append(fd)
                    origs.append(f)
                if not img_batch:
                    continue
                img_arr = np.array(img_batch) / 255.0
                mel_arr = np.array(mel_batch)
                masked = img_arr.copy()
                masked[:, img_size//2:, :, :] = 0
                img_in = torch.FloatTensor(
                    np.transpose(np.concatenate([masked, img_arr], axis=3), (0, 3, 1, 2))).to(device)
                mel_in = torch.FloatTensor(mel_arr[:, np.newaxis, :, :]).to(device)
                with torch.no_grad():
                    pred = model(mel_in, img_in)
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
                for p, fd, orig in zip(pred, coords, origs):
                    y1, y2, x1, x2 = fd
                    p = cv2.resize(p.astype(np.uint8), (x2-x1, y2-y1))
                    result = orig.copy()
                    result[y1:y2, x1:x2] = p
                    result_frames.append(result)

            h, w = result_frames[0].shape[:2]
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
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

            print(f"[Wav2Lip] Done — {len(result_frames)} frames, {face_count} faces")
            return output_path
        except Exception as e:
            print(f"[Wav2Lip] Failed: {e} — using fallback")

    # ── Fallback: merge video+audio with ffmpeg, keep video at original speed ──
    print("[LipSync] Fallback: overlay audio on full-length video")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        output_path
    ], capture_output=True)
    return output_path


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
    result_path = apply_lip_sync(video_path, audio_path, sync_path)
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
            "saved_to_volume": saved}


def handler(event):
    try:
        inp = event.get("input", {})
        action = inp.get("action", "t2v")

        # Storage actions (no GPU needed — fast)
        if action == "list_outputs":
            return handle_list_outputs(inp)
        if action == "get_output":
            return handle_get_output(inp)
        if action == "clean_outputs":
            return handle_clean_outputs(inp)

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
