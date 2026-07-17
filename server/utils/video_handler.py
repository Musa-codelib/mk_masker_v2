import cv2
import os
import shutil
import base64
import subprocess
import numpy as np
import io
from pathlib import Path
from PIL import Image
from utils.errors import EmptyVideoError, UnsupportedVideoError

# ✅ HIDDEN CACHE PATH
CACHE_DIR = Path.home() / "Library" / "Caches" / "com.mkmasker.pro"
TEMP_DIR = CACHE_DIR / "temp_frames"

def setup_workspace():
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_DIR

def get_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise UnsupportedVideoError(
            "Could not open the video file.",
            detail=str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h, count = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
    cap.release()
    return fps, w, h, count

def extract_frames(video_path):
    setup_workspace()
    fps, orig_w, orig_h, total_frames = get_video_metadata(video_path)

    if total_frames <= 0 or orig_w <= 0 or orig_h <= 0:
        raise EmptyVideoError(
            "The video has no readable frames.",
            detail=f"fps={fps}, size={orig_w}x{orig_h}, frames={total_frames}")
    
    # AI Scale logic
    scale = min(1.0, 1024 / max(orig_w, orig_h))
    ai_w, ai_h = int(orig_w * scale), int(orig_h * scale)
    
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    while True:
        success, frame = cap.read()
        if not success: break
        # AI Proxy
        cv2.imwrite(str(TEMP_DIR / f"{count:08d}.jpg"), cv2.resize(frame, (ai_w, ai_h)))
        # Original (This is what you see in the UI)
        cv2.imwrite(str(TEMP_DIR / f"orig_{count:08d}.png"), frame)
        count += 1
    cap.release()
    # Return orig_w first, then orig_h
    return total_frames, fps, orig_w, orig_h, scale

def get_frame_base64(frame_idx, mask_np=None):
    """
    REPLICATED V1.2 LOGIC:
    Uses original PNG + exact cv2.addWeighted math to eliminate grain.
    """
    img_path = TEMP_DIR / f"orig_{frame_idx:08d}.png"
    if not img_path.exists(): return None
    
    # Load original 1080p frame (BGR)
    frame = cv2.imread(str(img_path))
    
    if mask_np is not None:
        # 1. Ensure mask is upscaled to 1080p
        if mask_np.shape[:2] != frame.shape[:2]:
            mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 2. Create blue overlay (BGR: Blue=255)
        overlay = np.zeros_like(frame)
        overlay[:, :, 0] = 255 
        
        # 3. EXACT V1.2 BLEND
        # mask_np must be uint8 (0 or 255)
        mask_uint8 = (mask_np > 127).astype(np.uint8) * 255
        mask_vis = cv2.bitwise_and(overlay, overlay, mask=mask_uint8)
        frame = cv2.addWeighted(frame, 1.0, mask_vis, 0.6, 0)

    # 4. Convert BGR to RGB for PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    # 5. Encode as PNG (Lossless)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

# DEPRECATED: superseded by utils.export.ExportPipeline + server.handle_start_processing.
# Kept for reference only. Do NOT route new processing through this function.
def compile_output_video(all_segments, original_video_path, output_dir, mode, fps, total_frames, orig_w, orig_h, progress_callback=None):
    output_dir = Path(output_dir)
    target_name = Path(original_video_path).stem
    blank_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    for i in range(total_frames):
        mask = (all_segments[i] * 255).astype(np.uint8) if i in all_segments else blank_mask
        mask_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        orig = cv2.imread(str(TEMP_DIR / f"orig_{i:08d}.png"))
        rgba = cv2.merge([cv2.split(orig)[0], cv2.split(orig)[1], cv2.split(orig)[2], mask_full])
        cv2.imwrite(str(TEMP_DIR / f"rgba_{i:08d}.png"), rgba)

    output_path = output_dir / f"cutout_{target_name}.mov"
    if mode == "balanced":
        cmd = ['ffmpeg', '-y', '-framerate', str(fps), '-i', str(TEMP_DIR / 'rgba_%08d.png'),
               '-c:v', 'hevc_videotoolbox', '-alpha_quality', '0.75', '-tag:v', 'hvc1', str(output_path)]
    else:
        cmd = ['ffmpeg', '-y', '-framerate', str(fps), '-i', str(TEMP_DIR / 'rgba_%08d.png'),
               '-c:v', 'prores_videotoolbox', '-profile:v', '4', '-pix_fmt', 'ayuv64le', str(output_path)]
    
    subprocess.run(cmd, check=True)
    return str(output_path)