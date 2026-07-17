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

# Frames are extracted into a hidden cache folder so they don't clutter the user's project.
CACHE_DIR = Path.home() / "Library" / "Caches" / "com.mkmasker.pro"
TEMP_DIR = CACHE_DIR / "temp_frames"

def setup_workspace():
    # Wipe any previous extraction so we always start from a clean slate.
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_DIR

def get_video_metadata(video_path):
    # Open the video with OpenCV and read its basic properties.
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
    # 1. Reset the workspace and read the source video's metadata.
    setup_workspace()
    fps, orig_w, orig_h, total_frames = get_video_metadata(video_path)

    # 2. Guard against broken/empty videos before doing any heavy work.
    if total_frames <= 0 or orig_w <= 0 or orig_h <= 0:
        raise EmptyVideoError(
            "The video has no readable frames.",
            detail=f"fps={fps}, size={orig_w}x{orig_h}, frames={total_frames}")
    
    # 3. The AI models run faster on smaller images, so we downscale long edges to <=1024px.
    #    `scale` lets us map UI click coordinates back to full resolution later.
    scale = min(1.0, 1024 / max(orig_w, orig_h))
    ai_w, ai_h = int(orig_w * scale), int(orig_h * scale)
    
    # 4. Decode every frame and save two copies:
    #    - a small JPG used by the AI engines for tracking/matting
    #    - a full-res PNG (orig_*) shown to the user in the UI
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    while True:
        success, frame = cap.read()
        if not success: break
        cv2.imwrite(str(TEMP_DIR / f"{count:08d}.jpg"), cv2.resize(frame, (ai_w, ai_h)))
        cv2.imwrite(str(TEMP_DIR / f"orig_{count:08d}.png"), frame)
        count += 1
    cap.release()
    # Returns: (total frame count, fps, original width, original height, downscale factor)
    return total_frames, fps, orig_w, orig_h, scale

def get_frame_base64(frame_idx, mask_np=None):
    """Load a stored frame and return it as a base64 PNG data-URI for the UI canvas.

    If `mask_np` is supplied (a 0/255 alpha mask), it is blended as a translucent blue
    overlay on top of the frame so the user can see the current selection.
    """
    img_path = TEMP_DIR / f"orig_{frame_idx:08d}.png"
    if not img_path.exists(): return None
    
    # Load the full-resolution frame (OpenCV reads images as BGR).
    frame = cv2.imread(str(img_path))
    
    if mask_np is not None:
        # Resize the mask to match the frame if dimensions differ.
        if mask_np.shape[:2] != frame.shape[:2]:
            mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Build a solid-blue overlay, then keep only the masked region (bitwise_and with the mask).
        overlay = np.zeros_like(frame)
        overlay[:, :, 0] = 255 
        
        mask_uint8 = (mask_np > 127).astype(np.uint8) * 255
        mask_vis = cv2.bitwise_and(overlay, overlay, mask=mask_uint8)
        # Blend the blue mask over the frame at 60% opacity.
        frame = cv2.addWeighted(frame, 1.0, mask_vis, 0.6, 0)

    # Convert BGR -> RGB (PIL expects RGB) and encode losslessly as PNG.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

# DEPRECATED / UNUSED: the live export is handled by utils.export.ExportPipeline and
# invoked from server.handle_start_processing. This older function is kept only as a
# reference implementation and must NOT be called by new code.
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