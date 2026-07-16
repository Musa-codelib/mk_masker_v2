import subprocess
from pathlib import Path
from typing import Callable, Optional
from PIL import Image
import cv2
import numpy as np
from utils.paths import resolve_ffmpeg_binary

class ExportPipeline:
    def render_matte_frames(self, masks_dir: Path, output_dir: Path, on_progress: Optional[Callable[[int, int], None]] = None) -> Path:
        """Converts raw AI masks to clean sequential B&W PNGs"""
        masks_dir, output_dir = Path(masks_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_files = sorted(masks_dir.glob("mask_*.png"))
        total = len(mask_files)
        for i, mask_path in enumerate(mask_files):
            matte_img = Image.open(mask_path).convert("L")
            matte_img.save(output_dir / f"matte_{i:04d}.png")
            if on_progress: on_progress(i, total)
        return output_dir

    def prepare_rgba_frames(self, frames_dir: Path, masks_dir: Path, output_dir: Path, total_frames: int):
        """Merges original color PNGs with masks into 4-channel RGBA PNGs"""
        output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(total_frames):
            orig = cv2.imread(str(frames_dir / f"orig_{i:08d}.png"))
            mask = cv2.imread(str(masks_dir / f"matte_{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
            
            # Ensure mask matches frame size
            if mask.shape[:2] != orig.shape[:2]:
                mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            b, g, r = cv2.split(orig)
            rgba = cv2.merge([b, g, r, mask])
            cv2.imwrite(str(output_dir / f"rgba_{i:04d}.png"), rgba)

    def encode_video(self, input_dir: Path, output_path: Path, fps: float, mode: str) -> Path:
        """Universal FFmpeg dispatcher"""
        ffmpeg_bin = resolve_ffmpeg_binary()
        
        if mode == "bw":
            # Grayscale MP4 (Mask Only)
            input_pattern = str(input_dir / "matte_%04d.png")
            cmd = [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", input_pattern,
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(output_path.with_suffix(".mp4"))]
        
        elif mode == "balanced":
            # H.265 with Alpha (HEVC)
            input_pattern = str(input_dir / "rgba_%04d.png")
            cmd = [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", input_pattern,
                   "-c:v", "hevc_videotoolbox", "-alpha_quality", "0.75", "-tag:v", "hvc1", str(output_path.with_suffix(".mov"))]
        
        else:
            # ProRes 4444 (High Quality Alpha)
            input_pattern = str(input_dir / "rgba_%04d.png")
            cmd = [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", input_pattern,
                   "-c:v", "prores_videotoolbox", "-profile:v", "4", "-pix_fmt", "ayuv64le", str(output_path.with_suffix(".mov"))]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        return output_path