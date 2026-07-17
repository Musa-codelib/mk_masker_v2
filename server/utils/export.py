"""Video export pipeline.

Takes the per-frame masks produced by the AI engines and turns them into a final
masked video. Three sequential stages:

  1. render_matte_frames  : masks -> clean B&W "matte_*.png" files
  2. prepare_rgba_frames  : original color + matte -> 4-channel "rgba_*.png" (alpha = mask)
  3. encode_video         : feeds the PNG sequence to ffmpeg and writes the output file

`on_progress(current, total)` and `on_phase(label)` callbacks let the UI show a
live progress bar / phase label during the (potentially long) export.
"""

import subprocess
from pathlib import Path
from typing import Callable, Optional
from PIL import Image
import cv2
import numpy as np
from utils.paths import resolve_ffmpeg_binary
from utils.errors import FFmpegFailedError, FrameMissingError

# on_progress signature: (current: int, total: int) -> None
ProgressCb = Optional[Callable[[int, int], None]]
# on_phase signature: (label: str) -> None  (one-shot phase change notification)
PhaseCb = Optional[Callable[[str], None]]


class ExportPipeline:
    def render_matte_frames(self, masks_dir: Path, output_dir: Path,
                            on_progress: ProgressCb = None,
                            on_phase: PhaseCb = None) -> Path:
        """Converts raw AI masks to clean sequential B&W PNGs (matte_*.png)."""
        masks_dir, output_dir = Path(masks_dir), Path(output_dir)
        if on_phase:
            on_phase("Rendering alpha masks…")
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_files = sorted(masks_dir.glob("mask_*.png"))
        total = len(mask_files)
        if total == 0:
            raise FrameMissingError(
                "No mask frames were produced by the AI engine.",
                detail=f"Searched {masks_dir} for mask_*.png")
        for i, mask_path in enumerate(mask_files):
            # Normalise to pure grayscale and re-save with a sequential index.
            matte_img = Image.open(mask_path).convert("L")
            matte_img.save(output_dir / f"matte_{i:04d}.png")
            if on_progress:
                on_progress(i + 1, total)
        return output_dir

    def prepare_rgba_frames(self, frames_dir: Path, matte_dir: Path, rgba_dir: Path,
                            total_frames: int,
                            on_progress: ProgressCb = None,
                            on_phase: PhaseCb = None) -> Path:
        """Merges original color PNGs with mattes into 4-channel RGBA PNGs.

        Args:
            frames_dir: directory containing orig_{i:08d}.png (full-res source).
            matte_dir:  directory containing matte_{i:04d}.png (from render_matte_frames).
            rgba_dir:   output directory for rgba_{i:04d}.png.
        """
        frames_dir, matte_dir, rgba_dir = Path(frames_dir), Path(matte_dir), Path(rgba_dir)
        if on_phase:
            on_phase("Preparing RGBA frames…")
        rgba_dir.mkdir(parents=True, exist_ok=True)

        # Guard: the matte sequence must exist before we merge.
        first_matte = matte_dir / "matte_0000.png"
        if not first_matte.exists():
            raise FrameMissingError(
                "Matte frames missing before RGBA merge.",
                detail=f"Expected {first_matte}")

        for i in range(total_frames):
            orig_path = frames_dir / f"orig_{i:08d}.png"
            matte_path = matte_dir / f"matte_{i:04d}.png"

            orig = cv2.imread(str(orig_path))
            if orig is None:
                raise FrameMissingError(
                    "Missing original frame for RGBA export.",
                    detail=str(orig_path))

            mask = cv2.imread(str(matte_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # If matte is missing for a frame, make it fully transparent.
                mask = np.zeros((orig.shape[0], orig.shape[1]), dtype=np.uint8)

            # Ensure mask matches frame size
            if mask.shape[:2] != orig.shape[:2]:
                mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

            b, g, r = cv2.split(orig)
            rgba = cv2.merge([b, g, r, mask])
            cv2.imwrite(str(rgba_dir / f"rgba_{i:04d}.png"), rgba)
            if on_progress:
                on_progress(i + 1, total_frames)

        return rgba_dir

    def encode_video(self, input_dir: Path, output_path: Path, fps: float, mode: str,
                     on_phase: PhaseCb = None) -> Path:
        """Universal FFmpeg dispatcher.

        mode:
            "bw"        -> Grayscale MP4 (matte only)
            "balanced"  -> H.265 / HEVC with alpha (macOS VideoToolbox)
            "prores"    -> ProRes 4444 with alpha (macOS VideoToolbox)
        """
        input_dir, output_path = Path(input_dir), Path(output_path)
        ffmpeg_bin = resolve_ffmpeg_binary()

        # Build the ffmpeg command for the requested output format.
        if mode == "bw":
            label = "Encoding B&W mask…"
            input_pattern = str(input_dir / "matte_%04d.png")
            out_file = output_path.with_suffix(".mp4")
            cmd = [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", input_pattern,
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                   str(out_file)]
        elif mode == "balanced":
            label = "Encoding H.265 (alpha)…"
            input_pattern = str(input_dir / "rgba_%04d.png")
            out_file = output_path.with_suffix(".mov")
            cmd = [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", input_pattern,
                   "-c:v", "hevc_videotoolbox", "-alpha_quality", "0.75", "-tag:v", "hvc1",
                   str(out_file)]
        else:
            label = "Encoding ProRes 4444 (alpha)…"
            input_pattern = str(input_dir / "rgba_%04d.png")
            out_file = output_path.with_suffix(".mov")
            cmd = [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", input_pattern,
                   "-c:v", "prores_videotoolbox", "-profile:v", "4", "-pix_fmt", "ayuv64le",
                   str(out_file)]

        if on_phase:
            on_phase(label)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise FFmpegFailedError(
                f"FFmpeg failed while {label}",
                detail=result.stderr[-2000:] if result.stderr else "No stderr captured.")
        return out_file
