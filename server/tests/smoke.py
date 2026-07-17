"""Smoke test for the ExportPipeline wiring.

This is the regression guard against the "mess-up" problem: it verifies that
render_matte_frames -> prepare_rgba_frames -> encode_video are wired correctly
for ALL THREE export modes without needing the AI models (SAM2/RVM) or a GPU.

Run from the repo root:
    python -m server.tests.smoke
or:
    PYTHONPATH=server python server/tests/smoke.py
"""
import shutil
import sys
import traceback
from pathlib import Path

# Allow running both as a module and as a script.
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SERVER = ROOT / "server"
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

import numpy as np
from PIL import Image

from utils.export import ExportPipeline
from utils.paths import resolve_ffmpeg_binary
from utils.errors import FFmpegMissingError


def _make_synthetic_frames(work: Path, n: int = 5):
    frames_dir = work / "frames"
    masks_dir = work / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    h = w = 64
    for i in range(n):
        # orig_{i:08d}.png — solid color frames
        orig = np.full((h, w, 3), (i * 10 % 255, 100, 200), dtype=np.uint8)
        Image.fromarray(orig).save(frames_dir / f"orig_{i:08d}.png")
        # mask_{i:04d}.png — raw AI mask (0/255)
        mask = np.full((h, w), 255 if i % 2 == 0 else 0, dtype=np.uint8)
        Image.fromarray(mask, mode="L").save(masks_dir / f"mask_{i:04d}.png")
    return frames_dir, masks_dir


def _run_mode(pipeline, mode, work, frames_dir, masks_dir):
    render_temp = work / "render_temp"
    if render_temp.exists():
        shutil.rmtree(render_temp)
    render_temp.mkdir(parents=True)

    # Phase 1: masks -> matte PNGs
    pipeline.render_matte_frames(masks_dir, render_temp)
    assert (render_temp / "matte_0000.png").exists(), f"[{mode}] matte_0000.png missing"

    if mode in ("prores", "balanced"):
        # Phase 2: orig + matte -> rgba PNGs (the fragile wiring)
        pipeline.prepare_rgba_frames(frames_dir, render_temp, render_temp, total_frames=5)
        assert (render_temp / "rgba_0000.png").exists(), f"[{mode}] rgba_0000.png missing"

    # Phase 3: encode (needs ffmpeg). Skip if not installed.
    try:
        ffmpeg = resolve_ffmpeg_binary()
    except FFmpegMissingError:
        print(f"  [{mode}] SKIP encode (ffmpeg not found) — matte/rgba wiring OK")
        return

    out = pipeline.encode_video(render_temp, work / f"cutout_{mode}", fps=24.0, mode=mode)
    assert Path(out).exists(), f"[{mode}] encode produced no file"
    print(f"  [{mode}] OK -> {out}")


def main():
    print("=== Mk Masker export smoke test ===")
    work = ROOT / "server" / "tests" / "_smoke_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    frames_dir, masks_dir = _make_synthetic_frames(work)
    pipeline = ExportPipeline()

    for mode in ("bw", "prores", "balanced"):
        print(f"Mode: {mode}")
        _run_mode(pipeline, mode, work, frames_dir, masks_dir)

    shutil.rmtree(work, ignore_errors=True)
    print("=== ALL EXPORT MODES WIRED CORRECTLY ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
