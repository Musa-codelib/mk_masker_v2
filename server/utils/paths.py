from pathlib import Path
import os
import sys
import shutil
import subprocess
from typing import Callable



def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def get_data_dir() -> Path:
    return get_project_root() / "data"

def get_weights_dir() -> Path:
    return get_project_root() / "checkpoints"

def resolve_rvm_checkpoint() -> Path:
    return get_weights_dir() / "rvm_mobilenetv3.pth"

# ✅ UPDATED REGISTRY TO MATCH YOUR FILE
SAM2_MODELS: dict[str, dict] = {
    "small": {
        "cfg": "sam2_hiera_s.yaml",
        "filename": "sam2_hiera_small.pt", # Matches your actual file
        "label": "Small",
    }
}

VALID_SAM2_VARIANTS = list(SAM2_MODELS.keys())
DEFAULT_SAM2_VARIANT = "small"

def get_sam2_model_info(variant: str) -> dict:
    return SAM2_MODELS.get(variant, SAM2_MODELS["small"])

def resolve_sam2_checkpoint(variant: str) -> Path:
    info = get_sam2_model_info(variant)
    return get_weights_dir() / info["filename"]

def resolve_sam2_config(variant: str) -> str:
    return get_sam2_model_info(variant)["cfg"]

def resolve_ffmpeg_binary() -> str:
    # ✅ FIX: Prioritize System FFmpeg over the broken bundled one
    system = shutil.which("ffmpeg")
    if system: return system
    
    bundled = get_project_root() / "bin" / "ffmpeg"
    if bundled.exists() and os.access(bundled, os.X_OK):
        return str(bundled)
    
    raise RuntimeError("FFmpeg not found. Run 'brew install ffmpeg'.")


def resolve_ffprobe_binary() -> str:
    """
    Resolve the ffprobe binary path using the following priority:

    1. KVN_ROTOSCOPE_FFPROBE environment variable override
    2. ffprobe next to the resolved ffmpeg binary
    3. System ffprobe found in PATH
    """
    # 1. Explicit override
    env_override = os.environ.get("KVN_ROTOSCOPE_FFPROBE")
    if env_override:
        override = Path(env_override).expanduser()
        if override.exists() and os.access(override, os.X_OK):
            return str(override)

    # 2. Sibling of ffmpeg (try with and without .exe for Windows)
    try:
        ffmpeg_path = resolve_ffmpeg_binary()
        parent = Path(ffmpeg_path).parent
        for name in ("ffprobe.exe", "ffprobe"):
            candidate = parent / name
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate)
    except RuntimeError:
        pass

    # 3. System ffprobe
    system = shutil.which("ffprobe")
    if system:
        return system

    if sys.platform == "win32":
        raise RuntimeError(
            "FFprobe not found. Set KVN_ROTOSCOPE_FFPROBE to the path of an ffprobe binary,\n"
            "or try reinstalling KVN Rotoscope."
        )
    raise RuntimeError(
        "FFprobe not found. Install it via Homebrew:\n"
        "    brew install ffmpeg\n"
        "(ffprobe ships alongside ffmpeg)"
    )


# ---------------------------------------------------------------------------
# ffprobe helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _ffprobe_fps(file_path: str) -> float:
    ffprobe_bin = resolve_ffprobe_binary()
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        file_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        rate_str = result.stdout.strip()
        if not rate_str:
            raise RuntimeError("ffprobe returned empty frame rate")

        if "/" in rate_str:
            num, den = rate_str.split("/", 1)
            return float(num) / float(den)

        return float(rate_str)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffprobe timed out on {file_path}")
    except (ValueError, ZeroDivisionError) as e:
        raise RuntimeError(f"Failed to parse frame rate {rate_str!r}: {e}")


def get_source_fps(file_path: str, media_pool_item=None) -> float:
    if media_pool_item is not None:
        try:
            props = media_pool_item.GetClipProperty() or {}
            for key in ["FPS", "Video Frame Rate", "Video FR", "Frame Rate"]:
                if key in props:
                    val = props[key]
                    if isinstance(val, str):
                        val = val.lower().replace("fps", "").strip()
                    try:
                        fps = float(val)
                        if fps > 0:
                            return fps
                    except (ValueError, TypeError):
                        continue
        except Exception:
            pass

    return _ffprobe_fps(file_path)
