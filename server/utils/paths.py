"""Path + binary resolution helpers.

Centralises where model weights, the ffmpeg binary and project data live, so the
rest of the app never hard-codes absolute paths. Also raises typed errors
(utils.errors) when a required asset is missing.
"""

from pathlib import Path
import os
import shutil
from typing import Callable

from utils.errors import FFmpegMissingError, WeightsMissingError


def get_project_root() -> Path:
    # paths.py lives in server/utils/, so three levels up is the project root.
    return Path(__file__).resolve().parent.parent.parent

def get_data_dir() -> Path:
    return get_project_root() / "data"

def get_weights_dir() -> Path:
    # Model checkpoints (.pt / .pth) resolution order:
    #   1. MK_MASKER_WEIGHTS_DIR env var override (production installer sets this)
    #   2. <project_root>/checkpoints (local development — repo ships with models)
    #   3. ~/Library/Caches/com.mkmasker.pro/weights (production / packaged app)
    override = os.environ.get("MK_MASKER_WEIGHTS_DIR")
    if override:
        return Path(override).expanduser()

    checkpoints = get_project_root() / "checkpoints"
    if checkpoints.exists():
        return checkpoints

    prod_dir = Path.home() / "Library/Caches/com.mkmasker.pro/weights"
    return prod_dir

def resolve_rvm_checkpoint() -> Path:
    # Locate the RVM TorchScript weights, failing with a clear message if absent.
    path = get_weights_dir() / "rvm_mobilenetv3.pth"
    if not path.exists():
        raise WeightsMissingError(
            "RVM weights missing.",
            detail=f"Expected {path}")
    return path

# Registry of supported SAM2 sizes. "small" is the default; "tiny" is the lighter alternative.
SAM2_MODELS: dict[str, dict] = {
    "tiny": {
        "cfg": "sam2.1_hiera_t.yaml",     # Hydra config for the SAM2.1 tiny model architecture
        "filename": "sam2_hiera_tiny.pt", # Matching checkpoint file in checkpoints/
        "label": "Tiny (Lite)",
    },
    "small": {
        "cfg": "sam2_hiera_s.yaml",       # Hydra config for the model architecture
        "filename": "sam2_hiera_small.pt", # Matching checkpoint file in checkpoints/
        "label": "Small",
    }
}

VALID_SAM2_VARIANTS = list(SAM2_MODELS.keys())
DEFAULT_SAM2_VARIANT = "small"

def get_sam2_model_info(variant: str) -> dict:
    # Fall back to "small" if an unknown variant is requested.
    return SAM2_MODELS.get(variant, SAM2_MODELS["small"])

def resolve_sam2_checkpoint(variant: str) -> Path:
    info = get_sam2_model_info(variant)
    path = get_weights_dir() / info["filename"]
    if not path.exists():
        raise WeightsMissingError(
            f"SAM2 ({variant}) weights missing.",
            detail=f"Expected {path}")
    return path

def resolve_sam2_config(variant: str) -> str:
    return get_sam2_model_info(variant)["cfg"]

def resolve_ffmpeg_binary() -> str:
    # Prefer a system-installed ffmpeg; fall back to the one bundled in bin/; otherwise error.
    system = shutil.which("ffmpeg")
    if system: return system
    
    bundled = get_project_root() / "bin" / "ffmpeg"
    if bundled.exists() and os.access(bundled, os.X_OK):
        return str(bundled)
    
    raise FFmpegMissingError(
        "FFmpeg not found.",
        detail="Run 'brew install ffmpeg' or place an ffmpeg binary in bin/.")
