#!/usr/bin/env python3
"""
Model download script for Mk Masker Pro.

Downloads SAM2 model checkpoints to the weights directory.
Can be called directly by the Electron app or from the server.
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path

# Model definitions (URLs from official SAM2 release)
MODELS = {
    "tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_tiny.pt",
        "filename": "sam2_hiera_tiny.pt",
        "size_mb": 38,
    },
    "small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_small.pt",
        "filename": "sam2_hiera_small.pt",
        "size_mb": 176,
    },
}


def get_weights_dir() -> Path:
    """Return the directory where model weights should be stored.

    Uses the same resolution logic as server/utils/paths.py so the wizard,
    the server, and the installer all agree on where models live:
      1. MK_MASKER_WEIGHTS_DIR env var
      2. <project_root>/checkpoints (local dev — repo ships with models)
      3. ~/Library/Caches/com.mkmasker.pro/weights (production / packaged app)
    """
    override = os.environ.get("MK_MASKER_WEIGHTS_DIR")
    if override:
        return Path(override).expanduser()

    checkpoints = Path(__file__).resolve().parent.parent.parent / "checkpoints"
    if checkpoints.exists():
        return checkpoints

    prod_dir = Path.home() / "Library/Caches/com.mkmasker.pro/weights"
    return prod_dir


def download_model(variant: str, on_progress=None) -> Path:
    """Download a model checkpoint if not already present."""
    if variant not in MODELS:
        raise ValueError(f"Unknown model variant: {variant}. Valid: {list(MODELS.keys())}")

    model_info = MODELS[variant]
    weights_dir = get_weights_dir()
    weights_dir.mkdir(parents=True, exist_ok=True)
    dest = weights_dir / model_info["filename"]

    if dest.exists():
        return dest

    print(f"Downloading {variant} model ({model_info['size_mb']} MB)...")
    print(f"From: {model_info['url']}")
    print(f"To: {dest}")

    try:
        urllib.request.urlretrieve(model_info["url"], dest)
        print(f"Download complete: {dest}")
        return dest
    except Exception as e:
        # Clean up partial download
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {variant} model: {e}") from e


def list_models() -> dict:
    """Return status of all models."""
    weights_dir = get_weights_dir()
    result = {}
    for variant, info in MODELS.items():
        path = weights_dir / info["filename"]
        result[variant] = {
            "downloaded": path.exists(),
            "path": str(path),
            "size_mb": info["size_mb"],
        }
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: download_models.py <variant>")
        print(f"Available variants: {list(MODELS.keys())}")
        sys.exit(1)

    variant = sys.argv[1]
    try:
        path = download_model(variant)
        print(f"Model ready at: {path}")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
