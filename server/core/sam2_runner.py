"""SAM2 (Segment Anything 2) engine wrappers.

SAM2 is the *interactive* model: the user clicks points on a frame and SAM2 predicts
a mask there, then propagates that mask across the whole video. Two wrappers:

  * SAM2ImageAnnotator - single-frame click-to-mask (used for live UI previews)
  * SAM2VideoRunner    - bidirectional temporal propagation over all frames

Both lazily build the model on first use and report load failures via EngineLoadError.
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, List
import shutil
from utils.paths import resolve_sam2_checkpoint, resolve_sam2_config
from utils.errors import EngineLoadError
from hydra import compose, initialize_config_dir

def _pick_device() -> torch.device:
    # Prefer CUDA (NVIDIA), then Apple Silicon's MPS, then fall back to CPU.
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

class SAM2ImageAnnotator:
    """Interactive per-frame mask predictor driven by user clicks."""
    def __init__(self, variant: str = "small"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.device = _pick_device()
        checkpoint_path = resolve_sam2_checkpoint(variant)
        config_dir = os.path.dirname(os.path.abspath(__file__))
        config_name = "sam2_hiera_s.yaml"
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized(): GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=config_dir, version_base=None)
        try:
            sam2 = build_sam2(config_name, str(checkpoint_path), device=self.device)
        except Exception as e:
            raise EngineLoadError(f"SAM2 model failed to load ({variant}).", detail=str(e))
        self.predictor = SAM2ImagePredictor(sam2)
        self._current_frame_index = None

    def set_image(self, image: np.ndarray, frame_index: Optional[int] = None) -> None:
        # SAM2 re-encodes the image on every set_image call, which is expensive, so we
        # skip it when the same frame is requested again.
        if frame_index is not None and frame_index == self._current_frame_index: return
        with torch.inference_mode(): self.predictor.set_image(image)
        self._current_frame_index = frame_index

    def predict(self, points: list, labels: list) -> Image:
        # points: [x, y] click coordinates; labels: 1 = foreground, 0 = background.
        # Returns a single binary (0/255) mask as a PIL "L" image.
        with torch.inference_mode():
            torch.mps.synchronize()
            masks, _, _ = self.predictor.predict(
                point_coords=np.array(points, dtype=np.float32), 
                point_labels=np.array(labels, dtype=np.int32), 
                multimask_output=False
            )
            torch.mps.synchronize()
        return Image.fromarray((masks[0] > 0).astype(np.uint8) * 255, mode="L")

    def clear_cache(self):
        # Drop the cached frame embedding so the next set_image starts fresh.
        self._current_frame_index = None
        self.predictor.reset_predictor()

class _LazyFrameLoader:
    """Loads + normalises frames on demand for SAM2's video state (ImageNet stats)."""
    def __init__(self, img_paths: list, image_size: int):
        from sam2.utils.misc import _load_img_as_tensor
        self._load, self.img_paths, self.image_size = _load_img_as_tensor, img_paths, image_size
        # Mean/std used to normalise inputs the way SAM2 was trained.
        self._M = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        self._S = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    def __getitem__(self, index: int) -> torch.Tensor:
        img, _, _ = self._load(self.img_paths[index], self.image_size)
        return (img.float() - self._M) / self._S
    def __len__(self) -> int: return len(self.img_paths)

class SAM2VideoRunner:
    """Temporal mask propagation over a whole clip from a few user clicks."""
    def __init__(self, variant: str = "small"):
        from sam2.build_sam import build_sam2_video_predictor
        self.device = _pick_device()
        self.checkpoint_path = resolve_sam2_checkpoint(variant)
        config_dir = os.path.dirname(os.path.abspath(__file__))
        config_name = "sam2_hiera_s.yaml"
        try:
            self.predictor = build_sam2_video_predictor(config_name, str(self.checkpoint_path), device=self.device)
        except Exception as e:
            raise EngineLoadError(f"SAM2 video model failed to load ({variant}).", detail=str(e))

    def run_bidirectional(self, frame_files, points_dict, labels_dict, output_dir, on_progress=None):
        """Propagate masks across the video in 50-frame chunks, both directions.

        To keep memory bounded, the clip is split into CHUNK_SIZE windows. Each window
        is tracked forward, then its ending mask is handed off as a seed to the next
        window so tracking stays continuous. After the forward pass we repeat backward
        from the seed click for stable coverage on both sides.
        """
        total_frames = len(frame_files)
        CHUNK_SIZE = 50
        seed_frame = sorted(points_dict.keys())[0]
        work_chunk_dir = frame_files[0].parent.parent / "_working_chunk"

        tracked_segments = {}

        def run_partition(start_idx, end_idx, direction="forward", mask_handoff=None):
            # Copy just this window's frames into a temp folder for SAM2 to read.
            if work_chunk_dir.exists(): shutil.rmtree(work_chunk_dir)
            work_chunk_dir.mkdir()
            
            for i in range(start_idx, end_idx):
                shutil.copy(str(frame_files[i]), work_chunk_dir)
            
            p_state = self.predictor.init_state(video_path=str(work_chunk_dir), async_loading_frames=False)
            p_state["images"] = _LazyFrameLoader([str(work_chunk_dir / f"{i:08d}.jpg") for i in range(start_idx, end_idx)], self.predictor.image_size)
            p_state["num_frames"] = end_idx - start_idx
            
            # Replay any user clicks that fall inside this window.
            for f_idx in points_dict:
                if start_idx <= f_idx < end_idx:
                    self.predictor.add_new_points_or_box(
                        p_state, f_idx - start_idx, 1, 
                        np.array(points_dict[f_idx], dtype=np.float32), 
                        np.array(labels_dict[f_idx], dtype=np.int32)
                    )
            
            # Seed the window with the mask handed over from the previous window.
            if mask_handoff is not None:
                h_idx = 0 if direction == "forward" else (end_idx - start_idx - 1)
                self.predictor.add_new_mask(p_state, h_idx, 1, mask_handoff)

            # Propagate the mask across the window (forward or backward).
            is_rev = (direction == "backward")
            for o_idx, _, o_logits in self.predictor.propagate_in_video(p_state, reverse=is_rev):
                torch.mps.synchronize()
                binary = (o_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                global_idx = start_idx + o_idx
                
                # Write each mask straight to the output masks_dir.
                Image.fromarray(binary, "L").save(output_dir / f"mask_{global_idx:04d}.png")
                tracked_segments[global_idx] = True
                
                if on_progress: on_progress(len(tracked_segments), total_frames)

            # Keep the final mask to hand off to the next window.
            next_h = (o_logits[0, 0] > 0.0).cpu().numpy()
            
            self.predictor.reset_state(p_state)
            torch.mps.empty_cache()
            return next_h

        with torch.inference_mode():
            curr_h = None
            # Forward Pass: from the seed click to the end of the clip.
            for s in range(seed_frame, total_frames, CHUNK_SIZE):
                curr_h = run_partition(s, min(s + CHUNK_SIZE, total_frames), "forward", curr_h)
            
            # Backward Pass: from the seed click back to frame 0 for full coverage.
            seed_mask_img = Image.open(output_dir / f"mask_{seed_frame:04d}.png")
            curr_h = (np.array(seed_mask_img) > 127)
            
            prev_s = seed_frame
            for s in range(seed_frame - CHUNK_SIZE, -CHUNK_SIZE, -CHUNK_SIZE):
                real_s = max(0, s)
                if real_s < prev_s:
                    curr_h = run_partition(real_s, prev_s, "backward", curr_h)
                    prev_s = real_s

        # Any frame not covered by propagation gets a fully-transparent (blank) mask.
        blank = np.zeros((self.predictor.image_size, self.predictor.image_size), dtype=np.uint8)
        for i in range(total_frames):
            if i not in tracked_segments:
                Image.fromarray(blank, "L").save(output_dir / f"mask_{i:04d}.png")
        
        if work_chunk_dir.exists(): shutil.rmtree(work_chunk_dir)