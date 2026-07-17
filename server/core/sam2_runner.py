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
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

class SAM2ImageAnnotator:
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
        if frame_index is not None and frame_index == self._current_frame_index: return
        with torch.inference_mode(): self.predictor.set_image(image)
        self._current_frame_index = frame_index

    def predict(self, points: list, labels: list) -> Image:
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
        self._current_frame_index = None
        self.predictor.reset_predictor()

class _LazyFrameLoader:
    def __init__(self, img_paths: list, image_size: int):
        from sam2.utils.misc import _load_img_as_tensor
        self._load, self.img_paths, self.image_size = _load_img_as_tensor, img_paths, image_size
        self._M = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        self._S = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    def __getitem__(self, index: int) -> torch.Tensor:
        img, _, _ = self._load(self.img_paths[index], self.image_size)
        return (img.float() - self._M) / self._S
    def __len__(self) -> int: return len(self.img_paths)

class SAM2VideoRunner:
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
        """ ✅ 50-Frame Chunking with Hidden State Handoff """
        total_frames = len(frame_files)
        CHUNK_SIZE = 50
        seed_frame = sorted(points_dict.keys())[0]
        work_chunk_dir = frame_files[0].parent.parent / "_working_chunk"

        tracked_segments = {}

        def run_partition(start_idx, end_idx, direction="forward", mask_handoff=None):
            if work_chunk_dir.exists(): shutil.rmtree(work_chunk_dir)
            work_chunk_dir.mkdir()
            
            for i in range(start_idx, end_idx):
                shutil.copy(str(frame_files[i]), work_chunk_dir)
            
            p_state = self.predictor.init_state(video_path=str(work_chunk_dir), async_loading_frames=False)
            p_state["images"] = _LazyFrameLoader([str(work_chunk_dir / f"{i:08d}.jpg") for i in range(start_idx, end_idx)], self.predictor.image_size)
            p_state["num_frames"] = end_idx - start_idx
            
            # Apply clicks if they fall inside this partition
            for f_idx in points_dict:
                if start_idx <= f_idx < end_idx:
                    self.predictor.add_new_points_or_box(
                        p_state, f_idx - start_idx, 1, 
                        np.array(points_dict[f_idx], dtype=np.float32), 
                        np.array(labels_dict[f_idx], dtype=np.int32)
                    )
            
            # Apply hidden state handshake from previous partition
            if mask_handoff is not None:
                h_idx = 0 if direction == "forward" else (end_idx - start_idx - 1)
                self.predictor.add_new_mask(p_state, h_idx, 1, mask_handoff)

            # Propagate
            is_rev = (direction == "backward")
            for o_idx, _, o_logits in self.predictor.propagate_in_video(p_state, reverse=is_rev):
                torch.mps.synchronize()
                binary = (o_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                global_idx = start_idx + o_idx
                
                # Save cleanly to the masks_dir directly
                Image.fromarray(binary, "L").save(output_dir / f"mask_{global_idx:04d}.png")
                tracked_segments[global_idx] = True
                
                if on_progress: on_progress(len(tracked_segments), total_frames)

            # Get mask for handoff to the next iteration
            next_h = (o_logits[0, 0] > 0.0).cpu().numpy()
            
            self.predictor.reset_state(p_state)
            torch.mps.empty_cache()
            return next_h

        with torch.inference_mode():
            curr_h = None
            # Forward Pass
            for s in range(seed_frame, total_frames, CHUNK_SIZE):
                curr_h = run_partition(s, min(s + CHUNK_SIZE, total_frames), "forward", curr_h)
            
            # Backward Pass
            # Read the seed mask we just created to initiate backward tracking safely
            seed_mask_img = Image.open(output_dir / f"mask_{seed_frame:04d}.png")
            curr_h = (np.array(seed_mask_img) > 127)
            
            prev_s = seed_frame
            for s in range(seed_frame - CHUNK_SIZE, -CHUNK_SIZE, -CHUNK_SIZE):
                real_s = max(0, s)
                if real_s < prev_s:
                    curr_h = run_partition(real_s, prev_s, "backward", curr_h)
                    prev_s = real_s

        # Blank out untracked frames just in case
        blank = np.zeros((self.predictor.image_size, self.predictor.image_size), dtype=np.uint8)
        for i in range(total_frames):
            if i not in tracked_segments:
                Image.fromarray(blank, "L").save(output_dir / f"mask_{i:04d}.png")
        
        if work_chunk_dir.exists(): shutil.rmtree(work_chunk_dir)