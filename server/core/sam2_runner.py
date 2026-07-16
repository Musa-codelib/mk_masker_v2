import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, List
from utils.paths import resolve_sam2_checkpoint, resolve_sam2_config
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
        sam2 = build_sam2(config_name, str(checkpoint_path), device=self.device)
        self.predictor = SAM2ImagePredictor(sam2)
        self._current_frame_index = None

    def set_image(self, image: np.ndarray, frame_index: Optional[int] = None) -> None:
        if frame_index is not None and frame_index == self._current_frame_index: return
        with torch.inference_mode(): self.predictor.set_image(image)
        self._current_frame_index = frame_index

    def predict(self, points: list, labels: list) -> Image:
        with torch.inference_mode():
            torch.mps.synchronize()
            masks, _, _ = self.predictor.predict(point_coords=np.array(points, dtype=np.float32), point_labels=np.array(labels, dtype=np.int32), multimask_output=False)
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
        self.predictor = build_sam2_video_predictor(config_name, str(self.checkpoint_path), device=self.device)

    def run_bidirectional(self, frame_files, seed_idx, points, labels, output_dir, on_progress=None):
        """ ✅ v1.2 LOGIC: Single state, two directions, anchored to Hero Points """
        total = len(frame_files)
        with torch.inference_mode():
            # 1. Initialize ONE brain for the whole video
            state = self.predictor.init_state(video_path=str(frame_files[0].parent), async_loading_frames=False)
            state["images"] = _LazyFrameLoader([str(f) for f in frame_files], self.predictor.image_size)
            state["num_frames"] = total

            # 2. Anchor the brain to your manual selection (Hero Points)
            self.predictor.add_new_points_or_box(
                inference_state=state, frame_idx=seed_idx, obj_id=1,
                points=np.array(points, dtype=np.float32),
                labels=np.array(labels, dtype=np.int32)
            )

            # 3. PASS 1: Radiate Forward from Seed
            for f_idx, _, mask_logits in self.predictor.propagate_in_video(state, start_frame_idx=seed_idx):
                torch.mps.synchronize()
                binary = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                Image.fromarray(binary, "L").save(output_dir / f"mask_{f_idx:04d}.png")
                if on_progress: on_progress(f_idx, total)

            # 4. PASS 2: Radiate Backward from Seed
            for f_idx, _, mask_logits in self.predictor.propagate_in_video(state, start_frame_idx=seed_idx, reverse=True):
                torch.mps.synchronize()
                binary = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                Image.fromarray(binary, "L").save(output_dir / f"mask_{f_idx:04d}.png")
                if on_progress: on_progress(f_idx, total)

            self.predictor.reset_state(state); torch.mps.empty_cache()