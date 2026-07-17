import torch
import numpy as np
from PIL import Image
from pathlib import Path
from utils.errors import WeightsMissingError, EngineLoadError

class RVMRunner:
    """Robust Video Matting (RVM) engine.

    Unlike SAM2, RVM is fully automatic: it mattes a human subject from every frame
    with no user clicks. It keeps recurrent (temporal) state between frames so the
    result is temporally stable. Each call to process_frame returns a single-channel
    0-255 alpha mask.
    """

    def __init__(self, model_path: Path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        if not model_path.exists():
            raise WeightsMissingError("RVM weights missing.", detail=str(model_path))

        try:
            # Load the TorchScript model
            self.model = torch.jit.load(str(model_path), map_location='cpu').to(self.device).eval()
            print(f"✅ RVM Engine Loaded on {self.device}")
        except Exception as e:
            raise EngineLoadError("RVM model failed to load.", detail=str(e))
            
        self.states = None 

    def reset(self):
        """Wipe recurrent memory before starting a new video."""
        self.states = None

    def process_frame(self, frame_np):
        """Process a single BGR frame and return a 0-255 uint8 alpha mask."""
        # Convert BGR -> RGB and scale pixel values to the 0.0 - 1.0 range the model expects.
        img = frame_np[:, :, ::-1] / 255.0
        # Rearrange to CHW and add a batch dimension for PyTorch.
        img = torch.from_numpy(img).permute(2, 0, 1).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            # Downsample to 1/4 resolution: good speed/quality trade-off for 1080p.
            ratio = torch.tensor([0.25]).to(self.device)
            
            # First frame has no recurrent state; later frames reuse the previous states.
            if self.states is None:
                out = self.model(img, None, None, None, None, ratio)
            else:
                out = self.model(img, *self.states, ratio)
            
            # Keep the last 4 outputs as the recurrent state for the next frame.
            self.states = out[-4:]
            
            # The alpha (matte) channel is always the 2nd output of RVM.
            alpha_tensor = out[1]
            return (alpha_tensor.cpu().numpy()[0, 0] * 255).astype(np.uint8)