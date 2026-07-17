import torch
import numpy as np
from PIL import Image
from pathlib import Path
from utils.errors import WeightsMissingError, EngineLoadError

class RVMRunner:
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
        """Processes BGR frame and returns a 0-255 uint8 alpha mask."""
        # Convert BGR to RGB and normalize to 0.0 - 1.0
        img = frame_np[:, :, ::-1] / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            # Standard downsample ratio for 1080p speed/quality balance
            ratio = torch.tensor([0.25]).to(self.device)
            
            # Flexible unpacking to handle any RVM TorchScript version
            if self.states is None:
                out = self.model(img, None, None, None, None, ratio)
            else:
                out = self.model(img, *self.states, ratio)
            
            # The last 4 elements are the recurrent states (r1, r2, r3, r4)
            self.states = out[-4:]
            
            # Alpha (pha) is always index 1 in the RVM outputs
            alpha_tensor = out[1]
            return (alpha_tensor.cpu().numpy()[0, 0] * 255).astype(np.uint8)