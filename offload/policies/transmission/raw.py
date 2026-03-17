import numpy as np
from typing import List, Generator
from offload.common.protocol import Patch, ExperimentConfig
from ..interface import ITransmissionPolicy

class RawTransmissionPolicy(ITransmissionPolicy):
    """Encodes/Decodes patches with Batch dimension support using vectorized ops."""

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        """Encode images to patches using vectorized operations."""
        patches = []
        B, H, W, C = images.shape
        ph, pw = config.patch_size
        
        # Grid dimensions
        gh, gw = H // ph, W // pw
        
        reshaped = images.reshape(B, gh, ph, gw, pw, C)
        transposed = reshaped.transpose(0, 1, 3, 2, 4, 5)
        patch_tensor = transposed.reshape(B, gh * gw, ph, pw, C)
        
        num_patches = gh * gw
        for b in range(B):
            for i in range(num_patches):
                data = patch_tensor[b, i].tobytes()
                patches.append(Patch(image_idx=b, spatial_idx=i, data=data))
                
        yield patches

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None) -> np.ndarray:
        """Decode patches into image canvas."""
        B = config.batch_size
        H, W, C = config.image_shape
        ph, pw = config.patch_size
        gw = W // pw
        
        # Initialize canvas if not provided
        if canvas is None:
            batch_tensor = np.zeros((B, H, W, C), dtype=np.uint8)
        else:
            batch_tensor = canvas
        
        for p in patches:
            # Calculate grid coordinates
            r, c = divmod(p.spatial_idx, gw)
            y, x = r * ph, c * pw
            
            # Restore patch array
            chunk = np.frombuffer(p.data, dtype=np.uint8).reshape(ph, pw, C)
            
            # Place on canvas
            batch_tensor[p.image_idx, y:y+ph, x:x+pw] = chunk
            
        return batch_tensor
