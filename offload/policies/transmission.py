import numpy as np
from typing import List
from offload.common.protocol import Patch, ExperimentConfig
from .interface import ITransmissionPolicy

class RawTransmissionPolicy(ITransmissionPolicy):
    """Encodes/Decodes patches with Batch dimension support."""

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> List[Patch]:
        """
        Input: images (B, H, W, C)
        Output: Flattened list of patches with (image_idx, spatial_idx).
        """
        patches = []
        B, H, W, C = images.shape
        ph, pw = config.patch_size
        
        grid_h = H // ph
        grid_w = W // pw
        
        for b in range(B):
            spatial_idx = 0
            for r in range(grid_h):
                for c in range(grid_w):
                    y1, y2 = r * ph, (r + 1) * ph
                    x1, x2 = c * pw, (c + 1) * pw
                    
                    crop = images[b, y1:y2, x1:x2]
                    
                    patch = Patch(
                        image_idx=b,
                        spatial_idx=spatial_idx,
                        data=crop.tobytes()
                    )
                    patches.append(patch)
                    spatial_idx += 1
        return patches

    def decode(self, patches: List[Patch], config: ExperimentConfig) -> np.ndarray:
        """
        Input: List of Patch objects (can be sparse/subset).
        Output: Reconstructed Tensor (B, H, W, C). Missing patches are zeros.
        """
        B = config.batch_size
        H, W, C = config.image_shape
        ph, pw = config.patch_size
        
        # Initialize Batch Tensor with Zeros
        batch_tensor = np.zeros((B, H, W, C), dtype=np.uint8)
        
        grid_w = W // pw
        
        for p in patches:
            b = p.image_idx
            # Calculate spatial position
            r = p.spatial_idx // grid_w
            c = p.spatial_idx % grid_w
            
            y1, y2 = r * ph, (r + 1) * ph
            x1, x2 = c * pw, (c + 1) * pw
            
            # Restore data
            chunk = np.frombuffer(p.data, dtype=np.uint8).reshape(ph, pw, C)
            batch_tensor[b, y1:y2, x1:x2] = chunk
            
        return batch_tensor