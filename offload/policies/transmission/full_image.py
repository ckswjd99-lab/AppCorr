import numpy as np
import cv2
from typing import List, Generator
from offload.common.protocol import Patch, ExperimentConfig
from ..interface import ITransmissionPolicy

class FullImageCompressionPolicy(ITransmissionPolicy):
    """
    Compresses the entire image at once (JPEG/PNG).
    Sends data in the first patch (spatial_idx=0).
    Sends empty patches for the rest to satisfy Worker logic.
    Forces sequential processing but high compression efficiency.
    """

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        patches = []
        B, H, W, C = images.shape
        ph, pw = config.patch_size

        gh, gw = H // ph, W // pw
        num_patches = gh * gw
        
        fmt = config.transmission_kwargs.get('format', 'jpg').lower()
        quality = config.transmission_kwargs.get('quality', 95 if fmt == 'jpg' else 3)
        ext = '.jpg' if fmt == 'jpg' else '.png'
        
        encode_params = []
        if fmt == 'jpg':
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif fmt == 'png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, quality]
            
        all_patches = []
        for b in range(B):
            # Encode Full Image
            success, encoded_img = cv2.imencode(ext, images[b], encode_params)
            if not success:
                raise RuntimeError(f"Failed to encode full image {b} with format {fmt}")
            
            full_data = encoded_img.tobytes()
            
            # Create Patches
            # Patch 0 carries data
            all_patches.append(Patch(image_idx=b, spatial_idx=0, data=full_data))
            
            # Patches 1..N-1 carry empty data (metadata only)
            empty_data = b''
            for i in range(1, num_patches):
                all_patches.append(Patch(image_idx=b, spatial_idx=i, data=empty_data))
                
        yield all_patches

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None) -> np.ndarray:
        B = config.batch_size
        H, W, C = config.image_shape
        
        if canvas is None:
            batch_tensor = np.zeros((B, H, W, C), dtype=np.uint8)
        else:
            batch_tensor = canvas
        
        # Extract full image data from spatial_idx=0
        data_patches = {}
        
        for p in patches:
            if p.spatial_idx == 0 and len(p.data) > 0:
                 data_patches[p.image_idx] = p.data
        
        for b, data in data_patches.items():
            try:
                raw_bytes = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                
                if img is None:
                     raise ValueError(f"imdecode returned None for image {b}")
                
                if img.shape != (H, W, C):
                     # Best effort resize if dimension mismatch
                     if img.shape[:2] != (H, W):
                         img = cv2.resize(img, (W, H))
                         
                batch_tensor[b] = img
                
            except Exception as e:
                print(f"!!! [FullImageCompressionPolicy] Decode failed for image {b}: {e}")
                
        return batch_tensor
