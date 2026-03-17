import numpy as np
import zlib
from typing import List, Generator
from concurrent.futures import ThreadPoolExecutor
from offload.common.protocol import Patch, ExperimentConfig
from ..interface import ITransmissionPolicy

class ZlibTransmissionPolicy(ITransmissionPolicy):
    """
    Lossless Coding Policy.
    Compresses each patch using zlib (DEFLATE = LZ77 + Huffman).
    """

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        """Encode images into zlib compressed patches."""
        patches = []
        B, H, W, C = images.shape
        ph, pw = config.patch_size
        gh, gw = H // ph, W // pw

        compression_level = config.transmission_kwargs.get('compression_level', 1)
        
        # Vectorized Patchify (Same as Raw)
        reshaped = images.reshape(B, gh, ph, gw, pw, C)
        transposed = reshaped.transpose(0, 1, 3, 2, 4, 5)
        patch_tensor = transposed.reshape(B, gh * gw, ph, pw, C)
        
        num_patches = gh * gw
        
        all_patches = []
        # Compress and Wrap
        for b in range(B):
            for i in range(num_patches):
                raw_bytes = patch_tensor[b, i].tobytes()
                compressed_data = zlib.compress(raw_bytes, level=compression_level)
                all_patches.append(Patch(image_idx=b, spatial_idx=i, data=compressed_data))
                
        yield all_patches

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None) -> np.ndarray:
        """Decode zlib compressed patches."""
        B = config.batch_size
        H, W, C = config.image_shape
        ph, pw = config.patch_size
        gw = W // pw
        
        if canvas is None:
            batch_tensor = np.zeros((B, H, W, C), dtype=np.uint8)
        else:
            batch_tensor = canvas
        
        def _decompress_patch(p):
            try:
                raw_bytes = zlib.decompress(p.data)
                chunk = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(ph, pw, C)
                r, c = divmod(p.spatial_idx, gw)
                y, x = r * ph, c * pw
                batch_tensor[p.image_idx, y:y+ph, x:x+pw] = chunk
            except Exception as e:
                print(f"!!! [ZlibPolicy] Decompression failed for patch {p.spatial_idx}: {e}")

        with ThreadPoolExecutor() as executor:
            executor.map(_decompress_patch, patches)
            
        return batch_tensor
