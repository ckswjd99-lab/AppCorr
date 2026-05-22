import numpy as np
import cv2
import math
from typing import Any, List, Generator
from offload.common.protocol import Patch, ExperimentConfig
from ..interface import ITransmissionPolicy

class RawTransmissionPolicy(ITransmissionPolicy):
    """Encodes/Decodes patches with Batch dimension support using vectorized ops."""

    @staticmethod
    def _short_side(config: ExperimentConfig) -> int:
        profile_config = config.get_input_profile_config()
        return int(profile_config.get('mobile_resize_short_side', min(config.image_shape[:2])))

    @staticmethod
    def _align_up(value: float, multiple: int) -> int:
        return int(math.ceil(float(value) / multiple)) * multiple

    @classmethod
    def _resize_short_side(cls, image: np.ndarray, short_side: int, patch_size: tuple[int, int] | list[int]) -> np.ndarray:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise RuntimeError(f"Cannot resize image with shape {image.shape}")
        ph, pw = patch_size
        if h <= w:
            target_h = cls._align_up(short_side, ph)
            target_w = cls._align_up(w / h * target_h, pw)
        else:
            target_w = cls._align_up(short_side, pw)
            target_h = cls._align_up(h / w * target_w, ph)
        if (h, w) == (target_h, target_w):
            return np.ascontiguousarray(image)
        return np.ascontiguousarray(cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR))

    def encode(self, images: Any, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        """Encode images to patches using vectorized operations."""
        preserve_input_shape = bool(config.transmission_kwargs.get('preserve_input_shape', False))
        if preserve_input_shape:
            image_list = [images[b] for b in range(images.shape[0])] if isinstance(images, np.ndarray) else list(images)
            patches = []
            for b, item in enumerate(image_list):
                metadata = {}
                image = item
                if isinstance(item, dict):
                    metadata = dict(item.get('metadata') or {})
                    image = item.get('image')
                if image.ndim != 3:
                    raise RuntimeError(f"Expected HWC image, got {image.shape}")
                image = self._resize_short_side(image, self._short_side(config), config.patch_size)
                target_shape = metadata.get('target_shape') or (0, 0)
                header = np.array(
                    [image.shape[0], image.shape[1], image.shape[2], target_shape[0], target_shape[1]],
                    dtype=np.int32,
                ).tobytes()
                patches.append(Patch(image_idx=b, spatial_idx=0, data=header + image.tobytes()))
            yield patches
            return

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

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None):
        """Decode patches into image canvas."""
        B = config.batch_size
        H, W, C = config.image_shape
        preserve_input_shape = bool(config.transmission_kwargs.get('preserve_input_shape', False))
        if preserve_input_shape:
            batch_images = list(canvas) if isinstance(canvas, list) else [None] * B
            for p in patches:
                if p.spatial_idx != 0 or len(p.data) == 0:
                    continue
                try:
                    header = np.frombuffer(p.data[:20], dtype=np.int32).tolist()
                    shape = tuple(header[:3])
                    target_shape = tuple(header[3:5])
                    image = np.frombuffer(p.data[20:], dtype=np.uint8).reshape(shape)
                    item = {'image': np.ascontiguousarray(image)}
                    if target_shape[0] > 0 and target_shape[1] > 0:
                        item['target_shape'] = target_shape
                    batch_images[p.image_idx] = item
                except Exception as e:
                    print(f"!!! [RawPolicy] Decode failed for image {p.image_idx}: {e}")
            return batch_images

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
