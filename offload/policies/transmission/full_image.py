import numpy as np
import cv2
from typing import Any, List, Generator
from offload.common.protocol import Patch, ExperimentConfig
from ..interface import ITransmissionPolicy

class FullImageCompressionPolicy(ITransmissionPolicy):
    """
    Compresses the entire image at once (JPEG/PNG).
    Sends data in the first patch (spatial_idx=0).
    Sends empty patches for the rest to satisfy Worker logic unless preserving input shape.
    Forces sequential processing but high compression efficiency.
    """
    _PRESERVE_HEADER_MAGIC = b'FICP'
    _PRESERVE_HEADER_SIZE = len(_PRESERVE_HEADER_MAGIC) + 2 * np.dtype(np.int32).itemsize

    @staticmethod
    def _split_preserve_item(item):
        metadata = {}
        image = item
        if isinstance(item, dict):
            metadata = dict(item.get('metadata') or {})
            image = item.get('image')
        if image is None:
            raise RuntimeError("FullImageCompression received an empty image slot.")
        return image, metadata

    @classmethod
    def _pack_preserve_payload(cls, encoded_img: np.ndarray, metadata: dict) -> bytes:
        target_shape = metadata.get('target_shape') or (0, 0)
        target_h, target_w = int(target_shape[0]), int(target_shape[1])
        header = np.array([target_h, target_w], dtype=np.int32).tobytes()
        return cls._PRESERVE_HEADER_MAGIC + header + encoded_img.tobytes()

    @classmethod
    def _unpack_preserve_payload(cls, data: bytes):
        target_shape = None
        image_data = data
        if len(data) >= cls._PRESERVE_HEADER_SIZE and data[:len(cls._PRESERVE_HEADER_MAGIC)] == cls._PRESERVE_HEADER_MAGIC:
            start = len(cls._PRESERVE_HEADER_MAGIC)
            stop = cls._PRESERVE_HEADER_SIZE
            target_h, target_w = np.frombuffer(data[start:stop], dtype=np.int32).tolist()
            if target_h > 0 and target_w > 0:
                target_shape = (int(target_h), int(target_w))
            image_data = data[stop:]
        return image_data, target_shape

    def encode(self, images: Any, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        preserve_input_shape = bool(config.transmission_kwargs.get('preserve_input_shape', False))
        if preserve_input_shape:
            if isinstance(images, np.ndarray):
                image_items = [(images[b], {}) for b in range(images.shape[0])]
            else:
                image_items = [self._split_preserve_item(item) for item in images]
            C = config.image_shape[2]
            num_patches = 1
        else:
            B, H, W, C = images.shape
            image_items = [(images[b], {}) for b in range(B)]
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
        for b, (image, metadata) in enumerate(image_items):
            if image.ndim != 3 or image.shape[2] != C:
                raise RuntimeError(f"Expected HWC image with {C} channels, got {image.shape}")
            # Encode Full Image
            success, encoded_img = cv2.imencode(ext, np.ascontiguousarray(image), encode_params)
            if not success:
                raise RuntimeError(f"Failed to encode full image {b} with format {fmt}")
            
            if preserve_input_shape:
                full_data = self._pack_preserve_payload(encoded_img, metadata)
            else:
                full_data = encoded_img.tobytes()
            
            # Create Patches
            # Patch 0 carries data
            all_patches.append(Patch(image_idx=b, spatial_idx=0, data=full_data))
            
            # Patches 1..N-1 carry empty data (metadata only)
            empty_data = b''
            for i in range(1, num_patches):
                all_patches.append(Patch(image_idx=b, spatial_idx=i, data=empty_data))
                
        yield all_patches

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None):
        B = config.batch_size
        H, W, C = config.image_shape
        preserve_input_shape = bool(config.transmission_kwargs.get('preserve_input_shape', False))
        if preserve_input_shape:
            batch_images = list(canvas) if isinstance(canvas, list) else [None] * B
            data_patches = {}
            for p in patches:
                if p.spatial_idx == 0 and len(p.data) > 0:
                    data_patches[p.image_idx] = p.data

            for b, data in data_patches.items():
                try:
                    image_data, target_shape = self._unpack_preserve_payload(data)
                    raw_bytes = np.frombuffer(image_data, dtype=np.uint8)
                    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError(f"imdecode returned None for image {b}")
                    item = {'image': np.ascontiguousarray(img)}
                    if target_shape is not None:
                        item['target_shape'] = target_shape
                    batch_images[b] = item
                except Exception as e:
                    print(f"!!! [FullImageCompressionPolicy] Decode failed for image {b}: {e}")

            return batch_images
        
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
