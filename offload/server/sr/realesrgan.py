import math
import os
from contextlib import nullcontext
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_REALESRGAN_MODELS = {
    'realesrgan_x4plus': {
        'scale': 4,
        'weights': 'RealESRGAN_x4plus.pth',
        'arch': 'rrdb',
        'arch_kwargs': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 23,
            'num_grow_ch': 32,
            'scale': 4,
        },
    },
    'realesrnet_x4plus': {
        'scale': 4,
        'weights': 'RealESRNet_x4plus.pth',
        'arch': 'rrdb',
        'arch_kwargs': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 23,
            'num_grow_ch': 32,
            'scale': 4,
        },
    },
    'realesrgan_x4plus_anime_6b': {
        'scale': 4,
        'weights': 'RealESRGAN_x4plus_anime_6B.pth',
        'arch': 'rrdb',
        'arch_kwargs': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 6,
            'num_grow_ch': 32,
            'scale': 4,
        },
    },
    'realesrgan_x2plus': {
        'scale': 2,
        'weights': 'RealESRGAN_x2plus.pth',
        'arch': 'rrdb',
        'arch_kwargs': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 23,
            'num_grow_ch': 32,
            'scale': 2,
        },
    },
    'realesr_animevideov3': {
        'scale': 4,
        'weights': 'realesr-animevideov3.pth',
        'arch': 'srvgg',
        'arch_kwargs': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_conv': 16,
            'upscale': 4,
            'act_type': 'prelu',
        },
    },
    'realesr_general_x4v3': {
        'scale': 4,
        'weights': 'realesr-general-x4v3.pth',
        'arch': 'srvgg',
        'arch_kwargs': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_conv': 32,
            'upscale': 4,
            'act_type': 'prelu',
        },
    },
}


def _load_checkpoint(path: str) -> Dict[str, Any]:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Real-ESRGAN weight file not found: {path}")

    print(f"[SR] Loading weights from {path} with mmap=True...")
    return torch.load(path, map_location='cpu', mmap=True)


class RealESRGANUpscaler:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.model_name = str(config.get('model', 'realesrgan_x4plus')).lower()
        if self.model_name not in _REALESRGAN_MODELS:
            supported = ", ".join(sorted(_REALESRGAN_MODELS))
            raise ValueError(f"Unsupported Real-ESRGAN model: {self.model_name}. Supported: {supported}")

        self.spec = _REALESRGAN_MODELS[self.model_name]
        self.weights_dir = os.path.expanduser(config.get('weights_dir', '~/cjpark/weights/realesrgan'))
        self.tile = int(config.get('tile', 0))
        self.tile_pad = int(config.get('tile_pad', 10))
        self.pre_pad = int(config.get('pre_pad', 0))
        self.scale = int(self.spec['scale'])
        self.mod_scale = 2 if self.scale == 2 else None
        self.dtype_name = str(config.get('dtype', 'fp16')).lower()
        self.dtype = self._resolve_dtype()
        self.model = None

        self._load()

    def _load(self):
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        except ImportError as exc:
            raise ImportError(
                "realesrgan is required for lowres_sr. Install dependencies from requirements.txt."
            ) from exc

        if self.spec['arch'] == 'rrdb':
            model = RRDBNet(**self.spec['arch_kwargs'])
        else:
            model = SRVGGNetCompact(**self.spec['arch_kwargs'])

        model_path = os.path.join(self.weights_dir, self.spec['weights'])
        state_dict = _load_checkpoint(model_path)
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device=self.device, dtype=self.dtype)
        self.model = model

    @torch.no_grad()
    def upscale_tensor(self, batch_np: np.ndarray, target_hw: tuple[int, int]) -> torch.Tensor:
        if batch_np.ndim != 4 or batch_np.shape[-1] != 3:
            raise ValueError(f"Expected batch image tensor [B, H, W, 3], got {batch_np.shape}")

        input_tensor, meta = self._preprocess(batch_np)
        with self._autocast_context():
            output_tensor = self._forward(input_tensor)
        output_tensor = self._postprocess(output_tensor, meta)
        output_tensor = output_tensor[:, [2, 1, 0], :, :].contiguous()

        target_h, target_w = target_hw
        if output_tensor.shape[-2:] != (target_h, target_w):
            output_tensor = F.interpolate(
                output_tensor,
                size=(target_h, target_w),
                mode='bicubic',
                align_corners=False,
            )
        return output_tensor.clamp_(0, 1)

    @torch.no_grad()
    def upscale(self, batch_np: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        output_tensor = self.upscale_tensor(batch_np, target_hw)
        output_np = output_tensor.float().cpu().numpy()
        output_np = np.transpose(output_np, (0, 2, 3, 1))
        output_np = np.clip(np.rint(output_np * 255.0), 0, 255).astype(np.uint8)
        return output_np

    def _preprocess(self, batch_np: np.ndarray) -> tuple[torch.Tensor, Dict[str, int]]:
        batch_np = batch_np.astype(np.float32) / 255.0
        # Mirror RealESRGANer.enhance() channel handling so output stays in the repo's RGB convention.
        batch_np = batch_np[..., ::-1].copy()

        tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(device=self.device, dtype=self.dtype)

        if self.pre_pad != 0:
            tensor = F.pad(tensor, (0, self.pre_pad, 0, self.pre_pad), mode='reflect')

        mod_pad_h = 0
        mod_pad_w = 0
        if self.mod_scale is not None:
            _, _, h, w = tensor.shape
            mod_pad_h = (self.mod_scale - h % self.mod_scale) % self.mod_scale
            mod_pad_w = (self.mod_scale - w % self.mod_scale) % self.mod_scale
            if mod_pad_h != 0 or mod_pad_w != 0:
                tensor = F.pad(tensor, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')

        return tensor, {'mod_pad_h': mod_pad_h, 'mod_pad_w': mod_pad_w}

    def _resolve_dtype(self) -> torch.dtype:
        if self.device.type != 'cuda':
            return torch.float32

        dtype_map = {
            'fp16': torch.float16,
            'float16': torch.float16,
            'half': torch.float16,
            'bf16': torch.bfloat16,
            'bfloat16': torch.bfloat16,
            'fp32': torch.float32,
            'float32': torch.float32,
        }
        if self.dtype_name not in dtype_map:
            supported = ", ".join(sorted(dtype_map))
            raise ValueError(f"Unsupported lowres_sr_dtype: {self.dtype_name}. Supported: {supported}")
        return dtype_map[self.dtype_name]

    def _autocast_context(self):
        if self.device.type != 'cuda' or self.dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=self.dtype)

    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.tile > 0:
            return self._tile_forward(input_tensor)
        return self.model(input_tensor)

    def _tile_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = input_tensor.shape
        output = input_tensor.new_zeros((batch, channel, height * self.scale, width * self.scale))

        tiles_x = math.ceil(width / self.tile)
        tiles_y = math.ceil(height / self.tile)

        for y_idx in range(tiles_y):
            for x_idx in range(tiles_x):
                input_start_x = x_idx * self.tile
                input_end_x = min(input_start_x + self.tile, width)
                input_start_y = y_idx * self.tile
                input_end_y = min(input_start_y + self.tile, height)

                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                input_tile = input_tensor[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]
                output_tile = self.model(input_tile)

                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + (input_end_x - input_start_x) * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + (input_end_y - input_start_y) * self.scale

                output[
                    :,
                    :,
                    output_start_y:output_end_y,
                    output_start_x:output_end_x,
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]

        return output

    def _postprocess(self, output_tensor: torch.Tensor, meta: Dict[str, int]) -> torch.Tensor:
        mod_pad_h = meta['mod_pad_h']
        mod_pad_w = meta['mod_pad_w']
        if mod_pad_h != 0 or mod_pad_w != 0:
            _, _, h, w = output_tensor.shape
            output_tensor = output_tensor[
                :,
                :,
                0:h - mod_pad_h * self.scale,
                0:w - mod_pad_w * self.scale,
            ]

        if self.pre_pad != 0:
            _, _, h, w = output_tensor.shape
            output_tensor = output_tensor[
                :,
                :,
                0:h - self.pre_pad * self.scale,
                0:w - self.pre_pad * self.scale,
            ]

        return output_tensor


def create_lowres_sr_engine(config: Dict[str, Any], device: torch.device) -> RealESRGANUpscaler:
    return RealESRGANUpscaler(config, device)
