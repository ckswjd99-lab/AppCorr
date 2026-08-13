"""VGGT-Omega input preprocessing, forked from upstream `vggt_omega/utils/load_fn.py`.

Forked rather than called because upstream takes *file paths* and this pipeline never has them: the
server receives frames that a transmission policy reconstructed in memory, possibly degraded. The
geometry below (aspect crop, patch-aligned target shape, common-size padding) is upstream's,
reproduced so the model sees exactly what it was trained to see.

Why it matters more than it looks: squashing a non-square frame into a square silently destroys pose
estimation while leaving depth apparently fine. Depth is relative structure and survives the
distortion; camera pose depends on the effective intrinsics, which an anisotropic resize invalidates.
Measured on Co3Dv2 with a square resize, depth came out at AbsRel 0.024 while median relative-rotation
error sat near 100 degrees -- and no ground-truth convention could rescue it, because the fault was
in the input, not the comparison.

The other reason to keep this separate: it defines two coordinate systems that must not be confused.
Frames arrive in *native* geometry, and the model consumes a *patch-aligned, padded* canvas. A
transmission policy degrades and scores patches in the first; the executor works in the second. That
is the same split that, when conflated, made COCO's approximation stop approximating.
"""

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

MIN_ASPECT, MAX_ASPECT = 0.5, 2.0


def _crop_to_supported_aspect_ratio(img: np.ndarray) -> np.ndarray:
    """Centre-crop only the extremes into [0.5, 2.0]; ordinary frames pass through untouched."""
    h, w = img.shape[:2]
    aspect = h / max(w, 1)
    if aspect < MIN_ASPECT:
        cw = min(w, max(1, int(round(h / MIN_ASPECT))))
        left = max((w - cw) // 2, 0)
        return img[:, left:left + cw]
    if aspect > MAX_ASPECT:
        ch = min(h, max(1, int(round(w * MAX_ASPECT))))
        top = max((h - ch) // 2, 0)
        return img[top:top + ch, :]
    return img


def _round_to_patch_multiple(value: float, patch_size: int) -> int:
    return max(patch_size, int(np.round(float(value) / patch_size)) * patch_size)


def _balanced_target_shape(aspect: float, resolution: int, patch_size: int) -> Tuple[int, int]:
    """Keep the token count near (resolution/patch)**2 while following the frame's own aspect."""
    tokens = (resolution // patch_size) ** 2
    w_p = max(1, int(np.round(np.sqrt(tokens / aspect))))
    h_p = max(1, int(np.round(tokens / w_p)))
    return h_p * patch_size, w_p * patch_size


def _max_size_target_shape(aspect: float, resolution: int, patch_size: int) -> Tuple[int, int]:
    if aspect >= 1.0:
        return resolution, _round_to_patch_multiple(resolution / aspect, patch_size)
    return _round_to_patch_multiple(resolution * aspect, patch_size), resolution


def preprocess_frames(
    frames: Sequence[np.ndarray],
    mode: str = "balanced",
    image_resolution: int = 512,
    patch_size: int = 16,
    device: torch.device | None = None,
) -> torch.Tensor:
    """`[S, H, W, C]` uint8/float frames (native, possibly ragged) -> `[1, S, 3, H', W']` float.

    Mirrors upstream's `load_and_preprocess_images` for in-memory input. Frames that end up at
    different shapes are zero-padded to a common size, as upstream does, because the aggregator
    needs one tensor.
    """
    if len(frames) == 0:
        raise ValueError("At least 1 frame is required")
    if mode not in ("balanced", "max_size"):
        raise ValueError("mode must be 'balanced' or 'max_size'")
    if image_resolution % patch_size:
        raise ValueError("image_resolution must be divisible by patch_size")

    import cv2

    out: List[torch.Tensor] = []
    for f in frames:
        img = np.asarray(f)
        if img.ndim != 3 or img.shape[2] != 3:
            raise RuntimeError(f"Expected an HWC RGB frame, got {img.shape}")
        img = _crop_to_supported_aspect_ratio(img)
        h, w = img.shape[:2]
        aspect = h / max(w, 1)
        th, tw = (
            _balanced_target_shape(aspect, image_resolution, patch_size)
            if mode == "balanced"
            else _max_size_target_shape(aspect, image_resolution, patch_size)
        )
        # INTER_AREA downscales without aliasing; upstream uses PIL BICUBIC, which matters little
        # next to getting the aspect ratio right.
        interp = cv2.INTER_AREA if (th < h or tw < w) else cv2.INTER_CUBIC
        img = cv2.resize(img, (tw, th), interpolation=interp)
        t = torch.from_numpy(np.ascontiguousarray(img))
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
        out.append(t.permute(2, 0, 1))

    shapes = {tuple(t.shape[1:]) for t in out}
    if len(shapes) > 1:
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        out = [F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1])) for t in out]

    stacked = torch.stack(out).unsqueeze(0).contiguous()
    return stacked.to(device) if device is not None else stacked
