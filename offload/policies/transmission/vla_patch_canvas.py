"""
vla_patch_canvas.py

Transmission policy for VLA (OpenVLA) workloads: progressive patch-level canvas.

Encode (mobile side): group 0 = the full frame downsampled by `base_factor` (the low-res "approx"
base layer) + the task instruction text; groups 1..num_groups = true-resolution 14x14 pixel blocks,
ranked by per-patch residual RMS between the true frame and the upsampled base (highest-energy
residuals first, AppCorr's `residual_rms` mobile pscore), split evenly across groups. `coverage`
< 1.0 sends only the top fraction of patches.

Decode (server side): rebuilds the *cumulative canvas* -- upsampled base with all so-far-arrived
true patches pasted in. Faithful canvas semantics are REQUIRED for interleaved correction (the
approx continuation above the frontier consumes reconstructed stream rows -- see the semantics
audit in analysis/experiments/audit_progressive_semantics.py).
"""

from typing import Generator, List

import cv2
import numpy as np

from offload.common.protocol import ExperimentConfig, Patch

from ..interface import ITransmissionPolicy


class VLAPatchCanvasPolicy(ITransmissionPolicy):
    @staticmethod
    def _params(config: ExperimentConfig):
        tk = config.transmission_kwargs
        ph, pw = config.patch_size if not isinstance(config.patch_size, int) else (config.patch_size,) * 2
        return {
            "num_groups": max(int(tk.get("num_groups", 4)), 1),
            "coverage": float(tk.get("coverage", 1.0)),
            "base_factor": int(tk.get("base_factor", 4)),
            "ph": ph,
            "pw": pw,
        }

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = images[None]
        assert images.shape[0] == 1, "VLAPatchCanvas currently supports batch_size 1 (one control step at a time)"
        img = np.ascontiguousarray(images[0])  # [H, W, 3] uint8
        H, W, C = img.shape
        p = self._params(config)
        text = str(config.transmission_kwargs.get("text", ""))

        # Group 0: downsampled base + instruction text.
        base = cv2.resize(img, (W // p["base_factor"], H // p["base_factor"]), interpolation=cv2.INTER_AREA)
        # Note: target_shape is intentionally NOT set (worker.py wraps batch_np into dicts when any
        # patch carries target_shape); the base dims are derived from config + base_factor on decode.
        base_patch = Patch(
            image_idx=0, spatial_idx=-1, data=np.ascontiguousarray(base).tobytes(),
            res_level=p["base_factor"], group_id=0, batch_group_total=1, text=text,
        )
        yield [base_patch]

        # Residual-RMS ranking against the upsampled base (mobile-side signal; no server feedback
        # in the static schedule).
        up = cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        resid = (img.astype(np.float32) - up)
        gh, gw = H // p["ph"], W // p["pw"]
        per_patch = np.sqrt(
            (resid ** 2).reshape(gh, p["ph"], gw, p["pw"], C).mean(axis=(1, 3, 4))
        ).reshape(-1)
        order = np.argsort(-per_patch)

        num_send = max(1, int(round(len(order) * p["coverage"])))
        chosen = order[:num_send]
        splits = np.array_split(chosen, p["num_groups"])

        for gid, group in enumerate(splits, start=1):
            group_patches = []
            for sp_idx in group.tolist():
                r, c = divmod(sp_idx, gw)
                block = img[r * p["ph"]:(r + 1) * p["ph"], c * p["pw"]:(c + 1) * p["pw"]]
                group_patches.append(Patch(
                    image_idx=0, spatial_idx=int(sp_idx), data=np.ascontiguousarray(block).tobytes(),
                    res_level=0, group_id=gid, batch_group_total=len(group),
                    pscore_hint=float(per_patch[sp_idx]),
                ))
            if group_patches:
                yield group_patches

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None) -> np.ndarray:
        H, W, C = config.image_shape
        p = self._params(config)
        if canvas is None:
            canvas = np.zeros((1, H, W, C), dtype=np.uint8)

        gw = W // p["pw"]
        base_shape = (H // p["base_factor"], W // p["base_factor"], C)
        for patch in patches:
            if patch.group_id == 0:
                base = np.frombuffer(patch.data, dtype=np.uint8).reshape(base_shape)
                canvas[0] = cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                r, c = divmod(patch.spatial_idx, gw)
                block = np.frombuffer(patch.data, dtype=np.uint8).reshape(p["ph"], p["pw"], C)
                canvas[0, r * p["ph"]:(r + 1) * p["ph"], c * p["pw"]:(c + 1) * p["pw"]] = block
        return canvas
