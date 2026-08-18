"""Round-trip probe: does the closed-loop transmission fix change anything at CLIP's geometry?

CLIP-bigG configs are 224x224 with patch_size 14 and pyramid_levels [2, 0]. Transmit *every*
group, decode, and compare against the source image. A lossless scheme must give 0.00% relative
L2. Runs the same probe with the pre-fix (open-loop) residual monkeypatched back in, so the two
columns are directly comparable.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np

from offload.common import ExperimentConfig
from offload.policies.transmission.progressive import ProgressiveLPyramidPolicy


def open_loop_residual(self, gaussians, prev_lvl, tgt_lvl, config, image_hw):
    """The pre-378e21d form: predict from the native gaussian, then project the residual."""
    pred = self._iterative_upsample_native(gaussians[prev_lvl], prev_lvl, tgt_lvl, gaussians)
    residual = gaussians[tgt_lvl].astype(np.int16) - pred.astype(np.int16)
    return self._project_band_to_target(residual, tgt_lvl, config, np.int16, image_hw)


def round_trip(cfg, images):
    policy = ProgressiveLPyramidPolicy()
    patches = []
    for group in policy.encode(images, cfg):
        patches.extend(group)
    return policy.decode(patches, cfg), len(patches)


def rel_l2(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return 100.0 * np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)


def main():
    cfg_path = sys.argv[1]
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["batch_size"] = 2
    raw["device"] = "cpu"
    cfg = ExperimentConfig(**raw)

    H, W, C = cfg.image_shape
    rng = np.random.default_rng(0)
    # Structured content, not white noise: a lossy predictor mismatch shows up on edges.
    yy, xx = np.mgrid[0:H, 0:W]
    base = (127 + 100 * np.sin(xx / 9.0) * np.cos(yy / 13.0)).astype(np.float32)
    img0 = np.clip(base[..., None] + rng.normal(0, 12, (H, W, C)), 0, 255).astype(np.uint8)
    img1 = np.clip(np.roll(base, 37, axis=1)[..., None] + rng.normal(0, 30, (H, W, C)), 0, 255).astype(np.uint8)
    images = np.stack([img0, img1])

    print(f"config      : {cfg_path}")
    print(f"image_shape : {cfg.image_shape}  patch_size: {cfg.patch_size}")
    print(f"levels      : {cfg.transmission_kwargs.get('pyramid_levels')}")

    fixed, n_fixed = round_trip(cfg, images)

    orig = ProgressiveLPyramidPolicy._closed_loop_residual
    ProgressiveLPyramidPolicy._closed_loop_residual = open_loop_residual
    try:
        old, n_old = round_trip(cfg, images)
    finally:
        ProgressiveLPyramidPolicy._closed_loop_residual = orig

    for i in range(len(images)):
        print(f"  image {i}: closed-loop relL2 = {rel_l2(fixed[i], images[i]):.4f}%   "
              f"open-loop relL2 = {rel_l2(old[i], images[i]):.4f}%")
    print(f"  patches: closed-loop {n_fixed}, open-loop {n_old}")
    ident = np.array_equal(np.asarray(fixed), np.asarray(old))
    print(f"  decoded images bit-identical between the two encoders: {ident}")
    if not ident:
        d = np.abs(np.asarray(fixed).astype(np.int32) - np.asarray(old).astype(np.int32))
        print(f"  max |diff| = {d.max()}, mean |diff| = {d.mean():.6f}, "
              f"pixels differing = {(d > 0).sum()} / {d.size}")

    # Positive control. Bit-identity above is consistent with two different things: the two
    # encoders agree, or `_closed_loop_residual` is never called at all and both runs went down
    # some other path. Corrupt it and check the output moves -- if it doesn't, the probe above
    # proved nothing about the live code path.
    calls = {"n": 0}

    def corrupted(self, gaussians, prev_lvl, tgt_lvl, config, image_hw):
        calls["n"] += 1
        return np.zeros_like(orig(self, gaussians, prev_lvl, tgt_lvl, config, image_hw))

    ProgressiveLPyramidPolicy._closed_loop_residual = corrupted
    try:
        zeroed, _ = round_trip(cfg, images)
    finally:
        ProgressiveLPyramidPolicy._closed_loop_residual = orig

    moved = not np.array_equal(np.asarray(zeroed), np.asarray(fixed))
    print(f"  [control] _closed_loop_residual called {calls['n']}x; "
          f"zeroing it changes the decode: {moved}  -> live path confirmed: {moved and calls['n'] > 0}")
    print(f"  [control] relL2 with residual zeroed = {rel_l2(zeroed[0], images[0]):.4f}% "
          f"(this is the base-only / no-residual level)")


if __name__ == "__main__":
    main()
