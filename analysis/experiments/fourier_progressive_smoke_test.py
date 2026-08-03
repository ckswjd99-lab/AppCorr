"""
Manual smoke test for FourierProgressiveTransmissionPolicy.

Run directly: python analysis/experiments/fourier_progressive_smoke_test.py

Does not require a dataset or GPU. Validates:
  1. encode() yields exactly two groups (binary mode) with correct
     group_id / batch_group_total.
  2. group 0 patches carry a non-negative residual-energy pscore_hint.
  3. decode_lowres(group0) returns a full-size uint8 image.
  4. decode(group0 + group1) reconstructs the original image within DCT/IDCT
     (+ optional int16 rounding) tolerance, for both coeff_dtype="int16"
     (default) and coeff_dtype="float32".
  5. int16 coefficients produce a meaningfully smaller wire payload than
     float32 for the same content.
  6. Server-side accumulation semantics: decode() raises a clear error when
     a high-frequency group arrives without its matching group 0 patches.
  7. windowed_groups=true (COCOWindowInterleaved-style) yields
     1 (low-freq) + 9 (window) groups, and decode() correctly reconstructs
     the image incrementally as windows arrive one at a time.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from offload.common.protocol import ExperimentConfig
from offload.policies import get_transmission, TRANSMISSION_REGISTRY

# int16 rounding on top of float DCT/IDCT roundoff; still "very small" per spec.
_MAE_TOLERANCE = {"int16": 0.15, "float32": 0.02}
_MAX_ERR_TOLERANCE = 2


def make_config(
    preserve_input_shape: bool,
    coeff_dtype: str = "int16",
    windowed: bool = False,
    image_shape: tuple = (32, 32, 3),
) -> ExperimentConfig:
    return ExperimentConfig(
        batch_size=2,
        image_shape=image_shape,
        patch_size=(16, 16),
        transmission_policy_name="FourierProgressive",
        transmission_kwargs={
            "pyramid_levels": [2, 0],
            "compression_level": 1,
            "preserve_input_shape": preserve_input_shape,
            "coeff_dtype": coeff_dtype,
            "windowed_groups": windowed,
        },
    )


def run_roundtrip(preserve_input_shape: bool, coeff_dtype: str) -> int:
    """Returns total compressed byte size of group0+group1 (for the dtype comparison)."""
    tag = f"preserve_input_shape={preserve_input_shape},coeff_dtype={coeff_dtype}"
    config = make_config(preserve_input_shape, coeff_dtype=coeff_dtype)
    assert "FourierProgressive" in TRANSMISSION_REGISTRY

    rng = np.random.default_rng(0)
    B, H, W, C = config.batch_size, *config.image_shape
    images = rng.integers(0, 256, size=(B, H, W, C), dtype=np.uint8)

    policy = get_transmission("FourierProgressive")

    groups = list(policy.encode(images, config))
    assert len(groups) == 2, f"[{tag}] expected exactly 2 groups, got {len(groups)}"
    group0, group1 = groups

    assert all(p.group_id == 0 for p in group0), f"[{tag}] group0 patches must have group_id=0"
    assert all(p.group_id == 1 for p in group1), f"[{tag}] group1 patches must have group_id=1"
    assert all(p.batch_group_total == len(group0) for p in group0), f"[{tag}] group0 batch_group_total mismatch"
    assert all(p.batch_group_total == len(group1) for p in group1), f"[{tag}] group1 batch_group_total mismatch"
    assert all(p.pscore_hint >= 0.0 for p in group0), f"[{tag}] group0 pscore_hint must be >= 0"

    expected_patches = B * (H // 16) * (W // 16)
    assert len(group0) == expected_patches, f"[{tag}] group0 patch count {len(group0)} != {expected_patches}"
    assert len(group1) == expected_patches, f"[{tag}] group1 patch count {len(group1)} != {expected_patches}"

    lowres = policy.decode_lowres(group0, config)
    lowres = np.stack(lowres) if isinstance(lowres, list) else lowres
    assert lowres.shape == images.shape, f"[{tag}] decode_lowres shape {lowres.shape} != {images.shape}"
    assert lowres.dtype == np.uint8, f"[{tag}] decode_lowres dtype {lowres.dtype} != uint8"

    full = policy.decode(group0 + group1, config)
    full = np.stack(full) if isinstance(full, list) else full
    assert full.shape == images.shape, f"[{tag}] decode shape {full.shape} != {images.shape}"
    assert full.dtype == np.uint8, f"[{tag}] decode dtype {full.dtype} != uint8"

    diff = np.abs(full.astype(np.int16) - images.astype(np.int16))
    mae = diff.mean()
    max_err = diff.max()
    total_bytes = sum(len(p.data) for p in group0) + sum(len(p.data) for p in group1)
    print(f"[{tag}] full reconstruction: MAE={mae:.4f}, max_err={max_err}, total_bytes={total_bytes}")
    assert mae < _MAE_TOLERANCE[coeff_dtype], f"[{tag}] mean abs error too high: {mae}"
    assert max_err <= _MAX_ERR_TOLERANCE, f"[{tag}] max error too high: {max_err}"

    # group0-only decode() must fall back to the lowres path (no group1 present).
    approx_only = policy.decode(group0, config)
    approx_only = np.stack(approx_only) if isinstance(approx_only, list) else approx_only
    assert np.array_equal(approx_only, lowres), f"[{tag}] decode(group0) must match decode_lowres(group0)"

    # group1 without its group0 counterpart must raise a clear error (mirrors the
    # server dropping/never receiving the accumulated low-frequency patches).
    try:
        policy.decode(group1, config)
        raise AssertionError(f"[{tag}] decode(group1-only) should have raised")
    except RuntimeError as e:
        assert "group 0" in str(e)

    print(f"[{tag}] OK")
    return total_bytes


def run_dtype_size_comparison() -> None:
    bytes_int16 = run_roundtrip(preserve_input_shape=False, coeff_dtype="int16")
    bytes_float32 = run_roundtrip(preserve_input_shape=False, coeff_dtype="float32")
    print(f"[dtype-comparison] int16={bytes_int16}B vs float32={bytes_float32}B "
          f"({100 * bytes_int16 / bytes_float32:.1f}% of float32 size)")
    assert bytes_int16 < bytes_float32, (
        f"int16 payload ({bytes_int16}B) should be smaller than float32 ({bytes_float32}B)"
    )


def run_windowed_roundtrip() -> None:
    tag = "windowed_groups=True"
    # 3x3 raster windows need a grid large enough for three patch-aligned,
    # non-zero-size window bands; 96x96 @ 16x16 patches gives evenly-sized
    # 32x32 windows.
    config = make_config(preserve_input_shape=False, coeff_dtype="int16", windowed=True, image_shape=(96, 96, 3))

    rng = np.random.default_rng(1)
    B, H, W, C = config.batch_size, *config.image_shape
    images = rng.integers(0, 256, size=(B, H, W, C), dtype=np.uint8)

    policy = get_transmission("FourierProgressive")
    groups = list(policy.encode(images, config))
    assert len(groups) == 1 + 9, f"[{tag}] expected 1 low-freq + 9 window groups, got {len(groups)}"

    group0 = groups[0]
    assert all(p.group_id == 0 for p in group0)
    window_group_ids = sorted({p.group_id for group in groups[1:] for p in group})
    assert window_group_ids == list(range(1, 10)), f"[{tag}] unexpected window group ids: {window_group_ids}"
    total_window_patches = sum(len(g) for g in groups[1:])
    expected_patches = B * (H // 16) * (W // 16)
    assert total_window_patches == expected_patches, (
        f"[{tag}] window groups should partition all patches exactly once: "
        f"{total_window_patches} != {expected_patches}"
    )

    # Simulate incremental arrival: group0, then each window group in turn,
    # accumulating into a growing patch buffer (mirrors worker.py's
    # context['patch_buffer'] behavior for non-incremental-decode policies).
    accumulated = list(group0)
    approx = policy.decode(accumulated, config)
    approx = np.stack(approx) if isinstance(approx, list) else approx
    prev_mae = np.abs(approx.astype(np.int16) - images.astype(np.int16)).mean()

    for window_group in groups[1:]:
        accumulated.extend(window_group)
        partial = policy.decode(accumulated, config)
        partial = np.stack(partial) if isinstance(partial, list) else partial
        mae = np.abs(partial.astype(np.int16) - images.astype(np.int16)).mean()
        assert mae <= prev_mae + 1e-6, f"[{tag}] MAE should be non-increasing as windows arrive: {prev_mae} -> {mae}"
        prev_mae = mae

    final = policy.decode(accumulated, config)
    final = np.stack(final) if isinstance(final, list) else final
    final_mae = np.abs(final.astype(np.int16) - images.astype(np.int16)).mean()
    print(f"[{tag}] final MAE after all windows: {final_mae:.4f}")
    assert final_mae < _MAE_TOLERANCE["int16"], f"[{tag}] final mean abs error too high: {final_mae}"
    print(f"[{tag}] OK")


def run_existing_policies_smoke() -> None:
    """Existing policies/configs must still instantiate unaffected."""
    for name in ("Raw", "Zlib", "Laplacian", "ProgressiveLaplacian", "COCOWindowProgressiveLaplacian"):
        get_transmission(name)
    print("[existing-policies] OK: Raw, Zlib, Laplacian, ProgressiveLaplacian, COCOWindowProgressiveLaplacian instantiate")


if __name__ == "__main__":
    run_dtype_size_comparison()
    run_roundtrip(preserve_input_shape=True, coeff_dtype="int16")
    run_windowed_roundtrip()
    run_existing_policies_smoke()
    print("All FourierProgressive smoke tests passed.")
