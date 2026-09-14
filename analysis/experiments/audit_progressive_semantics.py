"""
audit_progressive_semantics.py

Re-audit of the progressive-VLA implementation against original AppCorr semantics, testing two
specific claims made earlier that deserve scrutiny:

CLAIM A ("dead stream"): The driver passes the FULL TRUE image to correct_forward, whereas AppCorr's
  canvas keeps non-corrected regions blurred (decode() only fills where residuals arrived). This makes
  non-corrected positions' residual-stream values inconsistent hybrids (true embed + blur-pass deltas).
  The code-trace argument is that these values are never read by anything that reaches the decoded
  action (attention always reads the K/V cache; only queried rows feed norm1/MLP; the LLM consumes only
  the last position's logits + the K/V cache). TEST: run the same partial correction with (i) full-true
  image and (ii) a faithful AppCorr-style partial canvas (true pixels only inside corrected patches,
  blurred elsewhere). If the trace is right, last-position logits are bit-identical.

CLAIM B ("sequential multi-round is exact"): asserted earlier from the causal-LLM argument. For the
  BIDIRECTIONAL vision towers this cannot be bit-exact: round-1 patches' K/V at layers >= 1 were
  computed while round-2 patches were still blurred, and are never revisited after round 2 corrects
  them. TEST: compare (i) single-round 100% correction (should be bit-exact vs stock -- established in
  Phase 1) against (ii) two-round sequential 100% correction, at the level of final last-position
  LOGITS (not the binned action, which can mask small errors via argmax).

Run:
    USE_TF=0 USE_TORCH=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 MUJOCO_EGL_ALLOW_ANY_DEVICE=1 \
    python analysis/experiments/audit_progressive_semantics.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from analysis.experiments.progressive_vla_smoke import get_one_libero_frame, blur_downup, action_str
from appcorr.models.openvla.progressive_model import OpenVLAProgressiveModel

PATCH = 14  # both towers: 224px / patch14 -> 16x16 grid


def make_partial_canvas(true_px: torch.Tensor, blur_px: torch.Tensor, patch_idx: torch.Tensor) -> torch.Tensor:
    """AppCorr-faithful canvas: true pixels only inside `patch_idx` patches, blurred elsewhere.
    Applied identically to both channel groups (DINOv2 + SigLIP share the same patch grid)."""
    canvas = blur_px.clone()
    gw = true_px.shape[-1] // PATCH
    for p in patch_idx.tolist():
        r, c = divmod(p, gw)
        hs, ws = r * PATCH, c * PATCH
        canvas[:, :, hs:hs + PATCH, ws:ws + PATCH] = true_px[:, :, hs:hs + PATCH, ws:ws + PATCH]
    return canvas


def last_logits(model) -> torch.Tensor:
    x = model.cache_feature["_x"]
    return model._logits_from_x(x)[:, -1].float()


def report(name, a, b):
    d = (a - b).abs()
    print(f"    [{name}] max_abs={d.max().item():.6f} mean_abs={d.mean().item():.6f} "
          f"argmax_equal={bool((a.argmax(-1) == b.argmax(-1)).all().item())}")
    return d.max().item()


def main():
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    img_np, task_description = get_one_libero_frame("libero_spatial", 0)
    image = Image.fromarray(img_np).convert("RGB")
    model = OpenVLAProgressiveModel("openvla/openvla-7b-finetuned-libero-spatial", device, unnorm_key="libero_spatial")
    model.start_session(image, task_description, center_crop=True)
    true_px = model.reference_pixel_values.to(device=device, dtype=torch.bfloat16)
    blur_px = blur_downup(true_px.float(), factor=4).to(dtype=torch.bfloat16)

    with torch.no_grad():
        # Stock reference logits (last prefill position), via the oracle-identical forward.
        out = model.vla(
            input_ids=model.input_ids, pixel_values=true_px, use_cache=False, return_dict=True
        )
        stock_last = out.logits[:, -1].float()

        # ============ CLAIM A: full-true image vs faithful partial canvas ============
        print("\n=== CLAIM A: correct_forward(full true image) vs correct_forward(partial canvas) ===")
        k = 128  # 50% sequential prefix
        patch_idx = torch.arange(0, k, device=device)

        model.start_session(image, task_description, center_crop=True)
        model.approx_forward(blur_px)
        model.correct_forward(true_px, patch_idx)          # what the driver has been doing
        logits_fulltrue = last_logits(model)
        act_fulltrue = model.decode_action()

        model.start_session(image, task_description, center_crop=True)
        model.approx_forward(blur_px)
        canvas = make_partial_canvas(true_px, blur_px, patch_idx)
        model.correct_forward(canvas, patch_idx)           # AppCorr-faithful semantics
        logits_canvas = last_logits(model)
        act_canvas = model.decode_action()

        report("last-logits: full-true vs partial-canvas", logits_fulltrue, logits_canvas)
        print(f"    action(full-true):      {action_str(act_fulltrue)}")
        print(f"    action(partial-canvas): {action_str(act_canvas)}")
        print(f"    actions identical: {np.allclose(act_fulltrue, act_canvas)}")

        # ============ CLAIM B: single-round vs two-round sequential 100% ============
        print("\n=== CLAIM B: 100% correction -- single round vs two sequential rounds, vs stock ===")
        model.start_session(image, task_description, center_crop=True)
        model.approx_forward(blur_px)
        model.correct_forward(true_px, torch.arange(0, 256, device=device))
        logits_1round = last_logits(model)

        model.start_session(image, task_description, center_crop=True)
        model.approx_forward(blur_px)
        model.correct_forward(true_px, torch.arange(0, 128, device=device))
        model.correct_forward(true_px, torch.arange(128, 256, device=device))
        logits_2round = last_logits(model)

        e1 = report("single-round 100% vs stock", logits_1round, stock_last)
        e2 = report("two-round sequential 100% vs stock", logits_2round, stock_last)
        report("two-round vs single-round", logits_2round, logits_1round)

        print("\n=== Verdicts ===")
        print(f"    CLAIM A (dead-stream: canvas choice irrelevant to outputs): "
              f"{'CONFIRMED' if torch.equal(logits_fulltrue, logits_canvas) else 'REFUTED -- canvas choice changes consumed outputs'}")
        print(f"    CLAIM B (multi-round sequential bit-exact): "
              f"{'CONFIRMED (bit-exact)' if e2 <= e1 + 1e-6 and e2 < 1e-3 else 'REFUTED -- multi-round leaves residual error (as re-derived: bidirectional vision K/V staleness)'}")


if __name__ == "__main__":
    main()
