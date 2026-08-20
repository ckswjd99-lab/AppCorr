"""Numerical gate for the SAM 3 ViT approx/correct fork. Needs no weights and no GPU.

Three assertions, in the order they should fail if the fork is wrong:

1. `forward` reproduces the stock layer. If this breaks, the wrapper mis-copied a submodule and
   nothing below means anything.
2. `approx` reproduces `forward`. `approx` re-implements attention by hand (to get at K/V) instead
   of calling `self.attention`, so this is the check that the hand-rolled path -- RoPE, the window
   round trip, the SDPA scale -- matches the library's.
3. `correct` over **all** tokens reproduces `forward`, and `correct` over a **subset** leaves the
   untouched positions exactly where an approx-only forward left them.

(3) is the one that matters. A partial correction that quietly perturbs untouched tokens is not a
cheaper forward, it is a different model, and the error is invisible in any end-to-end metric.

The windowed case is tested at a geometry with more than one window, because the whole point of the
window handling is that a corrected query must attend only within its own window and must take its
RoPE phase from its position *inside* that window. Both mistakes -- attending across windows, or
using global coordinates for RoPE -- produce finite, reasonable-looking numbers.

    python analysis/experiments/sam3_vision_fork_unittest.py
"""

import sys

import torch

sys.path.insert(0, "/NHNHOME/share/cjpark/AppCorr-sam3")

from transformers.models.sam3.configuration_sam3 import Sam3ViTConfig
from transformers.models.sam3.modeling_sam3 import Sam3ViTLayer

from appcorr.models.sam3.vision.block import ApproxCorrectSam3ViTLayer


def _report(name, got, ref, tol):
    err = (got.float() - ref.float()).abs().max().item()
    ok = err <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} max|diff| = {err:.3e}")
    return ok


def run_case(window_size: int, grid: int, patch: int = 14, tol: float = 5e-4) -> bool:
    torch.manual_seed(0)
    image = grid * patch
    cfg = Sam3ViTConfig(
        hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
        patch_size=patch, image_size=image, window_size=max(window_size, 1),
        intermediate_size=128, attention_dropout=0.0, hidden_dropout=0.0,
    )
    stock = Sam3ViTLayer(cfg, window_size=window_size).eval()
    fork = ApproxCorrectSam3ViTLayer.from_stock(stock).eval()

    n_win = 1 if window_size <= 0 else (grid // window_size) ** 2
    print(f"\ngrid {grid}x{grid}, window_size={window_size} -> {n_win} window(s), "
          f"{grid*grid} tokens")

    x = torch.randn(2, grid, grid, cfg.hidden_size)
    with torch.no_grad():
        ref = stock(x)
        ok = _report("forward == stock", fork(x), ref, tol)

        approx_out, cache = fork.approx(x, {}, "t")
        ok &= _report("approx == stock", approx_out, ref, tol)

        every = torch.arange(grid * grid)
        full_correct, _ = fork.correct(x, every, dict(cache), "t")
        ok &= _report("correct(all tokens) == stock", full_correct, ref, tol)

        # A scattered subset, deliberately spanning several windows.
        subset = torch.arange(0, grid * grid, 7)
        part, _ = fork.correct(x, subset, dict(cache), "t")
        ok &= _report("correct(subset) matches stock on corrected positions",
                      part.reshape(2, -1, cfg.hidden_size)[:, subset],
                      ref.reshape(2, -1, cfg.hidden_size)[:, subset], tol)

        untouched = torch.tensor([i for i in range(grid * grid) if i % 7])
        ok &= _report("correct(subset) leaves untouched positions at approx",
                      part.reshape(2, -1, cfg.hidden_size)[:, untouched],
                      approx_out.reshape(2, -1, cfg.hidden_size)[:, untouched], tol)
    return ok


if __name__ == "__main__":
    # 72x72 / window 24 is SAM 3's own geometry; the small cases run fast and fail faster.
    cases = [(0, 8), (4, 8), (24, 72), (0, 72)]
    results = [run_case(w, g) for w, g in cases]
    print("\n" + ("ALL CASES PASS" if all(results) else f"{results.count(False)} CASE(S) FAILED"))
    sys.exit(0 if all(results) else 1)
