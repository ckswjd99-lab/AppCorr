"""Numerical gate for the InternViT approx/correct fork. Needs no weights.

Four assertions, in the order they should fail if the fork is wrong:

1. `forward` reproduces the stock layer. If this breaks, the wrapper mis-copied a submodule and
   nothing below means anything.
2. `approx` reproduces `forward`. `approx` re-implements attention by hand (to get at K/V) instead
   of calling `self.attention`, so this checks the hand-rolled path -- QK-norm, the SDPA scale, the
   LayerScale factors, and the order they are applied in.
3. `correct` over **all** tokens reproduces `forward`.
4. `correct` over a **subset** leaves the untouched positions exactly where an approx-only forward
   left them.

(4) is the one that matters. A partial correction that quietly perturbs untouched tokens is not a
cheaper forward, it is a different model, and the error is invisible in any end-to-end metric.

The batch dimension here is **tiles**, and the subset test uses a *different* selection per tile,
because that is how the real harness will call it: the patch score ranks patches across the whole
image, so one tile can need many tokens corrected and another almost none. A fork that only handles
a batch-shared index vector would pass a naive test and fail in use.

    python analysis/experiments/internvl_vision_fork_unittest.py [--device cuda:0]
"""

import argparse
import sys

import torch

sys.path.insert(0, "/NHNHOME/share/cjpark/AppCorr-internvl")

from transformers.models.internvl.configuration_internvl import InternVLVisionConfig
from transformers.models.internvl.modeling_internvl import InternVLVisionLayer

from appcorr.models.internvl.vision.block import ApproxCorrectInternVLVisionLayer


def _report(name, got, ref, tol):
    if got.numel() == 0:
        print(f"  SKIP  {name:<54} (empty selection)")
        return True
    err = (got.float() - ref.float()).abs().max().item()
    ok = err <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<54} max|diff| = {err:.3e}")
    return ok


def run_case(tiles: int, tokens: int, device: str, dtype: torch.dtype, tol: float) -> bool:
    torch.manual_seed(0)
    cfg = InternVLVisionConfig(
        hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=128, attention_dropout=0.0, hidden_dropout_prob=0.0,
    )
    stock = InternVLVisionLayer(cfg).to(device=device, dtype=dtype).eval()
    # LayerScale ships at a constant 0.1; randomise so a dropped lambda cannot pass by symmetry.
    with torch.no_grad():
        stock.lambda_1.copy_(torch.randn_like(stock.lambda_1) * 0.3 + 1.0)
        stock.lambda_2.copy_(torch.randn_like(stock.lambda_2) * 0.3 + 1.0)
    fork = ApproxCorrectInternVLVisionLayer.from_stock(stock).eval()

    print(f"\n{tiles} tile(s) x {tokens} tokens, hidden {cfg.hidden_size}, {device}/{dtype}")
    x = torch.randn(tiles, tokens, cfg.hidden_size, device=device, dtype=dtype)

    with torch.no_grad():
        ref = stock(x)
        ok = _report("forward == stock", fork(x), ref, tol)

        approx_out, cache = fork.approx(x, {}, "t")
        ok &= _report("approx == stock", approx_out, ref, tol)

        everything = torch.ones(tiles, tokens, dtype=torch.bool, device=device)
        full, _ = fork.correct(x, everything, dict(cache), "t")
        ok &= _report("correct(all tokens) == stock", full, ref, tol)

        # A DIFFERENT subset per tile, including one empty tile and one full tile.
        # Per-tile selections that differ in KIND, not just in count: a scattered subset, and --
        # once there are enough tiles -- one left empty and one taken whole, since those are the
        # two boundary cases a batch loop gets wrong.
        mask = torch.zeros(tiles, tokens, dtype=torch.bool, device=device)
        for b in range(tiles):
            if tiles > 2 and b == 0:
                continue                                   # empty: must be left at approx
            if tiles > 2 and b == 1:
                mask[b] = True                             # full: must match stock
            else:
                mask[b, torch.arange(b % 3, tokens, (b % 4) + 2)] = True
        part, _ = fork.correct(x, mask, dict(cache), "t")

        ok &= _report("correct(subset) matches stock where corrected",
                      part[mask], ref[mask], tol)
        ok &= _report("correct(subset) leaves untouched positions at approx",
                      part[~mask], approx_out[~mask], tol)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args()
    dt = torch.float32 if a.dtype == "float32" else torch.bfloat16
    tol = 5e-4 if dt is torch.float32 else 5e-2
    # (tiles, tokens): small cases fail fast; 13x1025 is InternVL3's own worst case
    # (12 tiles + thumbnail, 1 CLS + 32x32 patches).
    cases = [(1, 17), (4, 65), (13, 1025)]
    results = [run_case(t, n, a.device, dt, tol) for t, n in cases]
    print("\n" + ("ALL CASES PASS" if all(results) else f"{results.count(False)} CASE(S) FAILED"))
    sys.exit(0 if all(results) else 1)
