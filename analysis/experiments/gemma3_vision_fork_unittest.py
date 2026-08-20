"""Numerical gate for the SigLIP approx/correct fork. Needs no weights.

Four assertions, in the order they should fail if the fork is wrong:

1. `forward` reproduces the stock layer. If this breaks, the wrapper mis-copied a submodule and
   nothing below means anything.
2. `approx` reproduces `forward`. `approx` re-implements attention by hand (to get at K/V) instead
   of calling `self.self_attn`, so this checks the hand-rolled path -- the projection reshape, the
   SDPA scale, and the order of the two residual adds.
3. `correct` over **all** tokens reproduces `forward`.
4. `correct` over a **subset** leaves untouched positions exactly where an approx-only forward left
   them.

(4) is the one that matters. A partial correction that quietly perturbs untouched tokens is not a
cheaper forward, it is a different model, and the error is invisible in any end-to-end metric.

The subset case gives each batch element a **different** selection, including one empty and one
full, because that is how the real harness calls it -- a patch score ranks across the whole image and
different images need different amounts corrected. A fork that only handles a batch-shared index
vector passes a naive test and fails in use.

The last geometry is Gemma 3 4B's own: 896/14 = 64x64 = 4096 tokens.

    python analysis/experiments/gemma3_vision_fork_unittest.py [--device cuda:0]
"""

import argparse
import sys

import torch

sys.path.insert(0, "/NHNHOME/share/cjpark/AppCorr-gemma3")

from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

from appcorr.models.gemma3.vision.block import ApproxCorrectSiglipLayer


def _report(name, got, ref, rtol):
    if got.numel() == 0:
        print(f"  SKIP  {name:<52} (empty selection)")
        return True
    scale = max(ref.float().abs().max().item(), 1e-9)
    err = (got.float() - ref.float()).abs().max().item()
    ok = err / scale <= rtol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} rel {err/scale:.2e}  (abs {err:.3e})")
    return ok


def run_case(batch: int, tokens: int, hidden: int, device: str, dtype, rtol: float) -> bool:
    torch.manual_seed(0)
    cfg = SiglipVisionConfig(hidden_size=hidden, num_hidden_layers=1, num_attention_heads=4,
                             intermediate_size=hidden * 2, attention_dropout=0.0)
    cfg._attn_implementation = "sdpa"
    stock = SiglipEncoderLayer(cfg).to(device=device, dtype=dtype).eval()
    fork = ApproxCorrectSiglipLayer.from_stock(stock).eval()

    print(f"\n{batch} image(s) x {tokens} tokens, hidden {hidden}, {device}/{dtype}")
    x = torch.randn(batch, tokens, hidden, device=device, dtype=dtype)

    with torch.no_grad():
        ref = stock(x, attention_mask=None)
        ok = _report("forward == stock", fork(x), ref, rtol)

        approx_out, cache = fork.approx(x, {}, "t")
        ok &= _report("approx == stock", approx_out, ref, rtol)

        every = torch.ones(batch, tokens, dtype=torch.bool, device=device)
        full, _ = fork.correct(x, every, dict(cache), "t")
        ok &= _report("correct(all tokens) == stock", full, ref, rtol)

        # Different selection per image; with >2 images include an empty one and a whole one,
        # the two boundary cases a per-batch loop gets wrong.
        mask = torch.zeros(batch, tokens, dtype=torch.bool, device=device)
        for b in range(batch):
            if batch > 2 and b == 0:
                continue
            if batch > 2 and b == 1:
                mask[b] = True
            else:
                mask[b, torch.arange(b % 3, tokens, (b % 4) + 2)] = True
        part, _ = fork.correct(x, mask, dict(cache), "t")

        ok &= _report("correct(subset) matches stock where corrected", part[mask], ref[mask], rtol)
        ok &= _report("correct(subset) leaves untouched at approx",
                      part[~mask], approx_out[~mask], rtol)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args()
    dt = torch.float32 if a.dtype == "float32" else torch.bfloat16
    rtol = 1e-4 if dt is torch.float32 else 5e-2
    # (batch, tokens, hidden); the last is Gemma 3 4B's own 64x64 grid at its real width.
    cases = [(1, 17, 64), (4, 65, 64), (2, 4096, 1152)]
    results = [run_case(b, n, h, a.device, dt, rtol) for b, n, h in cases]
    print("\n" + ("ALL CASES PASS" if all(results) else f"{results.count(False)} CASE(S) FAILED"))
    sys.exit(0 if all(results) else 1)
