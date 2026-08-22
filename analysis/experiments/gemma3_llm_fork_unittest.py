"""Numerical gate for the Gemma 3 decoder approx/correct fork. Needs no weights.

The four standard assertions (forward == stock, approx == stock, correct(all) == stock,
correct(subset) leaves untouched tokens at approx), plus one that exists because this layer has a
failure mode producing finite, plausible numbers:

**RoPE slicing and mask rows.** A corrected token must be rotated by ITS OWN position and must use
ITS OWN mask row. Taking the first `Q` rows of cos/sin and of the mask — the natural mistake, since
that is what a contiguous prefill wants — still runs and still returns a tensor. Two things catch it:
the subset corrected here is *late and scattered* rather than a prefix, so wrong slicing cannot
coincide with right slicing; and a deliberately phase-shifted variant is checked to be detectably
different, which proves the test can tell them apart at all.

Gemma 3 also gives sliding and full layers **different rope parameters**, so cos/sin are per
layer_type and the model hands each layer its own. The unified-axis wrapper has to carry that
through; this test uses layer 0's type.

    python analysis/experiments/gemma3_llm_fork_unittest.py [--device cuda:0]
"""

import argparse
import sys

import torch

sys.path.insert(0, "/NHNHOME/share/cjpark/AppCorr-gemma3")

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3RotaryEmbedding

from appcorr.models.gemma3.llm.decoder_layer import ApproxCorrectGemma3DecoderLayer


def rep(name, got, ref, rtol):
    if got.numel() == 0:
        print(f"  SKIP  {name:<54} (empty selection)")
        return True
    scale = max(ref.float().abs().max().item(), 1e-9)
    err = (got.float() - ref.float()).abs().max().item()
    ok = err / scale <= rtol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<54} rel {err/scale:.2e}  (abs {err:.3e})")
    return ok


def prefix_lm_mask(n_img, n_txt, device, dtype):
    """Gemma 3's shape: image tokens bidirectional among themselves, text causal."""
    n = n_img + n_txt
    ok = torch.zeros(n, n, dtype=torch.bool, device=device)
    ok[:n_img, :n_img] = True
    for i in range(n_img, n):
        ok[i, : i + 1] = True
    m = torch.zeros(1, 1, n, n, dtype=dtype, device=device)
    m.masked_fill_(~ok, torch.finfo(dtype).min)
    return m


def run_case(n_img, n_txt, hidden, device, dtype, rtol):
    torch.manual_seed(0)
    n = n_img + n_txt
    cfg = Gemma3TextConfig(hidden_size=hidden, num_hidden_layers=1, num_attention_heads=8,
                           num_key_value_heads=4, head_dim=32, intermediate_size=hidden * 2,
                           attention_dropout=0.0, sliding_window=4096)
    cfg._attn_implementation = "eager"
    stock = Gemma3DecoderLayer(cfg, layer_idx=0).to(device=device, dtype=dtype).eval()
    fork = ApproxCorrectGemma3DecoderLayer.from_stock(stock).eval()

    rot = Gemma3RotaryEmbedding(cfg).to(device=device, dtype=dtype)
    x = torch.randn(1, n, hidden, device=device, dtype=dtype)
    pos = torch.arange(n, device=device).unsqueeze(0)
    pe = rot(x, pos, cfg.layer_types[0])
    mask = prefix_lm_mask(n_img, n_txt, device, dtype)

    print(f"\n{n_img} image + {n_txt} text = {n} tokens, hidden {hidden}, GQA 8/4, "
          f"layer_type={cfg.layer_types[0]}, {device}/{dtype}")
    with torch.no_grad():
        ref = stock(x, position_embeddings=pe, attention_mask=mask, position_ids=pos)
        ok = rep("forward == stock",
                 fork(x, position_embeddings=pe, attention_mask=mask, position_ids=pos), ref, rtol)

        approx_out, cache = fork.approx(x, pe, mask, {}, "t")
        ok &= rep("approx == stock", approx_out, ref, rtol)

        every = torch.ones(1, n, dtype=torch.bool, device=device)
        full, _ = fork.correct(x, every, pe, mask, dict(cache), "t")
        ok &= rep("correct(all tokens) == stock", full, ref, rtol)

        # LATE and SCATTERED, not a prefix: this is what separates correct RoPE/mask slicing from
        # taking the first Q rows.
        sel = torch.zeros(1, n, dtype=torch.bool, device=device)
        sel[0, n // 2 :: 3] = True
        sel[0, -1] = True                       # a text token at the very end
        part, _ = fork.correct(x, sel, pe, mask, dict(cache), "t")
        ok &= rep("correct(late, scattered) == stock where corrected", part[sel], ref[sel], rtol)
        ok &= rep("correct(subset) leaves untouched at approx", part[~sel], approx_out[~sel], rtol)

        # Prove the test can tell right phase from wrong: rotate the same tokens by a shifted set
        # of positions. Shape stays valid; only the phase is wrong.
        # Roll the WHOLE cos/sin along the sequence axis: `correct` slices it by token index, so
        # the tensor must stay full length (pre-slicing it here would index out of range inside the
        # fork -- the same off-by-design that the fork itself has to avoid).
        cos, sin = pe
        wrong_pe = (torch.roll(cos, 1, dims=1), torch.roll(sin, 1, dims=1))
        wrong, _ = fork.correct(x, sel, wrong_pe, mask, dict(cache), "t")
        wscale = max(ref[sel].float().abs().max().item(), 1e-9)
        d = (wrong[sel].float() - ref[sel].float()).abs().max().item() / wscale
        print(f"  {'PASS' if d > rtol * 10 else 'FAIL'}  "
              f"{'wrong-phase RoPE is detectably different':<54} rel {d:.2e}")
        ok &= d > rtol * 10
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args()
    dt = torch.float32 if a.dtype == "float32" else torch.bfloat16
    rtol = 2e-4 if dt is torch.float32 else 5e-2
    cases = [(8, 5, 128), (256, 21, 256)]     # the second is Gemma 3's real image-token count
    res = [run_case(i, t, h, a.device, dt, rtol) for i, t, h in cases]
    print("\n" + ("ALL CASES PASS" if all(res) else f"{res.count(False)} CASE(S) FAILED"))
    sys.exit(0 if all(res) else 1)
