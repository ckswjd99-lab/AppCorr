"""Gates for the streaming-prefill arm: exact chunked prefill over progressively arriving bands.

This arm makes the opposite trade from `interleaved_forward`. It gives up vision freshness -- a band
already prefilled is never revisited, even though vision attention is global and later bands change
it -- and in exchange the LLM half carries NO approximation: every token is prefilled exactly once,
in causal order, against real K/V.

Four gates:

  1. **g=1 == the ceiling, exactly.** With one band the schedule degenerates: the base pass runs,
     then the single band recomputes 100% of the patches from the full-resolution stream (which the
     axis check proved is the exact vision forward), then one prefill chunk covers the whole
     sequence. So this must reproduce the exact forward in feature space AND token for token. It is
     the strongest gate available here because it needs no tolerance argument.
  2. **Bands partition the image tokens.** Contiguous, increasing, covering every token exactly
     once -- otherwise "prefill this band" is not a chunk and a token is either skipped or done
     twice.
  3. **Every position is prefilled exactly once.** The counter must equal the sequence length. More
     means a chunk was re-run (the cost claim dies); fewer means something never entered the cache.
  4. **Vision cost is 2x, LLM cost is 1x.** The schedule's whole claim. One base pass plus bands
     that together are one more pass; the LLM prefill counted in tokens must equal the sequence.

    python analysis/experiments/ov2_streaming_gates.py [--groups 4] [--dtype float32]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from appcorr.models.ov2.unified import OV2UnifiedAxis
from experiments.ov2_degradation import hw_from_grid, l2_from_native
from experiments.ov2_oracle import encode, generate_from_kv, load, run_stock
from qwen_vl_prefill.datasets_eval import get_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chartqa")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args()

    from datasets import load_dataset

    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    rtol = 3e-4 if dtype is torch.float32 else 5e-2
    model, proc = load(a.device, dtype)
    axis = OV2UnifiedAxis(model.model).eval()

    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    idxs = list(range(0, len(ds), max(1, len(ds) // a.samples)))[:a.samples]

    ok = True
    same_text = 0
    with torch.no_grad():
        for k, i in enumerate(idxs, 1):
            img, prompt, _ = spec.prepare(ds[i], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
            enc = encode(proc, img, prompt, a.device)
            ids, pp = enc["input_ids"], enc["patch_positions"]
            px = enc["pixel_values"].to(dtype)
            deg = l2_from_native(img, a.level, proc,
                                 hw_from_grid(enc["image_grid_thw"], proc))
            px2 = encode(proc, deg, prompt, a.device)["pixel_values"].to(dtype)
            n_patch, seq = int(px.shape[0]), int(ids.shape[1])
            n_tok = axis.n_tokens(n_patch)

            # --- 2: bands partition the tokens --------------------------------------------- #
            for gN in (1, 2, a.groups, 8):
                b = axis.token_bands(gN, pp)
                nb = [x for x in b if x[1] > x[0]]
                covered = sum(t1 - t0 for t0, t1 in nb)
                contiguous = all(nb[j][1] == nb[j + 1][0] for j in range(len(nb) - 1))
                good = covered == n_tok and contiguous and nb[0][0] == 0
                ok &= good
                if k == 1:
                    print(f"  g={gN:<2} bands={str(nb)[:60]:<62} cover {covered}/{n_tok} "
                          f"contig={contiguous}  {'PASS' if good else 'FAIL'}")

            # --- 1: g=1 reproduces the exact forward --------------------------------------- #
            exact = axis.full_forward(px, pp, ids)
            h1, kv1, s1 = axis.streaming_forward(px2, px, pp, ids, 1)
            r = float((h1.float() - exact.float()[:, -h1.shape[1]:]).abs().max()
                      / exact.float().abs().max().clamp_min(1e-9))
            g1 = r <= rtol
            ok &= g1

            txt_ceil, _ = run_stock(model, proc, img, prompt, "ceiling", a.level, a.device, 24)
            txt_s1 = generate_from_kv(model, proc, h1, kv1, ids, 24)
            same = txt_ceil == txt_s1
            same_text += int(same)
            ok &= same

            # --- 3 & 4: cost counters ------------------------------------------------------ #
            _, _, sg = axis.streaming_forward(px2, px, pp, ids, a.groups)
            once = sg["prefill_tokens"] == seq
            vis2 = abs(sg["vision_layer_passes"] / axis.n_vision - 2.0) < 1e-6
            ok &= once and vis2

            print(f"  [{k}/{len(idxs)}] n_patch={n_patch:<5} seq={seq:<5} "
                  f"g=1 vs exact rel {r:.2e} {'OK' if g1 else 'BAD'}  "
                  f"text {'SAME' if same else 'DIFFER'}  "
                  f"prefill {sg['prefill_tokens']}/{seq} {'OK' if once else 'BAD'}  "
                  f"vision {sg['vision_layer_passes']/axis.n_vision:.3f}x "
                  f"{'OK' if vis2 else 'BAD'}", flush=True)
            if not same:
                print(f"        ceil={txt_ceil[:60]!r}\n        s1  ={txt_s1[:60]!r}")

    print(f"\n  g=1 == ceiling, token for token: {same_text}/{len(idxs)}")
    print("\n" + ("ALL GATES PASS" if ok else "GATE FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
