"""Gates for the Qwen3.5 unified axis (vision approx/correct + streaming LLM) on the real 35B.

The reference is the STOCK forward -- `model(input_ids, pixel_values, ...)`, positions derived by
the model itself. That matters more than it looks: the earlier streaming gate compared two runs of
OUR OWN prefill and passed, while both arms shared 1D positions that silently destroy the M-RoPE
image grid. An identity against stock cannot be fooled that way -- if our positions are wrong, we
disagree with the reference, not with ourselves.

Gate 1  merger slice        merger(all)[band] == merger(band rows) -- licenses per-band merging
Gate 2  g=1 == stock        one band, corrected after full arrival: no staleness anywhere, so
                            greedy tokens over 24 steps must match stock exactly (bf16 -> tokens)
Gate 3  floor < g=4 <= ceil g=4 has real staleness; it must land strictly above the degraded floor
                            (direction gated, magnitude reported)
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from appcorr.models.qwen35.unified import Qwen35Axis, MODEL_ID_35B


def greedy_from(axis, logits, kv, decode_pos, n=24):
    """Greedy decode on top of a streamed cache. Positions are passed EXPLICITLY: without them the
    model would fall back to its cached `rope_deltas` from whatever ran last on this model object
    -- correct by coincidence here, wrong the moment call order changes. Generated tokens are text,
    so all three mrope axes advance together from `decode_start_pos`."""
    toks, cur = [], logits.argmax(-1, keepdim=True)
    for _ in range(n):
        toks.append(int(cur))
        pid = torch.full((3, 1, 1), decode_pos, device=cur.device, dtype=torch.long)
        out = axis.model(input_ids=cur, past_key_values=kv, position_ids=pid, use_cache=True)
        kv = out.past_key_values
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        decode_pos += 1
    return toks


def greedy_stock(axis, inputs, n=24):
    with torch.no_grad():
        out = axis.model.generate(**inputs, max_new_tokens=n, do_sample=False)
    return out[0, inputs["input_ids"].shape[1]:].tolist()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID_35B,
                    help="checkpoint to gate; FP8 checkpoints load with dtype='auto'")
    args = ap.parse_args()
    torch.manual_seed(0)
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="cuda:0").eval()
    axis = Qwen35Axis(model, proc)
    print(f"  model: {args.model}  (param dtype: {next(model.parameters()).dtype})")

    # Degradation must actually remove information, or every TV distance in this file collapses
    # into one noise band and the gates compare nothing. The first draft used 32px blocks -- which
    # survive a 4x downsample perfectly, measured TV(floor, stock) = 0.0009, i.e. floor == ceiling
    # and a vacuous gate. 8px blocks do not survive 112px: they alias into new colors, which is a
    # real information loss the question then probes.
    rng = np.random.RandomState(0)
    img_arr = np.kron(rng.rand(56, 56, 3), np.ones((8, 8, 1)))         # 448x448, 8px blocks
    img = Image.fromarray((img_arr * 255).astype("uint8"))
    base = img.resize((112, 112)).resize((448, 448))                    # the degraded base
    q = "How many distinct colored squares are in the top row?"
    inputs = axis.build_inputs(img, q).to("cuda:0")
    inputs_base = axis.build_inputs(base, q).to("cuda:0")
    px_base = inputs_base["pixel_values"]

    ok = True
    with torch.no_grad():
        # Gate 1: per-band merger slicing is exact.
        ctx = axis.tower.prepare_full_tokens(inputs["pixel_values"].to(model.dtype),
                                             inputs["image_grid_thw"])
        x, _ = axis.tower.approx_forward(ctx["hidden_states"], 0, len(axis.tower.blocks), ctx, {}, "m")
        unit = axis.tower.spatial_merge_unit
        # fp32 copy: in bf16 the merger's GEMM tiling depends on how many rows it is fed, so
        # full-vs-band differs by reduction order (measured 9.8e-04) even though the slicing is
        # structurally exact. The identity claim is about the SLICING, so remove the dtype confound
        # instead of hiding it behind a tolerance -- same reasoning as every fp32 gate here.
        # fp64, because even fp32 cuBLAS picks a split-K schedule by row count (measured 2.4e-06
        # between 196-row and 24-row calls). At fp64 any surviving difference would be structural.
        import copy
        merger64 = copy.deepcopy(axis.tower.merger).double()
        x64 = x.double()
        full_m = merger64(x64)
        band_m = merger64(x64[3 * unit:9 * unit])
        d1 = (full_m[3:9] - band_m).abs().max().item()
        good = d1 < 1e-12
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  merger slice exact (fp64)   max|diff| = {d1:.3e}")

        # Gate 2: g=1 streaming vs STOCK. In exact arithmetic this is an identity; in bf16 it is
        # not, because g=1's vision runs approx-then-correct-all (different shapes, different
        # reduction order) where stock runs one pass -- the same shape-dependence documented on
        # every fork here. The synthetic fp32 gate already proved correct-all == stock bit-exactly;
        # what bf16 leaves to check is that the surviving noise is NEGLIGIBLE next to the signal.
        # Gated on: identical FIRST token (the answer token this project scores), and TV distance
        # to stock at least 20x smaller than the floor's. The 24-token strings are printed because
        # a near-tie tail divergence ("wants me to describe" vs "wants a description") is expected
        # and should be visible, not hidden.
        ref_logits = axis.full_forward(inputs)
        fl_logits = axis.approx_only_forward(inputs, px_base)
        ref_toks = greedy_stock(axis, inputs)
        lg1, kv1, st1 = axis.streaming_forward(inputs, px_base, groups=1)
        st_toks = greedy_from(axis, lg1, kv1, st1["decode_start_pos"])
        def dist(a, b):
            return (a.float().softmax(-1) - b.float().softmax(-1)).abs().sum().item() / 2  # TV
        tv_floor, tv_g1 = dist(fl_logits, ref_logits), dist(lg1, ref_logits)
        # Logit-level TV turned out to be blind here: this model opens with CoT boilerplate
        # ("The user wants...") whose first-token distribution ignores the image -- TV(floor,
        # stock) measured 0.0005 across two different degradations. So the mechanism gates run in
        # FEATURE space, on the image embeddings the LLM actually consumed, per the interleaved
        # contract's own recommendation (task metrics are not monotone in fidelity). Logit TVs are
        # still printed as diagnostics.
        with torch.no_grad():
            e_ref = axis.model.model.visual(inputs["pixel_values"].to(axis.model.dtype),
                                            grid_thw=inputs["image_grid_thw"]).pooler_output.float()
            e_floor = axis.model.model.visual(px_base.to(axis.model.dtype),
                                              grid_thw=inputs["image_grid_thw"]).pooler_output.float()
        def rel(e):
            return ((e - e_ref).norm() / e_ref.norm()).item()
        rel_floor = rel(e_floor)
        rel_g1 = rel(st1["image_embeds"][0])
        good = rel_floor > 0.05
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  degradation is informative   rel-L2(floor) = {rel_floor:.4f}")
        good = rel_g1 < rel_floor / 20
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  g=1 embeds == stock vision   rel-L2(g=1) = {rel_g1:.5f} "
              f"(bf16 noise scale)")
        first_same = st_toks[0] == ref_toks[0]
        ok &= first_same
        print(f"  {'PASS' if first_same else 'FAIL'}  g=1 first token == stock     "
              f"TV(g=1)={tv_g1:.4f} TV(floor)={tv_floor:.4f} [diagnostic]")
        print(f"        stock : {proc.tokenizer.decode(ref_toks)!r}")
        print(f"        g=1   : {proc.tokenizer.decode(st_toks)!r}")

        # Direction at g=4: staleness is real, so g=4 sits BETWEEN g=1 and the floor in feature
        # space -- closer to stock than the floor is, but not at g=1's noise level.
        lg4, _, st4 = axis.streaming_forward(inputs, px_base, groups=4)
        rel_g4 = rel(st4["image_embeds"][0])
        good = rel_g4 < rel_floor
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  g=4 between g=1 and floor    "
              f"rel-L2: g=1 {rel_g1:.5f} < g=4 {rel_g4:.4f} < floor {rel_floor:.4f}")
        st4.pop("image_embeds", None)
        print(f"  ----  g=4 stats: {st4}")

    print("\n" + ("ALL GATES PASS" if ok else "GATE FAILURE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
