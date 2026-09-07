"""Gates for the Qwen2.5-VL in-process axis (`appcorr/models/qwen25vl/unified.py`), the port of
`qwen35_axis_gate.py` to the windowed tower. Same reference (STOCK forward, positions derived by
the model itself), same three checks, plus the one thing this port adds:

Gate 0  row mapping         `_rows_of_groups` inverts the tower's window permutation: gathering
                            the last layer's rows through it and merging must equal the tower's
                            own `get_merged_output` (fp64, so any surviving difference is structural)
Gate 1  merger slice        merger(all)[band] == merger(band rows)
Gate 2  g=1 == stock        one band corrected after full arrival: image embeds within bf16 noise
                            of the stock tower, first greedy token identical
Gate 3  keep<1 runs         g=4 keep=0.5 selects through the row mapping without error and lands
                            between floor and ceiling in embedding distance (direction only)

Run (appcorr env, GPU0):
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python analysis/experiments/qwen25vl_axis_gate.py \
      --model Qwen/Qwen2.5-VL-7B-Instruct
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from appcorr.models.qwen25vl.unified import Qwen25VLAxis, MODEL_ID_7B
from analysis.experiments.qwen35_axis_gate import greedy_from, greedy_stock


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID_7B)
    args = ap.parse_args()
    torch.manual_seed(0)
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="cuda:0").eval()
    axis = Qwen25VLAxis(model, proc)
    print(f"  model: {args.model}  (param dtype: {next(model.parameters()).dtype})")

    rng = np.random.RandomState(0)
    img_arr = np.kron(rng.rand(56, 56, 3), np.ones((8, 8, 1)))         # 448x448, 8px blocks
    img = Image.fromarray((img_arr * 255).astype("uint8"))
    base = img.resize((112, 112)).resize((448, 448))
    q = "How many distinct colored squares are in the top row?"
    inputs = axis.build_inputs(img, q).to("cuda:0")
    px_base = axis.build_inputs(base, q).to("cuda:0")["pixel_values"]

    ok = True
    with torch.no_grad():
        ctx = axis.tower.prepare_full_tokens(inputs["pixel_values"].to(model.dtype),
                                             inputs["image_grid_thw"])
        x, _ = axis.tower.approx_forward(ctx["hidden_states"], 0, len(axis.tower.blocks), ctx, {}, "m")
        unit = axis.tower.spatial_merge_unit
        n_groups = ctx["seq_len"] // unit
        import copy
        merger64 = copy.deepcopy(axis.tower.merger).double()
        x64 = x.double()

        # Gate 0: row mapping == the tower's own un-permutation.
        ref_m = merger64(x64)[ctx["inv_window_index"]]
        rows = axis._rows_of_groups(ctx, torch.arange(n_groups, device=x.device))
        via_m = merger64(x64[rows])
        d0 = (ref_m - via_m).abs().max().item()
        good = d0 < 1e-12
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  row mapping == get_merged_output (fp64)  max|diff| = {d0:.3e}")

        # Gate 1: per-band merger slicing through the mapping.
        band_rows = axis._rows_of_groups(ctx, torch.arange(3, 9, device=x.device))
        d1 = (ref_m[3:9] - merger64(x64[band_rows])).abs().max().item()
        good = d1 < 1e-12
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  merger slice exact (fp64)   max|diff| = {d1:.3e}")

        # Gate 2: g=1 streaming vs stock.
        ref_logits = axis.full_forward(inputs)
        fl_logits = axis.approx_only_forward(inputs, px_base)
        ref_toks = greedy_stock(axis, inputs)
        lg1, kv1, st1 = axis.streaming_forward(inputs, px_base, groups=1)
        st_toks = greedy_from(axis, lg1, kv1, st1["decode_start_pos"])

        def dist(a, b):
            return (a.float().softmax(-1) - b.float().softmax(-1)).abs().sum().item() / 2

        tv_floor, tv_g1 = dist(fl_logits, ref_logits), dist(lg1, ref_logits)
        e_ref = axis.model.model.visual(inputs["pixel_values"].to(axis.model.dtype),
                                        grid_thw=inputs["image_grid_thw"]).pooler_output.float()
        e_floor = axis.model.model.visual(px_base.to(axis.model.dtype),
                                          grid_thw=inputs["image_grid_thw"]).pooler_output.float()

        def rel(e):
            return ((e - e_ref).norm() / e_ref.norm()).item()

        rel_floor, rel_g1 = rel(e_floor), rel(st1["image_embeds"][0])
        good = rel_floor > 0.05
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  degradation is informative   rel-L2(floor) = {rel_floor:.4f}")
        good = rel_g1 < rel_floor / 20
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  g=1 embeds == stock vision   rel-L2(g=1) = {rel_g1:.5f}")
        first_same = st_toks[0] == ref_toks[0]
        ok &= first_same
        print(f"  {'PASS' if first_same else 'FAIL'}  g=1 first token == stock     "
              f"TV(g=1)={tv_g1:.4f} TV(floor)={tv_floor:.4f} [diagnostic]")
        print(f"        stock : {proc.tokenizer.decode(ref_toks)!r}")
        print(f"        g=1   : {proc.tokenizer.decode(st_toks)!r}")

        # Gate 3: g=4 at keep 1.0 and 0.5 -- selection through the row mapping; direction only.
        for keep in (1.0, 0.5):
            lg4, kv4, st4 = axis.streaming_forward(inputs, px_base, groups=4, keep=keep)
            r4 = rel(st4["image_embeds"][0])
            toks4 = greedy_from(axis, lg4, kv4, st4["decode_start_pos"])
            good = r4 < rel_floor
            ok &= good
            print(f"  {'PASS' if good else 'FAIL'}  g=4 keep={keep}: rel-L2 = {r4:.4f} < floor {rel_floor:.4f}; "
                  f"corrected {st4['corrected_groups']}/{n_groups} groups, chunks {st4['chunks']}")
            print(f"        g=4   : {proc.tokenizer.decode(toks4)!r}")
    print("QWEN25VL_AXIS_GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
