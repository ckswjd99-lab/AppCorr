"""
reverse_order_parity.py -- sanity gate for reverse_order_correct.py.

`reverse_order_prefill` is NOT exactness-preserving in general (see its module docstring): a group
corrected while spatially-earlier groups are still base permanently bakes that staleness into its own
residual stream. The one case where it MUST reduce to exact monolithic prefill is `num_groups=1`
(the whole image is a single "group", so its own [0,L) correction, combined with post_text's [0,L)
correction in the same final round, has full true-causal context for everything before it -- see the
docstring's "Exactness DOES hold in one degenerate case" section). Verify THIS before trusting any
num_groups>1 accuracy number.

Run:
    conda activate appcorr
    python analysis/qwen_vl_prefill/reverse_order_parity.py \
        --model-id Qwen/Qwen2.5-VL-3B-Instruct --device cuda:0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

for _p in Path(__file__).resolve().parents[1:3]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qwen_vl_prefill import introspect as I
from qwen_vl_prefill import progressive as PR
from qwen_vl_prefill import progressive_correct as PC
from qwen_vl_prefill import reverse_order_correct as RC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-groups", type=int, default=1, help="must be 1 for the exactness gate")
    ap.add_argument("--base-factor", type=int, default=4)
    ap.add_argument("--image", default=None, help="default: RefCOCO val[0]")
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()
    device = args.device

    print(f"[parity] loading {args.model_id} ...")
    model, processor = I.load_model(args.model_id, device=device)
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    ip = processor.image_processor
    merge_size, patch_size = ip.merge_size, ip.patch_size
    factor = patch_size * merge_size * 4
    min_px, max_px = ip.size["shortest_edge"], ip.size["longest_edge"]

    if args.image is None:
        from datasets import load_dataset
        from qwen_vl_prefill.datasets_eval import RefCOCOSpec, GROUNDING_PROMPT
        spec = RefCOCOSpec()
        ds = spec.load(load_dataset)
        ex = ds[0]
        image_native = ex["image"].convert("RGB")
        expr = ex["answer"][0] if isinstance(ex["answer"], list) and ex["answer"] else str(ex["answer"])
        prompt = GROUNDING_PROMPT.format(expr=expr)
    else:
        from PIL import Image
        image_native = Image.open(args.image).convert("RGB")
        prompt = args.prompt or "Describe this image."

    full_np, base_np = RC.native_pyramid(image_native, args.base_factor, smart_resize, factor, min_px, max_px)
    from PIL import Image as _Image
    image_r = _Image.fromarray(full_np)

    prepared = I.prepare_inputs(model, processor, image_r, prompt, device=device)
    position_ids = I.compute_position_ids(model, prepared)
    bands = PR.band_layout(prepared.image_grid_thw, merge_size, patch_size, args.num_groups)
    print(f"[parity] num_groups={args.num_groups} (bands actually produced: {len(bands)}), "
          f"seq_len={prepared.seq_len}, n_visual_tokens={prepared.n_visual_tokens}")

    tower = PC.build_tower(model)
    llm_layers = RC.build_llm_layers(model)
    L = len(llm_layers)
    print(f"[parity] L (text-decoder layers) = {L}")

    # ---- reference: stock monolithic prefill on the TRUE full image ----
    from qwen_vl_prefill import prefill as PF
    with torch.inference_mode():
        full_embeds = I.extract_visual_embeds(model, prepared)
        inputs_embeds = I.build_inputs_embeds(model, prepared, full_embeds)
        ref_logits, _ = PF.monolithic_prefill(model, inputs_embeds, position_ids)
        ref_last = ref_logits[:, -1].clone()  # [1, vocab]

        # ---- reverse_order_prefill (num_groups=1 -> should be exact) ----
        last_hidden, cache_feature, layout = RC.reverse_order_prefill(
            model, tower, llm_layers, processor, prepared, position_ids, full_np, base_np, bands, device
        )
        test_logits = model.lm_head(last_hidden)[:, -1].clone()  # [1, vocab]

    diff = (ref_last.float() - test_logits.float()).abs()
    max_abs = diff.max().item()
    argmax_match = bool(torch.equal(ref_last.argmax(-1), test_logits.argmax(-1)))
    print(f"\n[parity] max abs logit diff (last position): {max_abs:.6e}")
    print(f"[parity] next-token argmax match: {argmax_match}")
    tol = 1e-2 if model.dtype == torch.float32 else 5.0  # bf16 accumulation-order noise, see equivalence_test.py
    ok = max_abs < tol and argmax_match
    print(f"\n[parity] {'PASS' if ok else 'FAIL'} (tol={tol}, dtype={model.dtype})")
    if not ok:
        print("[parity] FAIL -- do not trust num_groups>1 accuracy numbers until this passes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
