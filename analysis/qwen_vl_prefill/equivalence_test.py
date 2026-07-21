"""
equivalence_test.py -- FIRST MILESTONE.

Proves that chunked prefill (feeding the Qwen2.5-VL LLM the sequence in contiguous chunks and
appending each to one KV cache) reproduces monolithic prefill (whole sequence at once), given
identical visual embeddings and identical M-RoPE position_ids. This is the correctness foundation
for progressive visual-token streaming; nothing downstream (oracle streaming, ProgVFM correction)
is valid until this passes.

Run (appcorr env):
    python qwen_vl_prefill/equivalence_test.py \
        --model-id Qwen/Qwen2.5-VL-3B-Instruct --num-groups 4 --device cuda:0
    # (default image: RefCOCO val[0]; pass --image PATH --prompt "..." to override)

Reports: exact model class, where visual embeddings are extracted, how position_ids/cache_position
are handled, max/mean logit difference (monolithic vs chunked), and a timing breakdown.
"""

import argparse
import sys
from pathlib import Path

import torch

for _p in Path(__file__).resolve().parents[1:3]:  # analysis/ (qwen_vl_prefill) + repo root (appcorr)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qwen_vl_prefill import introspect as I
from qwen_vl_prefill import prefill as P


class CudaTimer:
    """GPU wall-time via CUDA events (synchronizes)."""
    def __init__(self, device):
        self.device = device
        self.events = {}

    def time(self, name, fn):
        torch.cuda.synchronize(self.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize(self.device)
        self.events[name] = start.elapsed_time(end)  # ms
        return out


def load_default_image():
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/RefCOCO", split="val")
    ex = ds[0]
    img = ex["image"].convert("RGB")
    expr = ex["answer"][0] if isinstance(ex["answer"], list) and ex["answer"] else str(ex["answer"])
    prompt = (f'Locate the region described by: "{expr}". Output ONLY the bounding box as four '
              f"numbers x1,y1,x2,y2 (top-left and bottom-right pixel coordinates in this image), "
              f"with no other text.")
    return img, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--image", default=None, help="image path; default = RefCOCO val[0]")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--num-groups", type=int, default=4, help="G visual-token groups for chunked prefill")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                    help="fp32 isolates whether residual diff is pure bf16 accumulation (mechanism-correct) vs a real bug")
    args = ap.parse_args()

    device = args.device
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"[equiv] loading {args.model_id} (dtype={args.dtype}) ...")
    model, processor = I.load_model(args.model_id, device=device, dtype=dtype)
    print(f"[equiv] model class: {type(model).__name__}  "
          f"(text: {type(model.model.language_model).__name__}, "
          f"vision: {type(model.model.visual).__name__})")

    if args.image is not None:
        from PIL import Image
        image = Image.open(args.image).convert("RGB")
        prompt = args.prompt or "Describe this image."
    else:
        image, default_prompt = load_default_image()
        prompt = args.prompt or default_prompt

    prepared = I.prepare_inputs(model, processor, image, prompt, device=device)
    print(f"[equiv] seq_len={prepared.seq_len}, n_visual_tokens={prepared.n_visual_tokens}, "
          f"image_grid_thw={prepared.image_grid_thw.tolist()}")

    timer = CudaTimer(device)
    visual_embeds = timer.time("visual_encoder", lambda: I.extract_visual_embeds(model, prepared))
    print(f"[equiv] visual embeds extracted via model.model.get_image_features(...).pooler_output "
          f"-> shape {tuple(visual_embeds.shape)}")

    inputs_embeds = I.build_inputs_embeds(model, prepared, visual_embeds)
    position_ids = I.compute_position_ids(model, prepared)
    print(f"[equiv] position_ids shape {tuple(position_ids.shape)} (M-RoPE [3,B,T], computed once, sliced per chunk)")

    layout = I.token_layout(prepared)
    boundaries = I.streaming_chunk_boundaries(layout, args.num_groups)
    print(f"[equiv] token layout: pre_text[0:{layout.img_start}] "
          f"image[{layout.img_start}:{layout.img_end}] post_text[{layout.img_end}:{layout.post_end}]")
    print(f"[equiv] chunk boundaries ({len(boundaries)} chunks): "
          + ", ".join(f"{lbl}[{a}:{b}]" for a, b, lbl in boundaries))

    mono_logits, _ = timer.time("monolithic_prefill", lambda: P.monolithic_prefill(model, inputs_embeds, position_ids))
    chunk_logits, _, _ = timer.time("chunked_prefill", lambda: P.chunked_prefill(model, inputs_embeds, position_ids, boundaries))

    cmp = P.compare_logits(mono_logits, chunk_logits)

    print("\n========== EQUIVALENCE REPORT ==========")
    print(f"model class:              {type(model).__name__}")
    print(f"visual embed extraction:  model.model.get_image_features(pixel_values, image_grid_thw).pooler_output")
    print(f"                          (post-merger, LLM-order; {prepared.n_visual_tokens} tokens, hidden={visual_embeds.shape[-1]})")
    print(f"position_ids handling:    get_rope_index -> [3,1,{prepared.seq_len}] M-RoPE, computed once, sliced per chunk")
    print(f"cache_position handling:  arange(a,b) per chunk into one DynamicCache (attention_mask=None, pure causal)")
    print(f"logits shape:             {cmp['shape']}")
    print(f"max abs logit diff:       {cmp['max_abs_diff']:.6e}")
    print(f"mean abs logit diff:      {cmp['mean_abs_diff']:.6e}")
    print(f"max rel logit diff:       {cmp['max_rel_diff']:.6e}")
    print(f"per-position argmax agree: {cmp['argmax_agreement_frac']*100:.4f}%")
    print(f"last-token argmax match:  {cmp['last_token_argmax_match']}")
    print("\n---------- TIMING (ms) ----------")
    for k, v in timer.events.items():
        print(f"  {k:20s}: {v:8.3f} ms")
    tol = 5e-2  # bf16 chunked-vs-monolithic tolerance; tighten/loosen after first run
    verdict = "PASS" if cmp["max_abs_diff"] < tol and cmp["last_token_argmax_match"] else "CHECK"
    print(f"\nVERDICT: {verdict} (tolerance max_abs_diff < {tol}, last-token argmax must match)")


if __name__ == "__main__":
    main()
