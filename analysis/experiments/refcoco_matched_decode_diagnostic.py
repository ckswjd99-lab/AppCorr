"""
refcoco_matched_decode_diagnostic.py

Diagnostic requested directly by the user: `head_inference` (the corrected-pipeline path) decodes
only the FIRST token from the corrected hidden state, then falls back to a SEPARATE stock
`model.generate()` call on `[input_ids, first_token]` for any remaining tokens. `full_inference`
(the true baseline) instead makes ONE continuous `model.generate()` call for the whole answer.
These are two different generation *mechanisms*, even when the underlying computation is otherwise
identical -- for RefCOCO's multi-token bbox-coordinate answers, a two-stage decode could in
principle behave differently from one continuous decode purely from floating-point
non-associativity between "recompute full prefix from scratch each call" (two-stage) vs "one
incremental decode with an internal KV cache" (continuous), independent of any actual correction
quality. This script isolates that variable: it runs 100% STOCK (unforked) computation both ways --
(a) true baseline: one continuous model.generate() call, (b) matched-decode: forward pass -> argmax
first token -> model.generate() fallback, exactly mirroring head_inference's mechanism -- on the
SAME samples, SAME full-resolution image, SAME prompt construction. If (a) and (b) differ noticeably
even though both are pure stock computation, that confirms the generation-mechanism itself is a
real confound in the "does correction match baseline" comparisons; if they match, the earlier
"100% keep_rate sometimes exceeds baseline" gaps are NOT explained by this mechanism and must come
from the correction pipeline's own numerical divergence.

No offload pipeline involved -- loads the model directly for speed/simplicity.

Run (appcorr env):
    python analysis/experiments/refcoco_matched_decode_diagnostic.py --model-path Qwen/Qwen2.5-VL-32B-Instruct --num-samples 400 --device cuda:0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.experiments.refcoco_offload_eval import GROUNDING_PROMPT_TMPL, score_answer

MERGE_UNIT = 4  # spatial_merge_size(2)**2, constant for Qwen2.5-VL


def build_prompt(processor, question, grid_thw, image_token_id, device):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    num_image_tokens = int((grid_thw.prod(dim=-1) // MERGE_UNIT).sum().item())
    image_pad = "<|image_pad|>" * num_image_tokens
    text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", f"<|vision_start|>{image_pad}<|vision_end|>")
    tok_out = processor.tokenizer(text, return_tensors="pt")
    input_ids = tok_out["input_ids"].to(device)
    attention_mask = tok_out["attention_mask"].to(device)
    return input_ids, attention_mask


def true_baseline_generate(model, processor, input_ids, attention_mask, pixel_values, grid_thw):
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=grid_thw,
            max_new_tokens=64, do_sample=False,
        )
    trimmed = gen_ids[:, input_ids.shape[1]:]
    return processor.tokenizer.decode(trimmed[0], skip_special_tokens=True)


def matched_decode_generate(model, processor, input_ids, attention_mask, pixel_values, grid_thw):
    """Mirrors head_inference exactly, but the prefill hidden state comes from a plain STOCK
    forward pass (no correction fork at all) -- isolates the two-stage decode mechanism."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=grid_thw,
            use_cache=False,
        )
        logits_last = outputs.logits[:, -1, :]
        first_token = logits_last.argmax(dim=-1)

        if first_token.item() == processor.tokenizer.eos_token_id:
            return ""

        extended_ids = torch.cat([input_ids, first_token.unsqueeze(0)], dim=1)
        extended_mask = torch.cat([attention_mask, torch.ones_like(first_token.unsqueeze(0))], dim=1)
        gen_ids = model.generate(
            input_ids=extended_ids, attention_mask=extended_mask,
            pixel_values=pixel_values, image_grid_thw=grid_thw,
            max_new_tokens=63, do_sample=False,
        )
    trimmed = gen_ids[:, input_ids.shape[1]:]
    return processor.tokenizer.decode(trimmed[0], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=400)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--label", type=str, default="matched_decode_diagnostic")
    args = p.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset

    print(f"[diag] loading {args.model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    image_token_id = model.config.image_token_id
    ip = processor.image_processor
    min_pixels, max_pixels = ip.size["shortest_edge"], ip.size["longest_edge"]
    factor = ip.patch_size * ip.merge_size * 4

    print("[diag] loading lmms-lab/RefCOCO (val split) ...")
    ds = load_dataset("lmms-lab/RefCOCO", split="val")
    n_total = len(ds)
    n_samples = min(args.num_samples, n_total)
    stride = max(n_total // n_samples, 1)
    indices = list(range(0, n_total, stride))[:n_samples]
    print(f"[diag] sampling {len(indices)} examples strided across {n_total} (stride={stride})")

    correct_baseline = correct_matched = 0
    iou_sum_baseline = iou_sum_matched = 0.0
    agree_count = 0
    t_start = time.time()

    for i, idx in enumerate(indices):
        ex = ds[idx]
        image = ex["image"].convert("RGB")
        expr = ex["answer"][0] if isinstance(ex["answer"], list) and ex["answer"] else str(ex["answer"])
        bx, by, bw, bh = ex["bbox"]
        gt_box_orig = (bx, by, bx + bw, by + bh)
        prompt = GROUNDING_PROMPT_TMPL.format(expr=expr)

        orig_w, orig_h = image.width, image.height
        target_h, target_w = smart_resize(orig_h, orig_w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
        resized = image.resize((target_w, target_h), Image.BILINEAR)
        sx, sy = target_w / orig_w, target_h / orig_h
        gt_box_resized = (gt_box_orig[0] * sx, gt_box_orig[1] * sy, gt_box_orig[2] * sx, gt_box_orig[3] * sy)

        proc_out = processor.image_processor(images=[np.array(resized, dtype=np.uint8)], return_tensors="pt")
        pixel_values = proc_out["pixel_values"].to(device=args.device, dtype=torch.bfloat16)
        grid_thw = proc_out["image_grid_thw"].to(args.device)

        input_ids, attention_mask = build_prompt(processor, prompt, grid_thw, image_token_id, args.device)

        pred_baseline = true_baseline_generate(model, processor, input_ids, attention_mask, pixel_values, grid_thw)
        pred_matched = matched_decode_generate(model, processor, input_ids, attention_mask, pixel_values, grid_thw)

        ok_b, iou_b = score_answer(pred_baseline, gt_box_resized)
        ok_m, iou_m = score_answer(pred_matched, gt_box_resized)
        correct_baseline += int(ok_b)
        correct_matched += int(ok_m)
        iou_sum_baseline += iou_b
        iou_sum_matched += iou_m
        agree_count += int(pred_baseline.strip() == pred_matched.strip())

        if (i + 1) % 20 == 0 or (i + 1) == len(indices):
            n = i + 1
            print(f"    [{n}/{len(indices)}] baseline_acc={100*correct_baseline/n:.2f}% "
                  f"matched_acc={100*correct_matched/n:.2f}% agree={100*agree_count/n:.1f}% "
                  f"elapsed={time.time()-t_start:.0f}s")
            sys.stdout.flush()

    n = len(indices)
    print(f"\n[diag] === Summary: {args.label} ===")
    print(f"    samples: {n}")
    print(f"    true_baseline (1 continuous generate()):  Acc@0.5={100*correct_baseline/n:.2f}% "
          f"({correct_baseline}/{n})  mean_iou={iou_sum_baseline/n:.4f}")
    print(f"    matched_decode (forward+argmax+fallback): Acc@0.5={100*correct_matched/n:.2f}% "
          f"({correct_matched}/{n})  mean_iou={iou_sum_matched/n:.4f}")
    print(f"    exact text agreement between the two mechanisms: {100*agree_count/n:.1f}% ({agree_count}/{n})")
    print(f"    gap (matched - true_baseline): {100*(correct_matched-correct_baseline)/n:+.2f}pp")


if __name__ == "__main__":
    main()
