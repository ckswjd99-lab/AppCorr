"""
accuracy_impact.py -- measure the ACCURACY impact of the vision-side degradation, BEFORE any
latency work. Self-contained (no dependency on the offload/experiment-branch code).

The progressive-prefill scheme's only accuracy risk is on the VISION side: visual tokens are
produced from partial (progressively-arriving) image information and frozen early (bidirectional
staleness), instead of the full image. This script brackets that impact on a real downstream task
(RefCOCO referring-expression grounding, exact-match IoU>0.5):

  - full   : stock model on the FULL image           -> baseline accuracy (upper bound)
  - base   : stock model on a low-frequency BASE only -> vision-side worst case (no residuals)
             (progressive finalization sits BETWEEN these -- late groups are near-full quality)

If base-only barely moves accuracy, progressive finalization is safe. If it drops a lot, the
finalization timing (which groups get residuals when) matters and must be measured per-group.

The base is a low-pass proxy for the Laplacian pyramid base: downsample by --base-factor then
upsample back (bilinear), matching ProgVFM's coarse base.

Run:
    python qwen_vl_prefill/accuracy_impact.py --model-id Qwen/Qwen2.5-VL-3B-Instruct \
        --num-samples 64 --base-factor 2 --device cuda:0
"""

import argparse
import re
import sys
from pathlib import Path

import torch

for _p in Path(__file__).resolve().parents[1:3]:  # analysis/ (qwen_vl_prefill) + repo root (appcorr)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qwen_vl_prefill import introspect as I

GROUNDING_PROMPT = (
    'Locate the region described by: "{expr}". Output ONLY the bounding box as four numbers '
    "x1,y1,x2,y2 (top-left and bottom-right pixel coordinates in this image), with no other text."
)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_bbox(text):
    nums = NUM_RE.findall(text)
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in nums[:4])
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def iou(a, b):
    if a is None:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def make_base(image_pil, factor):
    """Low-frequency base: downsample by `factor` then upsample back (bilinear). A proxy for the
    Laplacian pyramid base. factor=1 returns the image unchanged."""
    if factor <= 1:
        return image_pil
    from PIL import Image
    w, h = image_pil.size
    small = image_pil.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


@torch.inference_mode()
def answer(model, processor, image_pil, prompt, device, max_new_tokens=48):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image_pil], return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--base-factor", type=int, default=2, help="downsample factor for the low-freq base")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device

    print(f"[acc] loading {args.model_id} ...")
    model, processor = I.load_model(args.model_id, device=device)
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset("lmms-lab/RefCOCO", split="val")
    n_total = len(ds)
    stride = max(n_total // args.num_samples, 1)
    indices = list(range(0, n_total, stride))[:args.num_samples]
    ip = processor.image_processor
    factor = ip.patch_size * ip.merge_size * 4
    min_px, max_px = ip.size["shortest_edge"], ip.size["longest_edge"]

    full_correct = base_correct = 0
    full_iou = base_iou = 0.0
    for c, idx in enumerate(indices):
        ex = ds[idx]
        image = ex["image"].convert("RGB")
        expr = ex["answer"][0] if isinstance(ex["answer"], list) and ex["answer"] else str(ex["answer"])
        bx, by, bw, bh = ex["bbox"]
        # scale GT to the model's smart-resized frame (the coords the model outputs in)
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        sx, sy = tw / image.width, th / image.height
        gt = (bx * sx, by * sy, (bx + bw) * sx, (by + bh) * sy)
        image_r = image.resize((tw, th), Image.BILINEAR)
        prompt = GROUNDING_PROMPT.format(expr=expr)

        pred_full = parse_bbox(answer(model, processor, image_r, prompt, device))
        pred_base = parse_bbox(answer(model, processor, make_base(image_r, args.base_factor), prompt, device))
        i_full, i_base = iou(pred_full, gt), iou(pred_base, gt)
        full_iou += i_full; base_iou += i_base
        full_correct += int(i_full > 0.5); base_correct += int(i_base > 0.5)
        if (c + 1) % 16 == 0:
            print(f"  [{c+1}/{len(indices)}] full acc={100*full_correct/(c+1):.1f}% "
                  f"base acc={100*base_correct/(c+1):.1f}%", flush=True)

    n = len(indices)
    print(f"\n========== VISION-SIDE ACCURACY IMPACT (RefCOCO, N={n}) ==========")
    print(f"  full image  : acc@0.5 = {100*full_correct/n:.2f}%  ({full_correct}/{n})  mean_iou={full_iou/n:.4f}")
    print(f"  base only   : acc@0.5 = {100*base_correct/n:.2f}%  ({base_correct}/{n})  mean_iou={base_iou/n:.4f}  "
          f"(downsample factor {args.base_factor})")
    print(f"  delta (base - full): {100*(base_correct-full_correct)/n:+.2f}pp  "
          f"mean_iou {(base_iou-full_iou)/n:+.4f}")
    print(f"\n  Interpretation: base-only is the vision-side WORST case (all tokens from the coarse base).")
    print(f"  Progressive finalization sits between full and base -- later visual groups get more")
    print(f"  residuals and approach full quality, so its accuracy is >= base-only.")


if __name__ == "__main__":
    main()
