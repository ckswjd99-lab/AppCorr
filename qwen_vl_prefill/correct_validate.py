"""
correct_validate.py -- validate the ApproxCorrectQwen25VLVisionTower fork STANDALONE in the
prototype pipeline before trusting any Phase 5 accuracy number. Two tiers (vs the STOCK model):

  A. approx(base)            ~=  stock get_image_features(base)   [approx path + merge un-permute]
  B. correct-all-one-round   ~=  stock get_image_features(full)   [correct path, full refresh]

bf16 tolerance expected; the meaningful check is that (B) collapses the base->full gap to accumulation
noise while base_only stays far from full.

Run: python qwen_vl_prefill/correct_validate.py --model-id Qwen/Qwen2.5-VL-3B-Instruct --device cuda:0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_vl_prefill import introspect as I
from qwen_vl_prefill import progressive as PR
from qwen_vl_prefill import progressive_correct as PC


def rel(a, b):
    d = (a - b).abs()
    return d.max().item(), d.mean().item(), (b.abs().mean().item() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--base-factor", type=int, default=4)
    args = ap.parse_args()
    device = args.device

    print(f"[validate] loading {args.model_id} ...")
    model, processor = I.load_model(args.model_id, device=device)
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset
    from PIL import Image

    ip = processor.image_processor
    factor = ip.patch_size * ip.merge_size * 4
    min_px, max_px = ip.size["shortest_edge"], ip.size["longest_edge"]
    ds = load_dataset("lmms-lab/RealWorldQA", split="test")
    tower = PC.build_tower(model)

    for idx in (0, 100, 400):
        ex = ds[idx]
        image = ex["image"].convert("RGB")
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        image_r = image.resize((tw, th), Image.BILINEAR)
        full_np = np.array(image_r, dtype=np.uint8)
        base_np = PR.make_base(full_np, args.base_factor)

        ca, full, base, stock_base = PC.correct_all_one_round(model, tower, processor, full_np, base_np, device)
        mx_a, mn_a, scale_b = rel(base, stock_base)
        mx_b, mn_b, scale_f = rel(ca, full)
        gap_bf = (base - full).abs().mean().item()
        print(f"\n[idx {idx}] grid {th}x{tw}, {full.shape[0]} visual tokens")
        print(f"  A approx(base) vs stock(base): max={mx_a:.4e} mean={mn_a:.4e}  (token scale ~{scale_b:.3f})")
        print(f"  B correct-all  vs stock(full): max={mx_b:.4e} mean={mn_b:.4e}  (token scale ~{scale_f:.3f})")
        print(f"    base_only vs full gap (mean abs): {gap_bf:.4e}  -> correct-all recovers it to {mn_b:.4e}")


if __name__ == "__main__":
    main()
