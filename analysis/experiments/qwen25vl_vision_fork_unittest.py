"""
qwen25vl_vision_fork_unittest.py

3-tier validation of `appcorr/models/qwen25vl/vision/{attention,block,backbone}.py` against a real
stock forward pass, same protocol used for every prior AppCorr fork this session:

  (a) approx() alone (full stock-equivalent forward) must match a real stock forward, bf16 tolerance.
  (b) approx() on a BLURRED image, then correct() with ALL merge-groups from the TRUE image, must
      also match a stock forward on the true image (single-round 100% correction).
  (c) approx() on a blurred image, then correct() with only HALF the merge-groups, must be CLOSE to
      but not bit-exact vs. the true-image stock forward (expected approximation).
  (d) layer-chunked (matching how GroupTriggerPolicy would drive this, respecting the model's own
      fullatt_block_indexes={7,15,23,31} layer boundaries) approx() must match the one-shot result.

Run for TWO different real images (different aspect ratios / grid_thw) to stress-test the per-image
window-index logic specifically.

Run (appcorr env):
    python analysis/experiments/qwen25vl_vision_fork_unittest.py --model-path Qwen/Qwen2.5-VL-32B-Instruct
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from appcorr.models.qwen25vl.vision.backbone import ApproxCorrectQwen25VLVisionTower


def blur_image(image):
    import numpy as np
    from PIL import Image

    arr = np.array(image).astype("float32")
    small = Image.fromarray(arr.astype("uint8")).resize(
        (max(image.width // 8, 1), max(image.height // 8, 1)), Image.BILINEAR
    )
    return small.resize(image.size, Image.BILINEAR)


def run_case(model, processor, device, image, question, tower, num_layers, label):
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"].to(dtype=torch.bfloat16)
    grid_thw = inputs["image_grid_thw"]

    with torch.no_grad():
        stock_out = model.model.visual(pixel_values, grid_thw=grid_thw)
    stock_merged = stock_out.pooler_output.float()

    num_groups = pixel_values.shape[0] // tower.spatial_merge_unit
    print(f"[unittest][{label}] grid_thw={grid_thw.tolist()} pixel_values={tuple(pixel_values.shape)} "
          f"num_merge_groups={num_groups}")

    with torch.no_grad():
        # (a) approx()-only on the TRUE image should match stock exactly.
        ctx_a = tower.prepare_full_tokens(pixel_values, grid_thw)
        x_a, cache_a = tower.approx_forward(ctx_a["hidden_states"], 0, num_layers, ctx_a, {}, tag_prefix="a")
        merged_a = tower.get_merged_output(x_a, ctx_a).float()
        err_a = (merged_a - stock_merged).abs()
        print(f"[unittest][{label}] (a) approx()-only vs stock: mean_abs_err={err_a.mean().item():.6f} "
              f"max_abs_err={err_a.max().item():.6f}")

        # (b) approx() on a BLURRED image, then correct() with ALL groups from the TRUE image.
        blurred = blur_image(image)
        b_messages = [{"role": "user", "content": [{"type": "image", "image": blurred}, {"type": "text", "text": question}]}]
        b_image_inputs, _ = process_vision_info(b_messages)
        b_inputs = processor(text=[text], images=b_image_inputs, videos=None, padding=True, return_tensors="pt").to(device)
        blurred_pixel_values = b_inputs["pixel_values"].to(dtype=torch.bfloat16)
        assert blurred_pixel_values.shape == pixel_values.shape, "blur must not change grid_thw"

        ctx_b = tower.prepare_full_tokens(blurred_pixel_values, grid_thw)
        x_b, cache_b = tower.approx_forward(ctx_b["hidden_states"], 0, num_layers, ctx_b, {}, tag_prefix="b")

        ctx_b_true = tower.prepare_full_tokens(pixel_values, grid_thw)
        all_group_idx = torch.arange(num_groups, device=device)
        x_b_corrected, cache_b = tower.correct_forward(
            ctx_b_true["hidden_states"], all_group_idx, 0, num_layers, ctx_b_true, cache_b, tag_prefix="b"
        )
        merged_b = tower.get_merged_output(x_b_corrected, ctx_b_true).float()
        err_b = (merged_b - stock_merged).abs()
        print(f"[unittest][{label}] (b) correct(all groups, from blurred approx) vs stock: "
              f"mean_abs_err={err_b.mean().item():.6f} max_abs_err={err_b.max().item():.6f}")

        # (c) approx() on blurred, then correct() with only HALF the groups.
        ctx_c = tower.prepare_full_tokens(blurred_pixel_values, grid_thw)
        x_c, cache_c = tower.approx_forward(ctx_c["hidden_states"], 0, num_layers, ctx_c, {}, tag_prefix="c")
        half_group_idx = torch.arange(0, num_groups, 2, device=device)
        x_c_corrected, cache_c = tower.correct_forward(
            ctx_b_true["hidden_states"], half_group_idx, 0, num_layers, ctx_b_true, cache_c, tag_prefix="c"
        )
        merged_c = tower.get_merged_output(x_c_corrected, ctx_b_true).float()
        err_c = (merged_c - stock_merged).abs()
        print(f"[unittest][{label}] (c) correct(half groups, from blurred approx) vs stock: "
              f"mean_abs_err={err_c.mean().item():.6f} max_abs_err={err_c.max().item():.6f} "
              f"(expected: noticeably larger than (a)/(b), NOT bit-exact)")

        # (d) layer-chunked approx (respecting fullatt_block_indexes boundaries) vs one-shot (a).
        ctx_d = tower.prepare_full_tokens(pixel_values, grid_thw)
        x_d = ctx_d["hidden_states"]
        cache_d = {}
        chunk_bounds = [0, 8, 16, 24, num_layers]
        for s, e in zip(chunk_bounds[:-1], chunk_bounds[1:]):
            x_d, cache_d = tower.approx_forward(x_d, s, e, ctx_d, cache_d, tag_prefix="d")
        merged_d = tower.get_merged_output(x_d, ctx_d).float()
        err_d = (merged_d - merged_a).abs()
        print(f"[unittest][{label}] (d) layer-chunked approx vs one-shot (a): mean_abs_err={err_d.mean().item():.6f} "
              f"max_abs_err={err_d.max().item():.6f}")

    return {
        "ok_a": err_a.max().item() < 0.05,
        "ok_b": err_b.max().item() < 0.05,
        "ok_d": err_d.max().item() < 0.05,
    }


def main():
    import argparse
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from datasets import load_dataset

    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    print(f"[unittest] loading {args.model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    tower = ApproxCorrectQwen25VLVisionTower(model.model.visual).to(args.device).eval()
    num_layers = len(tower.blocks)
    print(f"[unittest] vision tower: {num_layers} layers, fullatt_block_indexes={sorted(tower.fullatt_block_indexes)}")

    ds = load_dataset("lmms-lab/RealWorldQA", split="test")

    all_results = {}
    # Pick two images with different aspect ratios / grid_thw (idx 0 and idx 1 already differ per Phase 0 oracle).
    for i, label in [(0, "img0"), (1, "img1")]:
        ex = ds[i]
        image = ex["image"].convert("RGB")
        results = run_case(model, processor, args.device, image, ex["question"], tower, num_layers, label)
        all_results[label] = results

    print("\n[unittest] === RESULTS ===")
    all_ok = True
    for label, r in all_results.items():
        ok = r["ok_a"] and r["ok_b"] and r["ok_d"]
        all_ok = all_ok and ok
        print(f"    {label}: (a)={'PASS' if r['ok_a'] else 'FAIL'} (b)={'PASS' if r['ok_b'] else 'FAIL'} "
              f"(d)={'PASS' if r['ok_d'] else 'FAIL'}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
