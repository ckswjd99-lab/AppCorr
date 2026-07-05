"""
test_openvla_vision_approx_correct.py

Phase 1 validation (see /home/nxclab/.claude/plans/async-stargazing-mango.md) for the forked
DINOv2/SigLIP blocks in appcorr/models/openvla/vision/. Three tiers, against the Phase 0 oracle
(analysis/experiments/openvla_oracle_dump.py):

  (a) `approx_forward` on the *real* image alone must match stock's used patch features exactly
      (bf16 tolerance) -- this is just a relabeled full forward.
  (b) `approx_forward` on a blurred/low-res version, followed by `correct_forward` with
      patch_idx = *all* patches (using the real image), must *also* match stock exactly --
      100% correction must reduce to a full forward regardless of what the approx pass saw.
  (c) Same as (b) but `correct_forward` only corrects a random subset of patches -- expected to be
      *close* to stock but not exact; this is the accepted approximation error the whole scheme
      is trading off against compute savings, and we report it rather than assert a tight bound.

Run (from repo root, in the `openvla` conda env):
    USE_TF=0 USE_TORCH=1 python analysis/experiments/test_openvla_vision_approx_correct.py \
        --oracle analysis/logs/openvla_oracle/oracle.pt
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from appcorr.models.openvla.vision.backbone import ApproxCorrectViTBackbone


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=str, default=str(REPO_ROOT / "analysis" / "logs" / "openvla_oracle" / "oracle.pt"))
    parser.add_argument("--correct-ratio", type=float, default=0.25, help="Fraction of patches to correct in tier (c).")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def blur_downup(x: torch.Tensor, factor: int = 4) -> torch.Tensor:
    """Simulates a low-res base layer: downsample by `factor` then upsample back (bilinear), per-tower."""
    B, C, H, W = x.shape
    low = F.interpolate(x, size=(H // factor, W // factor), mode="bilinear", align_corners=False)
    return F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)


def report(name: str, pred: torch.Tensor, ref: torch.Tensor):
    pred = pred.float()
    ref = ref.float()
    abs_err = (pred - ref).abs()
    rel_err = abs_err.mean() / (ref.abs().mean() + 1e-8)
    cos = F.cosine_similarity(pred.flatten(), ref.flatten(), dim=0)
    print(f"    [{name}] max_abs_err={abs_err.max().item():.5f} mean_abs_err={abs_err.mean().item():.5f} "
          f"rel_err={rel_err.item():.5f} cos_sim={cos.item():.6f}")
    return abs_err.max().item()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[test] Loading oracle from {args.oracle}...")
    oracle = torch.load(args.oracle, map_location="cpu")

    print(f"[test] Loading {oracle['checkpoint']}...")
    from transformers import AutoModelForVision2Seq

    vla = AutoModelForVision2Seq.from_pretrained(
        oracle["checkpoint"],
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()

    dino_featurizer = vla.vision_backbone.featurizer
    siglip_featurizer = vla.vision_backbone.fused_featurizer

    pixel_values = oracle["pixel_values"].to(device=device, dtype=torch.bfloat16)  # [1, 6, 224, 224]
    dino_px, siglip_px = torch.split(pixel_values, [3, 3], dim=1)

    towers = {
        "dino": (
            ApproxCorrectViTBackbone(dino_featurizer).to(device),
            dino_px,
            oracle["dino_block_outputs"],
        ),
        "siglip": (
            ApproxCorrectViTBackbone(siglip_featurizer).to(device),
            siglip_px,
            oracle["siglip_block_outputs"],
        ),
    }

    overall_max_err = 0.0
    with torch.no_grad():
        for name, (backbone, px, block_outputs) in towers.items():
            print(f"\n[test] === Tower: {name} ===")
            extract_idx = backbone.extract_block_idx
            stock_used = block_outputs[extract_idx][:, backbone.num_prefix_tokens :].to(device=device, dtype=torch.float32)
            num_patches = stock_used.shape[1]
            print(f"    extract_block_idx={extract_idx}, num_prefix_tokens={backbone.num_prefix_tokens}, num_patches={num_patches}")

            # --- Tier (a): approx alone on the real image ---
            cache_a: dict = {}
            patch_feat_a, cache_a = backbone.approx_forward(px, cache_a, tag_prefix=f"{name}_a")
            err_a = report("tier-a approx(real)==stock", patch_feat_a, stock_used)
            overall_max_err = max(overall_max_err, err_a)

            # --- Tier (b): approx(blurred) then correct(real, all patches) ---
            px_blur = blur_downup(px.float(), factor=4).to(dtype=px.dtype)
            cache_b: dict = {}
            _, cache_b = backbone.approx_forward(px_blur, cache_b, tag_prefix=f"{name}_b")
            all_patch_idx = torch.arange(num_patches, device=device)
            patch_feat_b, cache_b = backbone.correct_forward(px, all_patch_idx, cache_b, tag_prefix=f"{name}_b")
            err_b = report("tier-b correct(all)==stock", patch_feat_b, stock_used)
            overall_max_err = max(overall_max_err, err_b)

            # --- Tier (c): approx(blurred) then correct(real, random subset) ---
            cache_c: dict = {}
            _, cache_c = backbone.approx_forward(px_blur, cache_c, tag_prefix=f"{name}_c")
            k = max(1, int(num_patches * args.correct_ratio))
            subset_idx = torch.randperm(num_patches, device=device)[:k]
            patch_feat_c, cache_c = backbone.correct_forward(px, subset_idx, cache_c, tag_prefix=f"{name}_c")
            report(f"tier-c correct({args.correct_ratio:.0%})~=stock (expected nonzero)", patch_feat_c, stock_used)

    print(f"\n[test] Overall max abs error across exactness tiers (a)/(b): {overall_max_err:.5f}")
    if overall_max_err < 0.05:
        print("[test] PASS -- exactness tiers within bf16 tolerance.")
    else:
        print("[test] FAIL -- exactness tiers exceeded tolerance, investigate before proceeding.")


if __name__ == "__main__":
    main()
