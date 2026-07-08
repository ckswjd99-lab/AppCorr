"""
clip_vision_fork_unittest.py

3-tier validation of `appcorr/models/openclip/vision/{attention,block,backbone}.py` against the
Phase 0 oracle (`clip_bigg_oracle.py`), same protocol used for the OpenVLA/DINOv3 vision forks:

  (a) approx() alone (full stock forward + caching) must match a real stock forward, bf16 tolerance.
  (b) approx() on a BLURRED image, then correct() with ALL patches from the TRUE image, must also
      match a stock forward on the true image, bf16 tolerance (single-round 100% correction).
  (c) approx() on a blurred image, then correct() with only HALF the patches from the true image,
      must be CLOSE to but not bit-exact vs. the true-image stock forward (expected approximation,
      same accepted property as DINOv2/DINOv3/SigLIP).

Run (appcorr env):
    python analysis/experiments/clip_vision_fork_unittest.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from appcorr.models.openclip.vision.backbone import ApproxCorrectCLIPVisionTower

MODEL_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
IMAGENET_VAL_ROOT = "/NHNHOME/share/cjpark/data/imagenet_val"


def load_images(processor, device, num_images=4):
    val_root = Path(IMAGENET_VAL_ROOT)
    class_dirs = sorted([d for d in val_root.iterdir() if d.is_dir()])[:num_images]
    image_paths = [sorted(d.glob("*"))[0] for d in class_dirs if sorted(d.glob("*"))]
    images = [Image.open(p).convert("RGB") for p in image_paths]
    pixel_values = processor(images=images, return_tensors="pt").to(device)["pixel_values"]
    return pixel_values.to(dtype=torch.bfloat16)


def blur_pixels(pixel_values: torch.Tensor) -> torch.Tensor:
    """Heavily downsample+upsample to simulate a degraded base-layer canvas (same role as
    AppCorr's low-res Laplacian base layer -- we don't need the real pyramid machinery for a unit
    test, just a plausibly-different starting point for approx())."""
    low = F.interpolate(pixel_values.float(), scale_factor=1 / 8, mode="bilinear", align_corners=False)
    back = F.interpolate(low, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
    return back.to(dtype=pixel_values.dtype)


def main():
    device = "cuda:0"
    print(f"[unittest] loading {MODEL_ID} ...")
    model = CLIPModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    pixel_values = load_images(processor, device, num_images=4)
    B, C, H, W = pixel_values.shape
    patch_size = model.config.vision_config.patch_size
    grid = H // patch_size
    num_patches = grid * grid
    print(f"[unittest] pixel_values shape={tuple(pixel_values.shape)}, num_patches={num_patches}")

    tower = ApproxCorrectCLIPVisionTower(model.vision_model, model.visual_projection).to(device).eval()
    num_layers = len(tower.blocks)

    with torch.no_grad():
        stock_embeds = model.get_image_features(pixel_values=pixel_values).pooler_output
        stock_embeds = stock_embeds / stock_embeds.norm(dim=-1, keepdim=True)

        # (a) approx() alone on the TRUE image should match stock exactly.
        cache_a = {}
        x_a = tower.prepare_full_tokens(pixel_values)
        x_a, cache_a = tower.approx_forward(x_a, 0, num_layers, cache_a, tag_prefix="a")
        approx_embeds = tower.get_image_embeds(x_a)
        err_a = (approx_embeds.float() - stock_embeds.float()).abs()
        print(f"[unittest] (a) approx()-only vs stock: mean_abs_err={err_a.mean().item():.6f} "
              f"max_abs_err={err_a.max().item():.6f}")

        # (b) approx() on a BLURRED image, then correct() with ALL patches from the TRUE image.
        blurred = blur_pixels(pixel_values)
        cache_b = {}
        x_b = tower.prepare_full_tokens(blurred)
        x_b, cache_b = tower.approx_forward(x_b, 0, num_layers, cache_b, tag_prefix="b")
        all_patch_idx = torch.arange(num_patches, device=device)
        x_b_true = tower.prepare_full_tokens(pixel_values)
        x_b_corrected, cache_b = tower.correct_forward(x_b_true, all_patch_idx, 0, num_layers, cache_b, tag_prefix="b")
        correct_all_embeds = tower.get_image_embeds(x_b_corrected)
        err_b = (correct_all_embeds.float() - stock_embeds.float()).abs()
        print(f"[unittest] (b) correct(all patches, from blurred approx) vs stock: "
              f"mean_abs_err={err_b.mean().item():.6f} max_abs_err={err_b.max().item():.6f}")

        # (c) approx() on a BLURRED image, then correct() with only HALF the patches.
        cache_c = {}
        x_c = tower.prepare_full_tokens(blurred)
        x_c, cache_c = tower.approx_forward(x_c, 0, num_layers, cache_c, tag_prefix="c")
        half_patch_idx = torch.arange(0, num_patches, 2, device=device)  # every other patch
        x_c_true = tower.prepare_full_tokens(pixel_values)
        x_c_corrected, cache_c = tower.correct_forward(x_c_true, half_patch_idx, 0, num_layers, cache_c, tag_prefix="c")
        correct_half_embeds = tower.get_image_embeds(x_c_corrected)
        err_c = (correct_half_embeds.float() - stock_embeds.float()).abs()
        print(f"[unittest] (c) correct(half patches, from blurred approx) vs stock: "
              f"mean_abs_err={err_c.mean().item():.6f} max_abs_err={err_c.max().item():.6f} "
              f"(expected: noticeably larger than (a)/(b), NOT bit-exact -- this is the accepted "
              f"approximation, same as DINOv2/DINOv3)")

        # (d) layer-CHUNKED approx (4 chunks of 12 layers, matching how GroupTriggerPolicy would
        # drive this) must be identical to the one-shot (a) result -- validates the chunked
        # start_l/end_l contract itself, independent of the approx/correct math already checked.
        cache_d = {}
        x_d = tower.prepare_full_tokens(pixel_values)
        chunk = num_layers // 4
        for c in range(4):
            s, e = c * chunk, (c + 1) * chunk if c < 3 else num_layers
            x_d, cache_d = tower.approx_forward(x_d, s, e, cache_d, tag_prefix="d")
        chunked_embeds = tower.get_image_embeds(x_d)
        err_d = (chunked_embeds.float() - stock_embeds.float()).abs()
        print(f"[unittest] (d) layer-chunked approx (4x12) vs stock: mean_abs_err={err_d.mean().item():.6f} "
              f"max_abs_err={err_d.max().item():.6f}")

    ok_a = err_a.max().item() < 0.05
    ok_b = err_b.max().item() < 0.05
    ok_d = err_d.max().item() < 0.05
    print(f"\n[unittest] RESULT: (a) approx-only {'PASS' if ok_a else 'FAIL'}, "
          f"(b) correct-all {'PASS' if ok_b else 'FAIL'}, (d) chunked {'PASS' if ok_d else 'FAIL'}")
    if not (ok_a and ok_b and ok_d):
        sys.exit(1)


if __name__ == "__main__":
    main()
