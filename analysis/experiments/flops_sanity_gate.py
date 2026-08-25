"""Cross-check every measured ceiling against a closed-form FLOP estimate.

The bug this exists to catch: `backbone_modules()` returning `[None]` because it guessed at an
attribute the model does not have. `install()` used to skip a None root silently, and
`patch_attention` is global, so attention kept being counted -- the arm reported a smaller but
entirely plausible number rather than zero. It sat in THREE of the four DINOv3 executors
(detector, m2f segmentor, depther); only ImageNet's `model.backbone` happened to be right.

Nothing about the measurement looked wrong from inside. What gave it away was a comparison the
measurement itself cannot make: COCO at 4096 tokens reported LESS than ImageNet at 256, and a
transformer's forward pass is about `2 * params * tokens` regardless of how the wrapper is spelled.
That is the check, and it is cheap enough to run on every result file.

    python analysis/experiments/flops_sanity_gate.py

Tolerance is deliberately loose (0.4x - 2.5x). The estimate ignores norms, activations, softmax and
biases, counts attention only crudely, and does not model windowed attention, multi-crop inference
or TTA -- so a 2x disagreement is unremarkable and a 30x one is a wiring fault. This gate is for
orders of magnitude, not for accuracy; `flops_analytic.py` is the careful model.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLOPS = os.path.join(ROOT, "analysis", "results", "flops")

# tag -> (backbone params, tokens per image, note). Tokens = (side/patch)^2 for one forward; where a
# model runs several crops or frames per image the count is per crop and `crops` scales it.
# The third field is how many BACKBONE PASSES one image costs, and it is not always 1: NYU's
# depther runs test-time augmentation (`model(x)` plus `model(flip(x))`, averaged) and ADE20K's m2f
# segmentor runs sliding-window crops. Omitting it made NYU read 2.23x and ADE20K 2.31x -- close
# enough to the 2.5x bound to look like a real anomaly, which cost a round of investigation. It is
# a property of the eval protocol, not of the measurement, so it belongs in the expectation.
EXPECT = {
    "dinov3_imagenet_ceiling": (6.7e9, (256 // 16) ** 2, 1, "ViT-7B/16 @256px"),
    "dinov3_coco_ceiling":     (6.7e9, (1024 // 16) ** 2, 1, "ViT-7B/16 @1024px"),
    "dinov3_nyu_ceiling":      (6.7e9, (768 // 16) ** 2, 2, "ViT-7B/16 @768px, x2 TTA flip"),
    "dinov3_ade20k_ceiling":   (6.7e9, (896 // 16) ** 2, 2, "ViT-7B/16 @896px, x2 slide crops"),
    "openclip_imagenet_ceiling": (1.8e9, (224 // 14) ** 2, 1, "CLIP bigG vision @224px"),
}

LOW, HIGH = 0.4, 2.5


def main() -> int:
    ok = True
    print(f"{'tag':<30}{'measured':>12}{'expected':>12}{'ratio':>8}  note")
    for tag, (params, tokens, crops, note) in EXPECT.items():
        path = os.path.join(FLOPS, f"{tag}.json")
        if not os.path.exists(path):
            print(f"{tag:<30}{'--':>12}{'':>12}{'':>8}  not measured yet")
            continue
        d = json.load(open(path))
        bs = max(int(d.get("batch_size", 1) or 1), 1)
        measured = d["mean_total_gflops"] / bs
        expected = 2 * params * tokens * crops / 1e9
        ratio = measured / expected
        good = LOW <= ratio <= HIGH
        ok &= good
        flag = "     " if good else "  <<<"
        print(f"{tag:<30}{measured:>12.1f}{expected:>12.1f}{ratio:>8.2f}{flag} {note}")
    print()
    if ok:
        print("ALL WITHIN RANGE")
    else:
        print("OUT OF RANGE -- a ratio far from 1 usually means backbone_modules() is not "
              "resolving the trunk, so Linear/Conv hooks were never installed and only the "
              "globally-patched attention was counted.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
