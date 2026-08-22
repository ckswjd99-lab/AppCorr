"""Critical backbone FLOPs for SAM 3, interleaved g=4 at two recompute rates.

SAM 3's backbone is the vision encoder alone -- the mask decoder and the presence head are the
"header" a VFM stops before -- and it is the one model here whose attention is not uniform: 4 of its
32 layers are global over all 5184 tokens and the other 28 are windowed over 576. That is exactly
the structure a config-derived FLOP formula gets wrong, and exactly why the count comes from hooks
on the real shapes instead.

The oracle keeps its model loading inside `main()`, so this reproduces the few lines it needs rather
than importing a loader that does not exist. Images come from the same COCO root the oracle uses;
only the pixel tensors matter here, since FLOPs do not depend on the prompts or the predictions.

    python analysis/experiments/flops_report_sam3.py [--samples 12] [--groups 4]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from appcorr import flops
import experiments.sam3_coco_oracle as S


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="facebook/sam3")
    ap.add_argument("--images", default=None,
                    help="directory of images; defaults to the COCO val2017 root the oracle uses")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keeps", type=float, nargs="+", default=[0.30, 0.50])
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--bounds", default="pre_global")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from transformers import AutoModel, Sam3Processor

    from appcorr.models.sam3.vision.backbone import ApproxCorrectSam3VisionTower

    dev, dt = a.device, torch.bfloat16
    token = os.environ.get("HF_TOKEN")
    # `facebook/sam3` is a sam3_video checkpoint (Sam3VideoModel). Forcing it into
    # Sam3TrackerModel loads a subset of the architecture and leaves `vision_encoder` returning a
    # tuple -- surfacing later as "'tuple' object has no attribute 'shape'", far from the cause.
    model = AutoModel.from_pretrained(a.repo, dtype=dt, token=token).to(dev).eval()
    # `Sam3VideoModel` nests the image path under `detector_model`; the tracker half is a separate
    # subtree with no vision encoder of its own. Descend to whichever child actually owns one
    # rather than assuming a flat layout.
    if not hasattr(model, "vision_encoder"):
        for attr in ("detector_model", "model", "sam3", "vision_model"):
            inner = getattr(model, attr, None)
            if inner is not None and hasattr(inner, "vision_encoder"):
                model = inner
                break
    if not hasattr(model, "vision_encoder"):
        raise SystemExit("could not locate a vision_encoder on the loaded SAM 3 model")
    proc = Sam3Processor.from_pretrained(a.repo, token=token)
    ip = getattr(proc, "image_processor", proc)
    mean = torch.tensor(ip.image_mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, device=dev).view(1, 3, 1, 1)
    tower = ApproxCorrectSam3VisionTower(model.vision_encoder).eval()

    # The oracle resolves COCO through `sam3_datasets`, which points at a fiftyone export rather
    # than a bare val2017 directory. Ask it rather than guessing a layout.
    from experiments.sam3_datasets import COCO_VAL_IMAGES
    roots = [a.images] if a.images else [COCO_VAL_IMAGES]
    paths = []
    for r in roots:
        if r and os.path.isdir(r):
            paths = sorted(glob.glob(os.path.join(r, "*.jpg")))[:a.samples]
            break
    if not paths:
        raise SystemExit(f"no images found; pass --images. tried: {roots}")
    print(f"[sam3] {len(paths)} images, {tower.num_layers} layers, "
          f"global at {tower.global_layers}", flush=True)

    def run(arm, keep, groups):
        with flops.session(model.vision_encoder, enabled=True) as fl:
            for p in paths:
                px = S.load_pixels(p, dev, mean, std, dt, 0)
                px2 = S.load_pixels(p, dev, mean, std, dt, a.level)
                with fl.request(p):
                    if arm == "ceiling":
                        # No arrivals opened: the exact forward waits on the whole image by
                        # definition, so the rule makes all of it critical.
                        with fl.stage("full"):
                            out = model.vision_encoder(px)
                            _ = out[0] if isinstance(out, tuple) else out
                    else:
                        with S.swap_vision(model, tower, px, px2, "corrected", keep,
                                           groups=groups, bounds=a.bounds, flops=fl):
                            pass
        return fl.aggregate()

    full = run("ceiling", 1.0, 1)["mean_total_gflops"]
    print(f"\n══ SAM 3 · COCO val2017 ({len(paths)} images) ══")
    print(f"  full inference (ceiling)  {full:10.1f} GFLOPs/image")
    for keep in a.keeps:
        agg = run("corrected", keep, a.groups)
        crit = agg["mean_critical_gflops"]
        print(f"  interleaved g={a.groups} keep={keep:.0%}  critical {crit:9.1f}  "
              f"total {agg['mean_total_gflops']:9.1f} GFLOPs   "
              f"critical/full = {100*crit/full:5.1f}%   stages={agg['mean_stage_gflops']}")


if __name__ == "__main__":
    main()
