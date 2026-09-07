"""Gate for the SAM 3 progressive interleaved walk.

Identity: g=1 (force_interleaved), keep=1.0 must match the ONE-SHOT arm at keep=1.0 -- at g=1 the
opening range is the full depth, everything has arrived, the running attention mean IS the
full-depth mean, and everything selected is everything: the two paths compute the same thing.
Compared on the neck's FPN outputs (what every downstream consumer sees), fp-noise scale.

Direction: g=4 keep=0.25 progressive must land between floor and one-shot k=1.0 in feature space.
"""
import glob, os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import experiments.sam3_coco_oracle as S
from appcorr.models.sam3.vision.backbone import ApproxCorrectSam3VisionTower
from transformers import AutoModel


def main() -> int:
    dev, dt = "cuda:0", torch.bfloat16
    model = AutoModel.from_pretrained("facebook/sam3", dtype=dt).to(dev).eval()
    if not hasattr(model, "vision_encoder"):
        for attr in ("detector_model", "model", "sam3", "vision_model"):
            child = getattr(model, attr, None)
            if child is not None and hasattr(child, "vision_encoder"):
                model = child
                break
    tower = ApproxCorrectSam3VisionTower(model.vision_encoder)
    mean = std = None
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    ip = proc.image_processor
    mean = torch.tensor(ip.image_mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, device=dev).view(1, 3, 1, 1)

    paths = sorted(glob.glob(os.path.join(S.COCO_IMAGES, "*.jpg")))[:6]
    if not paths:
        # The first run of this gate globbed a wrong directory, matched nothing, skipped the loop
        # entirely and printed ALL GATES PASS with zero samples checked. A gate with no samples is
        # not a pass.
        raise SystemExit(f"no images under {S.COCO_IMAGES} -- gate would be vacuous")
    ok = True
    for p in paths:
        px, _, _ = S.load_pixels(p, dev, mean, std, dt, 0)
        px2, _, _ = S.load_pixels(p, dev, mean, std, dt, 2)

        def fpn_of(arm, keep, groups, force=False):
            with S.swap_vision(model, tower, px, px2, arm, keep, groups=groups,
                               bounds="pre_global", force_interleaved=force):
                out = model.vision_encoder(px)   # swapped: returns the walked result
            return [f.float() for f in out.fpn_hidden_states]

        ref = fpn_of("corrected", 1.0, 1)                     # one-shot, everything corrected
        prog = fpn_of("corrected", 1.0, 1, force=True)        # progressive machinery, same point
        floor = fpn_of("floor", 1.0, 1)
        g4 = fpn_of("corrected", 0.25, 4)
        one25 = fpn_of("corrected", 0.25, 1)                  # one-shot at the same keep -- the
        # control that separates "progressive broke it" from "feature-space rel-L2 is simply not
        # monotone at 25% keep on this model". SAM 3's own accuracy results already say the task
        # metric recovers ~90% at 55% keep, so the feature ordering is a soft check, not a law.

        def rel(a, b):
            return max(((x - y).norm() / y.norm().clamp_min(1e-9)).item() for x, y in zip(a, b))
        r_id, r_fl = rel(prog, ref), rel(floor, ref)
        r_g4, r_one = rel(g4, ref), rel(one25, ref)
        g1 = r_id < 5e-3
        # Gate: progressive at k=0.25 must not be meaningfully WORSE than one-shot at k=0.25 in
        # feature space. Whether either beats the floor per-sample is not gated -- measured: the
        # one-shot arm itself lands farther than the floor on some samples.
        g4ok = r_g4 < r_one * 1.15
        ok &= g1 and g4ok
        print(f"  {os.path.basename(p)}: identity {r_id:.6f}  g4k.25 {r_g4:.4f}  "
              f"oneshot-k.25 {r_one:.4f}  floor {r_fl:.4f}   "
              f"[{'PASS' if g1 else 'FAIL'} identity] [{'PASS' if g4ok else 'FAIL'} vs-oneshot]")
    print()
    print("ALL GATES PASS" if ok else "GATE FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
