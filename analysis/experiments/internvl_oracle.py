"""InternVL3 approx-then-correct on RealWorldQA: floor / corrected / ceiling.

Same three arms and the same reporting as the SAM 3 drivers, on the benchmark the Qwen2.5-VL work
already measured, so the two model families are directly comparable. Scoring reuses
`realworldqa_offload_eval.score_answer` rather than reimplementing it -- a second implementation of
MCQ-letter extraction would silently diverge and make the comparison meaningless.

Three things are specific to InternVL and are the reason this is not a copy of the SAM 3 driver:

**Tiling.** The processor cuts the image into at most 12 448x448 tiles plus a thumbnail, choosing the
grid from the image's aspect ratio and area. `get_optimal_tiled_canvas` never looks at pixel content,
so degrading the image at native resolution and restoring its size leaves the tile count and layout
*identical* across arms -- verified on six aspect ratios. That is what makes floor/ceiling comparable
at all, and it is the ADE20K crop-cover precedent: layout from the original, content degraded.

**Residual energy is computed in TILE space, not image space.** Both the full and the degraded image
go through the same processor, giving two `[tiles, 3, 448, 448]` tensors with identical layout, and
the per-patch energy is their squared difference. Mapping energy from original coordinates into tiles
by hand would need the grid, the resize, and the thumbnail to be re-derived -- three chances to be
off by a tile.

**The token budget is per image, not global.** A 1-tile image has 1024 patches and a 13-tile image
has 13,312, so `--keep-ratio` is a fraction of whatever this image has. The patch score ranks across
*all* tiles together, so one tile may be corrected heavily and another not at all; that is why the
fork takes a boolean mask rather than a shared index vector.

    python analysis/experiments/internvl_oracle.py --arm ceiling --num-samples 50
    python analysis/experiments/internvl_oracle.py --arm corrected --keep-ratio 0.55 --full
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "experiments"))

from appcorr.models.internvl.vision.backbone import ApproxCorrectInternVLVisionTower
from realworldqa_offload_eval import score_answer


def l2_from_native(img: Image.Image, level: int, cap: int) -> Image.Image:
    """Degrade content at native resolution, then restore the original size.

    Restoring the size is what keeps the tiling identical -- the processor picks its grid from the
    image dimensions. `cap` implements the direction half of the pyramid rule
    (docs/memo/pyramid_degradation_native_vs_canvas.md): when the native image is LARGER than what
    the model will sample, degrade relative to the model's resolution, not the file's, or the
    approximation comes out milder than the pipeline's own.
    """
    w, h = img.size
    short = min(min(w, h), cap)
    t = max(1, short // 2 ** level)
    if h <= w:
        th, tw = t, max(1, round(w / h * t))
    else:
        tw, th = t, max(1, round(h / w * t))
    return img.resize((tw, th), Image.BOX).resize((w, h), Image.BICUBIC)


def patch_energy(px_full: torch.Tensor, px_l2: torch.Tensor, patch: int) -> torch.Tensor:
    """[tiles, 3, H, W] pair -> [tiles, patches] mean squared difference per patch."""
    d = (px_full.float() - px_l2.float()) ** 2
    tiles, _, h, w = d.shape
    d = d.mean(dim=1)                                        # over channels
    d = d.reshape(tiles, h // patch, patch, w // patch, patch)
    return d.mean(dim=(2, 4)).reshape(tiles, -1)


def select_patches(energy: torch.Tensor, attn: torch.Tensor | None, keep: float) -> torch.Tensor:
    """energy x attention, each normalised to unit mean first, then a global top-k over all tiles.

    Ranking across tiles rather than within each one is deliberate: a thumbnail tile covering the
    whole scene and a detail tile covering one corner do not deserve equal budgets.
    """
    score = energy / energy.mean().clamp_min(1e-12)
    if attn is not None:
        score = score * (attn / attn.mean().clamp_min(1e-12)).to(score.device)
    flat = score.reshape(-1)
    k = max(1, int(round(keep * flat.numel())))
    idx = flat.topk(k).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    return mask.reshape(score.shape)


class swap_vision:
    """Temporarily make the model's image features be the ones we computed.

    `InternVLModel.get_image_features` is what the generate path calls, so replacing it lets every
    arm run stock code from the LLM inward -- the same trick the SAM 3 driver uses, for the same
    reason: reproducing the post-tower path here would be a second implementation that drifts.
    """

    def __init__(self, model, tower, px, px_l2, arm, keep_ratio, pscore):
        self.model, self.tower = model, tower
        self.px, self.px_l2 = px, px_l2
        self.arm, self.keep_ratio, self.pscore = arm, keep_ratio, pscore
        self.original = None

    def __enter__(self):
        if self.arm == "ceiling":
            return self
        want_attn = self.arm == "corrected" and self.pscore == "energy_attn"
        x = self.tower.prepare_tokens(self.px_l2)
        hidden, cache = self.tower.approx_forward(x, {}, collect_attn=want_attn)

        if self.arm == "corrected":
            patch = self.model.config.vision_config.patch_size
            patch = patch[0] if not isinstance(patch, int) else patch
            energy = patch_energy(self.px, self.px_l2, patch)
            attn = cache.get("vision_layer_patch_attn_layermean") if want_attn else None
            pmask = select_patches(energy, attn, self.keep_ratio)
            tmask = self.tower.patch_mask_to_token_mask(pmask)
            # The corrected tokens must be recomputed from the FULL-resolution stream, so the
            # residual actually enters the model; the rest stay on the approximate one.
            x_full = self.tower.prepare_tokens(self.px)
            mixed = torch.where(tmask.unsqueeze(-1), x_full, x)
            hidden, _ = self.tower.correct_forward(mixed, tmask, cache)

        feats = self.tower.run_projector(hidden)
        self.original = self.model.get_image_features
        self.model.get_image_features = lambda *a, **k: type(
            "O", (), {"pooler_output": feats})()
        return self

    def __exit__(self, *exc):
        if self.original is not None:
            self.model.get_image_features = self.original
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL3-8B-hf")
    ap.add_argument("--arm", choices=["ceiling", "floor", "corrected"], default="ceiling")
    ap.add_argument("--keep-ratio", type=float, default=0.55)
    ap.add_argument("--pscore", choices=["energy", "energy_attn"], default="energy_attn")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    token = os.environ.get("HF_TOKEN")
    device = torch.device(args.device)

    print(f"[internvl] loading {args.model} ({args.dtype})", flush=True)
    full = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, token=token).eval()
    full.to(device)
    proc = AutoProcessor.from_pretrained(args.model, token=token)
    model = full.model
    tower = ApproxCorrectInternVLVisionTower(model).eval()

    # The model never sees more than one tile's worth of samples per axis, which is what the
    # pyramid rule's "fit to the input first" half is measured against.
    tile = proc.image_processor.size
    cap = int(tile["height"] if isinstance(tile, dict) else tile.height)

    ds = load_dataset("lmms-lab/RealWorldQA", split="test")
    n = len(ds) if args.full else min(args.num_samples, len(ds))
    idxs = list(range(len(ds)))[:n] if args.full else \
        list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    print(f"[internvl] arm={args.arm} keep={args.keep_ratio} pscore={args.pscore} "
          f"level=L{args.level} cap={cap}  {n} of {len(ds)} examples", flush=True)

    correct = 0
    per_sample = []
    tile_counts = []
    t0 = time.time()
    for n_done, i in enumerate(idxs, 1):
        ex = ds[i]
        img = ex["image"].convert("RGB")
        question, gt = ex["question"], str(ex["answer"])
        # Build the prompt with the model's own chat template. Hand-writing the placeholder both
        # gets the token wrong (it is `<IMG_CONTEXT>`, not `<image>`) and drops the turn markers the
        # model was tuned with, which costs accuracy without erroring.
        msgs = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

        enc = proc(images=img, text=prompt, return_tensors="pt").to(device)
        px = enc["pixel_values"].to(dtype)
        if args.arm == "ceiling":
            px_l2 = px
        else:
            deg = l2_from_native(img, args.level, cap)
            px_l2 = proc(images=deg, text=prompt,
                         return_tensors="pt")["pixel_values"].to(device, dtype)
            assert px_l2.shape == px.shape, f"tiling changed: {px.shape} vs {px_l2.shape}"
        tile_counts.append(px.shape[0])

        with torch.no_grad(), swap_vision(model, tower, px, px_l2, args.arm,
                                          args.keep_ratio, args.pscore):
            out = full.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = proc.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        ok = score_answer(question, text, gt)
        correct += bool(ok)
        per_sample.append({"idx": i, "gt": gt, "pred": text, "correct": bool(ok),
                           "tiles": int(px.shape[0])})
        if n_done % 25 == 0 or n_done == n:
            el = time.time() - t0
            print(f"  [{n_done}/{n}] {el:.0f}s  {el/n_done:.2f}s/ex  "
                  f"acc={correct/n_done*100:.2f}%", flush=True)

    summary = {
        "model": args.model, "arm": args.arm,
        "keep_ratio": args.keep_ratio if args.arm == "corrected" else None,
        "pscore": args.pscore if args.arm == "corrected" else None,
        "level": args.level if args.arm != "ceiling" else None,
        "num_samples": n, "accuracy": correct / n,
        "correct": correct, "dtype": args.dtype,
        "mean_tiles": float(np.mean(tile_counts)),
    }
    print("\n=== Final Summary: " + json.dumps(summary))
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"summary": summary, "per_sample": per_sample}, f)
        print(f"[internvl] wrote {args.out_json}")


if __name__ == "__main__":
    main()
