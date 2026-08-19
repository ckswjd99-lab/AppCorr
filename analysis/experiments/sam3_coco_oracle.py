"""SAM 3 oracle baselines on COCO val2017 — the number every later arm is read against.

Two paths, because SAM 3 has two and they measure different things:

    --path tracker   `Sam3TrackerModel`, prompted with the ground-truth box. One box in, three
                     candidate masks out, keep the one the model scores highest. **Segmentation
                     quality in isolation** -- the model is told where the object is, so the score
                     moves only when its outlining changes. This is the arm to read the approx/
                     correct split against.

    --path detector  `Sam3Model`, prompted with the category name as text. 200 DETR queries out,
                     filtered by their logits. This is SAM 3's flagship concept-segmentation task
                     and matches the SA-Co row of the project page, but it folds **finding** the
                     objects into the same number as outlining them, so a drop here does not say
                     which of the two got worse.

Verified before writing this: both load from `facebook/sam3` with zero missing keys, and their
vision backbones share 520 of 538 tensors -- same architecture, 18 tensors separately tuned. So the
approx/correct fork applies to both, but **each path needs its own oracle**; they are not the same
tower.

Geometry: 1008x1008 input, patch 14, 72x72 = 5184 tokens, 32 layers, global attention at
[7, 15, 23, 31].

    python analysis/experiments/sam3_coco_oracle.py --path tracker  --num-images 100
    python analysis/experiments/sam3_coco_oracle.py --path detector --full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

COCO_ROOT = "/home/nxclab/fiftyone/coco-2017"
COCO_IMAGES = f"{COCO_ROOT}/validation/data"
COCO_INSTANCES = f"{COCO_ROOT}/raw/instances_val2017.json"
IMAGE_SIZE = 1008


def load_pixels(path, device, mean, std, dtype):
    img = Image.open(path).convert("RGB")
    ow, oh = img.size
    arr = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), copy=True)
    px = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
    return ((px - mean) / std).to(dtype), ow, oh


def rle_of(mask_bool):
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask_bool.cpu().numpy().astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def to_original(masks_logits, oh, ow):
    """[K, h, w] logits -> [K, oh, ow] booleans at the original image size."""
    m = torch.nn.functional.interpolate(
        masks_logits.unsqueeze(1).float(), size=(oh, ow), mode="bilinear", align_corners=False
    ).squeeze(1)
    return m > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=["tracker", "detector"], default="tracker")
    ap.add_argument("--repo", default="facebook/sam3")
    ap.add_argument("--num-images", type=int, default=100)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-boxes", type=int, default=20)
    ap.add_argument("--det-score-thresh", type=float, default=0.3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    token = os.environ.get("HF_TOKEN")

    coco = COCO(COCO_INSTANCES)
    index = []
    for iid in sorted(coco.getImgIds()):
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=False))
                if a.get("area", 0) > 1 and a.get("bbox")]
        if anns:
            index.append((iid, anns[: args.max_boxes]))
    if not args.full:
        index = index[: args.num_images]

    print(f"[oracle:{args.path}] loading {args.repo} ({args.dtype})", flush=True)
    cls = Sam3TrackerModel if args.path == "tracker" else Sam3Model
    model = cls.from_pretrained(args.repo, dtype=dtype, token=token).to(device).eval()
    processor = Sam3Processor.from_pretrained(args.repo, token=token)
    ip = getattr(processor, "image_processor", processor)
    mean = torch.tensor(ip.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, device=device).view(1, 3, 1, 1)
    cat_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    print(f"[oracle:{args.path}] {len(index)} images, "
          f"{sum(len(a) for _, a in index)} annotations", flush=True)

    results = []
    t0 = time.time()
    for n, (img_id, anns) in enumerate(index, 1):
        info = coco.loadImgs(img_id)[0]
        px, ow, oh = load_pixels(os.path.join(COCO_IMAGES, info["file_name"]), device, mean, std, dtype)
        sx, sy = IMAGE_SIZE / ow, IMAGE_SIZE / oh

        if args.path == "tracker":
            boxes = [[a["bbox"][0] * sx, a["bbox"][1] * sy,
                      (a["bbox"][0] + a["bbox"][2]) * sx, (a["bbox"][1] + a["bbox"][3]) * sy]
                     for a in anns]
            with torch.no_grad():
                out = model(pixel_values=px,
                            input_boxes=torch.tensor([boxes], device=device, dtype=torch.float32))
            # [B, num_boxes, 3, h, w] with [B, num_boxes, 3] scores: keep the model's own pick.
            masks, scores = out.pred_masks[0], out.iou_scores[0]
            best = scores.argmax(dim=-1)
            chosen = masks[torch.arange(masks.shape[0], device=masks.device), best]
            bin_masks = to_original(chosen.float(), oh, ow)
            for k, a in enumerate(anns):
                results.append({"image_id": img_id, "category_id": a["category_id"],
                                "segmentation": rle_of(bin_masks[k]),
                                "score": float(scores[k, best[k]])})
        else:
            # One forward per distinct category present, since the text prompt names the concept.
            for cid in sorted({a["category_id"] for a in anns}):
                enc = processor(text=cat_name[cid], return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(pixel_values=px, input_ids=enc["input_ids"],
                                attention_mask=enc.get("attention_mask"))
                logits = out.pred_logits[0].float()
                conf = logits.sigmoid().max(dim=-1).values if logits.dim() == 2 else logits.sigmoid().flatten()
                keep = (conf > args.det_score_thresh).nonzero(as_tuple=True)[0]
                if keep.numel() == 0:
                    continue
                bin_masks = to_original(out.pred_masks[0][keep].float(), oh, ow)
                for j, q in enumerate(keep.tolist()):
                    results.append({"image_id": img_id, "category_id": cid,
                                    "segmentation": rle_of(bin_masks[j]),
                                    "score": float(conf[q])})

        if n % 25 == 0 or n == len(index):
            el = time.time() - t0
            print(f"  [{n}/{len(index)}] {el:.0f}s  {el/n:.2f}s/img  preds={len(results)}", flush=True)

    if not results:
        print("[oracle] no predictions; nothing to score")
        sys.exit(1)

    # Score only the images actually run -- otherwise a subset is graded against all 5,000 and the
    # AP reads as a model failure rather than a bookkeeping one.
    ev = COCOeval(coco, coco.loadRes(results), "segm")
    ev.params.imgIds = [i for i, _ in index]
    ev.evaluate(); ev.accumulate(); ev.summarize()

    summary = {
        "path": args.path, "repo": args.repo, "num_images": len(index),
        "num_predictions": len(results), "dtype": args.dtype, "image_size": IMAGE_SIZE,
        "max_boxes": args.max_boxes,
        "det_score_thresh": args.det_score_thresh if args.path == "detector" else None,
        "mask_AP": float(ev.stats[0]), "mask_AP50": float(ev.stats[1]), "mask_AP75": float(ev.stats[2]),
        "mask_AP_small": float(ev.stats[3]), "mask_AP_medium": float(ev.stats[4]),
        "mask_AP_large": float(ev.stats[5]),
    }
    print("\n=== Final Summary: " + json.dumps(summary))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
