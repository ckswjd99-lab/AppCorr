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
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

# Run directly (`python analysis/experiments/...`) rather than as a module, so the repo root is not
# on sys.path and `appcorr` would not import.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

COCO_ROOT = "/home/nxclab/fiftyone/coco-2017"
COCO_IMAGES = f"{COCO_ROOT}/validation/data"
COCO_INSTANCES = f"{COCO_ROOT}/raw/instances_val2017.json"
IMAGE_SIZE = 1008


def _to_tensor(img, device, mean, std, dtype):
    arr = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), copy=True)
    px = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
    return ((px - mean) / std).to(dtype)


def load_pixels(path, device, mean, std, dtype, level: int = 0):
    """Model-canvas tensor for the given pyramid level, degraded natively first when level > 0."""
    img = Image.open(path).convert("RGB")
    ow, oh = img.size
    src = img if level == 0 else l2_from_native(img, level)
    return _to_tensor(src, device, mean, std, dtype), ow, oh


def rle_of(mask_bool):
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask_bool.cpu().numpy().astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def l2_from_native(img, level: int = 2) -> "Image.Image":
    """Build the level-`level` approximation **from the original image in native coordinates**.

    Required by docs/memo/pyramid_degradation_native_vs_canvas.md: pyramid levels are built from the
    original and only the selected level is scaled to the model input. Degrading the canvas instead
    barely touches real content when the canvas is an upscale -- COCO's native median is 480x640
    against SAM 3's 1008x1008 input, so canvas-relative degradation removes information the upscale
    had just invented. That defect once made COCO's floor and ceiling agree to 1e-4, leaving nothing
    for correction to recover; fixing it opened the gap to ~0.065.

    Short side / 2**level, aspect preserved, area-downsampled. The caller then resizes to the model
    canvas, which is the "only the selected level is scaled" half of the rule.
    """
    ow, oh = img.size
    scale = 2 ** level
    short = min(ow, oh)
    target_short = max(1, short // scale)
    if oh <= ow:
        th, tw = target_short, max(1, round(ow / oh * target_short))
    else:
        tw, th = target_short, max(1, round(oh / ow * target_short))
    return img.resize((tw, th), Image.BOX)


def residual_energy(px_full: torch.Tensor, px_l2: torch.Tensor, patch: int) -> torch.Tensor:
    """Per-patch squared difference between the real image and its L2 approximation. [H/p * W/p].

    Client-side signal: computable without the model, which is what makes it deployable.
    """
    resid = (px_full.float() - px_l2.float()).pow(2).sum(dim=1, keepdim=True)
    return torch.nn.functional.avg_pool2d(resid, patch).flatten()


def select_tokens(energy: torch.Tensor, attn: torch.Tensor | None, keep: float) -> torch.Tensor:
    """Top-`keep` patches by the chosen score.

    `energy x attn` rather than energy alone: energy says only where the approximation is wrong,
    with no notion of whether the network reads from there. Weighting by received attention asks for
    both. Each factor is normalised to unit mean first, so the product is not dominated by whichever
    happens to have the larger scale.
    """
    score = energy
    if attn is not None:
        e = energy / energy.mean().clamp_min(1e-12)
        a = attn.flatten().to(energy.device) / attn.mean().clamp_min(1e-12)
        score = e * a
    k = max(1, int(round(score.numel() * keep)))
    return score.topk(k).indices.sort().values


def layer_bounds(num_layers: int, global_idx, groups: int, mode: str) -> list[int]:
    """Layer index each interleaved round runs up to.

    `aligned` splits the depth evenly: at 32 layers and g=4 that is 8/16/24/32, and because SAM 3's
    global-attention layers sit at 7/15/23/31 every one of those ranges *ends just after* a global
    layer -- so rounds 2..g each re-correct it.

    `pre_global` stops one layer earlier, 7/15/23/32, ending just before each global layer instead.
    The global layers are the expensive ones to correct (attention over all 5184 tokens rather than
    576 inside a window), so deferring each one to the round after removes global-layer corrections
    from the early rounds. The last bound is always the full depth, so nothing is skipped overall --
    the work moves later, it does not disappear.
    """
    if groups <= 1:
        return [num_layers]
    if mode == "pre_global":
        gl = sorted(i for i in global_idx if 0 < i < num_layers)
        bounds = [gl[i] for i in range(min(groups - 1, len(gl)))]
        while len(bounds) < groups - 1:                    # more groups than global layers
            bounds.append(round(num_layers * (len(bounds) + 1) / groups))
        return sorted(set(bounds))[: groups - 1] + [num_layers]
    return [min(num_layers, round(num_layers * (r + 1) / groups)) for r in range(groups)]


def block_grid_groups(height: int, width: int, groups: int, device) -> list[torch.Tensor]:
    """Partition the token grid into `groups` contiguous blocks, block_grid style.

    block_grid rather than grid because it is the measured default -- it took ImageNet top1 and
    COCO i2t on the CLIP comparison. Needs a square group count, same constraint the transmission
    policy has.
    """
    s = int(round(groups ** 0.5))
    if s * s != groups:
        raise ValueError(f"block_grid needs a square group count, got {groups}")
    rows = torch.arange(height, device=device).unsqueeze(1)
    cols = torch.arange(width, device=device).unsqueeze(0)
    gid = (rows * s // height) * s + (cols * s // width)
    gid = gid.flatten()
    return [(gid == g).nonzero(as_tuple=True)[0] for g in range(groups)]


class swap_vision:
    """Temporarily make the model's vision encoder return features we computed.

    Both paths consume the tower differently -- `Sam3Model` takes a `vision_embeds` object while
    `Sam3TrackerModel` wants `image_embeddings` *after* its own post-processing (no-memory embedding
    added, reshaped by `backbone_feature_sizes`). Reproducing that second transform here would be a
    second implementation of something the model already does, and it is exactly the kind of thing
    that silently drifts. Swapping the encoder instead lets both paths run their stock code from the
    FPN onward, so the only thing this driver changes is which features go in.
    """

    def __init__(self, model, tower, px, px_l2, arm, keep_ratio, pscore="energy_attn",
                 groups=1, bounds="aligned", force_interleaved=False):
        self.model, self.tower, self.px, self.px_l2 = model, tower, px, px_l2
        self.arm, self.keep_ratio, self.pscore = arm, keep_ratio, pscore
        self.groups, self.bounds = groups, bounds
        # Gate only: run the interleaved machinery with one group, which must reproduce one-shot.
        self.force_interleaved = force_interleaved
        self.original = None

    def __enter__(self):
        if self.arm == "ceiling":
            return self
        from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

        px_l2 = self.px_l2
        want_attn = self.arm == "corrected" and self.pscore == "energy_attn"
        x_approx = self.tower.prepare_tokens(px_l2)
        hidden, cache = self.tower.approx_forward(x_approx, {}, collect_attn=want_attn)

        if self.arm == "corrected":
            energy = residual_energy(self.px, px_l2, self.tower.patch_size)
            attn = cache.get("vision_layer_patch_attn_layermean") if want_attn else None
            idx = select_tokens(energy, attn, self.keep_ratio).to(self.px.device)
            x_full = self.tower.prepare_tokens(self.px)
            b, h, w, c = x_approx.shape
            flat_a, flat_f = x_approx.reshape(b, h * w, c), x_full.reshape(b, h * w, c)
            mixed = flat_a.clone()
            mixed[:, idx] = flat_f[:, idx]
            mixed = mixed.reshape(b, h, w, c)

            if self.groups <= 1 and not self.force_interleaved:
                hidden, _ = self.tower.correct_forward(mixed, idx, cache)
            else:
                # Interleaved: redo the approximate pass in ranges so each round's cache reflects
                # only the depth reached so far, then correct the arrived groups over that depth.
                # Every round starts from the same input stream -- that is how the executor calls
                # it, and it is what makes persisting the corrected increment necessary.
                bounds = layer_bounds(self.tower.num_layers, self.tower.global_layers,
                                      self.groups, self.bounds)
                per_group = block_grid_groups(h, w, self.groups, idx.device)
                # Restrict each spatial group to the tokens the score actually selected, so the
                # union over all rounds is exactly the one-shot set. Getting this wrong is not
                # visible in the score: an earlier version indexed groups from 1 and silently
                # dropped group 0, correcting 41.3% while reporting a 55% keep ratio -- fewer
                # tokens than one-shot yet scoring higher, which is how it was caught.
                keep_mask = torch.zeros(h * w, dtype=torch.bool, device=idx.device)
                keep_mask[idx] = True
                per_group = [g[keep_mask[g]] for g in per_group]

                # The input stream may only carry full-resolution values for tokens that have
                # ALREADY arrived. Handing the whole `mixed` to the approximate pass, or to a
                # round whose arrived set is smaller than `idx`, feeds the model data from the
                # future: the un-arrived selected tokens sit in the residual stream at full
                # resolution (`correct` reconstructs untouched positions as `flat + increment`,
                # reading `flat` everywhere, not just at the corrected indices). That inflates
                # every interleaved arm against one-shot, which never has the gap because its
                # corrected set IS `idx`.
                def stream(tok):
                    """Layer-0 input with full resolution at every token that has ARRIVED.

                    Two separate things, easy to conflate: what the *stream* carries and what a
                    round *recomputes*. The stream is cumulative -- a corrected token's layer-0
                    value simply IS its full-resolution patch embedding, and `correct` rebuilds
                    untouched positions as `flat + increment`, reading `flat` everywhere. Handing
                    it the full `idx` instead would put un-arrived tokens in at full resolution,
                    which is data from the future.
                    """
                    m = flat_a.clone()
                    if tok.numel():
                        m[:, tok] = flat_f[:, tok]
                    return m.reshape(b, h, w, c)

                cache = {}
                # Nothing has arrived yet, so the opening pass is the pure approximate one --
                # the same input one-shot starts from.
                hidden, cache = self.tower.approx_forward(
                    x_approx, cache, layers=(0, bounds[0]))
                arrived = []
                for r in range(len(bounds)):
                    arrived.append(per_group[r])
                    # THIS ROUND'S GROUP ONLY -- never the accumulated set. Earlier groups are
                    # already corrected and stay corrected: their recomputed K/V sits in the
                    # cache, their corrected increment was persisted, and the approximate pass
                    # over the next layer range carried them forward. Re-listing them recomputes
                    # what is already right and, on the last round, collapses the whole schedule
                    # into one-shot correction. The VGGT path records the same mistake
                    # (`offload/server/model/vggt_omega.py`), where interleaved and one-shot then
                    # agreed to 16 decimals.
                    tok = per_group[r]
                    if tok.numel():
                        hidden, cache = self.tower.correct_forward(
                            stream(torch.cat(arrived).sort().values), tok, cache,
                            layers=(0, bounds[r]))
                    if r + 1 < len(bounds):
                        hidden, cache = self.tower.approx_forward(
                            hidden, cache, layers=(bounds[r], bounds[r + 1]))

        fpn, pos = self.tower.run_neck(hidden)
        out = Sam3VisionEncoderOutput(fpn_hidden_states=fpn, fpn_position_encoding=pos)
        self.original = self.model.vision_encoder.forward
        self.model.vision_encoder.forward = lambda *a, **k: out
        return self

    def __exit__(self, *exc):
        if self.original is not None:
            self.model.vision_encoder.forward = self.original
        return False


def to_original(masks_logits, oh, ow):
    """[K, h, w] logits -> [K, oh, ow] booleans at the original image size."""
    m = torch.nn.functional.interpolate(
        masks_logits.unsqueeze(1).float(), size=(oh, ow), mode="bilinear", align_corners=False
    ).squeeze(1)
    return m > 0


def sam3_datasets_subsets():
    import sam3_datasets
    return "/".join(sam3_datasets.SACO_GOLD_SUBSETS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="coco",
                    help="coco: val2017, 80 categories, COCOeval. "
                         "lvis: v1 val, 1203 categories with rare/common/frequent buckets, "
                         "LVISEval (federated annotations -- COCOeval would punish correct "
                         "detections of un-annotated objects). "
                         "saco_gold:<subset>: promptable concept segmentation scored by cgF1, "
                         f"subsets {sam3_datasets_subsets()}")
    ap.add_argument("--path", choices=["tracker", "detector"], default="tracker")
    ap.add_argument("--arm", choices=["ceiling", "floor", "corrected"], default="ceiling",
                    help="ceiling: stock forward on the full-resolution image. "
                         "floor: approximate forward on the L2 image, no correction. "
                         "corrected: approximate on L2, then recompute --keep-ratio of the tokens "
                         "from the full-resolution image.")
    ap.add_argument("--pscore", choices=["energy", "energy_attn"], default="energy_attn",
                    help="energy: residual energy only. energy_attn (default): residual energy x "
                         "the token's layer-mean received attention, the validated combination.")
    ap.add_argument("--groups", type=int, default=1,
                    help="1 = one-shot correction. >1 = interleaved: patches arrive in this many "
                         "block_grid groups and each round corrects what has arrived so far.")
    ap.add_argument("--bounds", choices=["aligned", "pre_global"], default="aligned",
                    help="Where each interleaved round stops. aligned: even splits (8/16/24/32 at "
                         "g=4). pre_global: one layer earlier (7/15/23/32), stopping just before "
                         "each global-attention layer instead of just after it.")
    ap.add_argument("--force-interleaved", action="store_true",
                    help="equivalence gate: drive the interleaved path even at --groups 1")
    ap.add_argument("--keep-ratio", type=float, default=0.55,
                    help="fraction of tokens the corrected arm recomputes, chosen by residual energy")
    ap.add_argument("--repo", default="facebook/sam3")
    ap.add_argument("--num-images", type=int, default=100)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-boxes", type=int, default=20)
    # Detection AP is a ranking metric: COCOeval sorts predictions by score and sweeps the
    # threshold itself, keeping the top `maxDets` per image. Cutting at a fixed confidence
    # therefore only ever *removes* recall the metric would have used -- a threshold is a knob
    # that can only hurt, and picking one by hand puts an arbitrary number in front of every
    # comparison. Keep the top-k per prompt instead and let the metric do its job.
    ap.add_argument("--det-per-cat", type=int, default=30,
                    help="top-k queries kept per category prompt, by confidence")
    ap.add_argument("--det-max-dets", type=int, default=100,
                    help="predictions kept per image after merging prompts; COCO's maxDets")
    ap.add_argument("--no-presence", action="store_true",
                    help="score queries by pred_logits alone, ignoring the presence token "
                         "(the pre-fix behaviour; kept to reproduce older numbers)")
    ap.add_argument("--det-score-thresh", type=float, default=0.0,
                    help="optional floor on confidence; 0 disables it (the default -- see above)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--pred-json", default=None,
                    help="where to dump raw predictions before scoring; defaults to "
                         "<out-json>.preds.json. Feed back with --score-only.")
    ap.add_argument("--score-only", default=None,
                    help="skip inference and score this predictions file instead")
    args = ap.parse_args()

    from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel

    import sam3_datasets

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    token = os.environ.get("HF_TOKEN")

    bench = sam3_datasets.build(args.dataset, args.max_boxes,
                                None if args.full else args.num_images)
    index = bench.items

    if args.score_only:
        with open(args.score_only) as f:
            results = json.load(f)
        print(f"[oracle] scoring {len(results)} predictions from {args.score_only}", flush=True)
        stats = bench.evaluate(results, [it.image_id for it in index])
        print("\n=== Final Summary: " + json.dumps(
            {"dataset": bench.name, "path": args.path, "arm": args.arm,
             "scored_from": args.score_only, "num_images": len(index),
             "num_predictions": len(results), **bench.meta, **stats}))
        return

    print(f"[oracle:{args.path}] loading {args.repo} ({args.dtype})", flush=True)
    cls = Sam3TrackerModel if args.path == "tracker" else Sam3Model
    model = cls.from_pretrained(args.repo, dtype=dtype, token=token).to(device).eval()
    processor = Sam3Processor.from_pretrained(args.repo, token=token)
    ip = getattr(processor, "image_processor", processor)
    mean = torch.tensor(ip.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, device=device).view(1, 3, 1, 1)
    cat_name = bench.cat_name

    tower = None
    if args.arm != "ceiling":
        from appcorr.models.sam3.vision.backbone import ApproxCorrectSam3VisionTower
        tower = ApproxCorrectSam3VisionTower(model.vision_encoder).eval()
        print(f"[oracle:{args.path}] arm={args.arm} keep_ratio={args.keep_ratio} "
              f"({tower.num_layers} layers, global at {tower.global_layers})", flush=True)

    print(f"[oracle:{args.path}] {bench.name}: {len(index)} images, "
          f"{sum(len(it.anns) for it in index)} annotations, "
          f"{len(cat_name)} categories", flush=True)

    results = []
    t0 = time.time()
    for n, item in enumerate(index, 1):
        img_id, anns = item.image_id, item.anns
        px, ow, oh = load_pixels(item.file_path, device, mean, std, dtype, level=0)
        px_l2 = (px if args.arm == "ceiling"
                 else load_pixels(item.file_path, device, mean, std, dtype, level=2)[0])
        sx, sy = IMAGE_SIZE / ow, IMAGE_SIZE / oh

        if args.path == "tracker":
            boxes = [[a["bbox"][0] * sx, a["bbox"][1] * sy,
                      (a["bbox"][0] + a["bbox"][2]) * sx, (a["bbox"][1] + a["bbox"][3]) * sy]
                     for a in anns]
            bt = torch.tensor([boxes], device=device, dtype=torch.float32)
            with torch.no_grad(), swap_vision(model, tower, px, px_l2, args.arm, args.keep_ratio, args.pscore,
                                        args.groups, args.bounds,
                                        args.force_interleaved):
                out = model(pixel_values=px, input_boxes=bt)
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
            per_image = []
            # What to ask, and what each answer belongs to. COCO/LVIS derive prompts from the GT
            # categories and key every prediction to this image; SA-Co supplies (pair_id, phrase)
            # pairs, where the pair is the datapoint and one image carries several. See
            # `sam3_datasets.Item.prompts`.
            prompts = (item.prompts if item.prompts is not None
                       else [(img_id, cat_name[cid], cid)
                             for cid in sorted({a["category_id"] for a in anns})])
            prompts = [(p[0], p[1], p[2] if len(p) > 2 else 1) for p in prompts]

            # ONE vision pass for the whole image, reused by every prompt. The approx/correct work
            # depends only on the pixels -- the text never enters the tower -- so running it inside
            # the prompt loop recomputed identical features once per prompt. On COCO that is 3.5
            # prompts per image and cost 1.9x; on SA-Co/Gold's metaclip subset it is 55.
            # Numerically a no-op: verified bit-identical on the detector ceiling.
            with torch.no_grad(), swap_vision(model, tower, px, px_l2, args.arm, args.keep_ratio,
                                              args.pscore, args.groups, args.bounds,
                                              args.force_interleaved):
                for result_id, text, cid in prompts:
                    enc = processor(text=text, return_tensors="pt").to(device)
                    out = model(pixel_values=px, input_ids=enc["input_ids"],
                                attention_mask=enc.get("attention_mask"))
                    # SAM 3 splits recognition from localisation: a dedicated presence token says
                    # whether the concept occurs at all, and `Sam3ModelOutput` documents the score
                    # as `pred_logits.sigmoid() * presence_logits.sigmoid()`. Using the query logit
                    # alone throws away the recognition half.
                    #
                    # It makes no difference to COCO/LVIS -- there only categories present in the GT
                    # are prompted, and a per-image constant factor cannot reorder a ranking metric.
                    # On SA-Co it is decisive: 80% of prompts ask about an absent concept, and
                    # without presence the harness answered "present" for 98% of them (IL_FPR 0.980,
                    # IL_MCC 0.073), which is most of cgF1 since cgF1 = positive_micro_F1 x IL_MCC.
                    logits = out.pred_logits[0].float()
                    conf = (logits.sigmoid().max(dim=-1).values if logits.dim() == 2
                            else logits.sigmoid().flatten())
                    pres = getattr(out, "presence_logits", None)
                    if pres is not None and not args.no_presence:
                        conf = conf * pres[0].float().sigmoid().reshape(-1)[0]
                    order = conf.argsort(descending=True)
                    if args.det_score_thresh > 0:
                        order = order[conf[order] > args.det_score_thresh]
                    keep = order[: args.det_per_cat]
                    if keep.numel() == 0:
                        continue
                    # Masks are decoded after the global top-k below, not here: RLE over every kept
                    # query of every category is the dominant cost of this path.
                    for q in keep.tolist():
                        per_image.append((float(conf[q]), cid, result_id,
                                          out.pred_masks[0][q].float().cpu()))

            # Rank across prompts, then cut. For SA-Co the cut is per prompt, not per image: a
            # datapoint is one (image, phrase) pair and its predictions must not compete with a
            # different phrase's for the same budget.
            per_image.sort(key=lambda t: -t[0])
            if item.prompts is not None:
                kept, seen = [], {}
                for rec in per_image:
                    n = seen.get(rec[2], 0)
                    if n < args.det_max_dets:
                        seen[rec[2]] = n + 1
                        kept.append(rec)
            else:
                kept = per_image[: args.det_max_dets]
            for score, cid, result_id, mlogit in kept:
                results.append({"image_id": result_id, "category_id": cid,
                                "segmentation": rle_of(to_original(mlogit[None].to(device), oh, ow)[0]),
                                "score": score})

        if n % 25 == 0 or n == len(index):
            el = time.time() - t0
            print(f"  [{n}/{len(index)}] {el:.0f}s  {el/n:.2f}s/img  preds={len(results)}", flush=True)

    if not results:
        print("[oracle] no predictions; nothing to score")
        sys.exit(1)

    # Dump predictions BEFORE scoring. Inference is the expensive half -- an hour of GPU on LVIS --
    # and the evaluators are third-party code that can fail for reasons unrelated to the run: the
    # `lvis` package (0.5.3, 2020) calls `np.float`, removed in numpy 1.24, and took a completed
    # 19,626-image ceiling arm down with it. With this, a scoring crash costs a rerun of
    # `--score-only`, not a rerun of the model.
    pred_path = args.pred_json or (f"{args.out_json}.preds.json" if args.out_json else None)
    if pred_path:
        os.makedirs(os.path.dirname(os.path.abspath(pred_path)), exist_ok=True)
        with open(pred_path, "w") as f:
            json.dump(results, f)
        print(f"[oracle] {len(results)} predictions -> {pred_path}", flush=True)

    stats = bench.evaluate(results, [it.image_id for it in index])

    summary = {
        "dataset": bench.name, "path": args.path, "arm": args.arm,
        "keep_ratio": args.keep_ratio if args.arm == "corrected" else None,
        "pscore": args.pscore if args.arm == "corrected" else None,
        "groups": args.groups if args.arm == "corrected" else None,
        "bounds": args.bounds if args.arm == "corrected" and args.groups > 1 else None,
        "repo": args.repo, "num_images": len(index),
        "num_predictions": len(results), "dtype": args.dtype, "image_size": IMAGE_SIZE,
        "max_boxes": args.max_boxes,
        "det_per_cat": args.det_per_cat if args.path == "detector" else None,
        "det_max_dets": args.det_max_dets if args.path == "detector" else None,
        "det_score_thresh": args.det_score_thresh if args.path == "detector" else None,
        **bench.meta,
        **stats,
    }
    print("\n=== Final Summary: " + json.dumps(summary))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
