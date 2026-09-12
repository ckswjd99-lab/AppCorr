"""Dump the per-merge-group selection score of one dataset -- the calibration input for
"adaptive k" (docs/memo/adaptive_keep_design.md). GPU; one tower pass per image.

For every sampled image it writes, per merge group g:

    rms    sqrt(mean_{rows in g} mean_j (x_full - x_base)_j^2) with x in RAW [0, 1] pixel units
           (the processor's per-channel normalisation undone with its own `image_std`, read off
           the LOADED processor -- see `QwenVLStreamingAxis._pixel_std`). Absolute and
           model-independent: 0.03 is ~8 gray levels of error on average.
    mse    the CURRENT arm's energy: the same residual squared, pooled to the group and divided
           by the image's mean (per-image mean-1, hence scale-free and blind to how degraded the
           base is -- which is why it is the wrong thing to threshold; dumped so the two can be
           compared at equal mean k).
    mse_raw    the same MSE before that normalisation, in normalised-pixel units squared.
    attn   N * mean_{rows in g} received-attention layer mean = the axis's `attn_term`, mean 1
           over the image (the received attention sums to 1 over rows, so this is "relative to
           uniform"). Full tower depth, eager -- the deferred path computes the identical vector
           later, and the unified axis's prefix mean is a different (shallower) signal that this
           dump does not cover.
    band   the arrival band under `--groups` (sequential grouping, `axis._bands`).

Arrays are CONCATENATED over images with an `offsets` index (group counts differ per image).
The tower runs on the BASE image only -- that is what the arm has when it must choose, and what
carries the attention; the full image is needed for the pixel residual alone, not for a pass.

  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=/NHNHOME/huggingface/hub \\
  PYTHONPATH=$PWD python analysis/experiments/pscore_dump.py --family qwen35 \\
      --model Qwen/Qwen3.5-35B-A3B --dataset vstar --degrade-filter pyr --samples 36
"""
import argparse, json, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from analysis.experiments.qwen35_accuracy import degrade          # noqa: E402
from analysis.experiments.qwen_vllm_accuracy import make_axis     # noqa: E402


@torch.no_grad()
def score_one(axis, inputs, px_base, groups):
    """(rms, mse, mse_raw, attn, band) per merge group, as `streaming_forward` would compute them."""
    dev = px_base.device
    grid = inputs["image_grid_thw"]
    unit = axis.tower.spatial_merge_unit
    px_full = inputs["pixel_values"].to(device=dev, dtype=axis.model.dtype)
    px_base = px_base.to(device=dev, dtype=px_full.dtype)
    gctx = axis.tower.prepare_grid(grid, dev)
    ctx_full = axis.tower.prepare_full_tokens(px_full, grid, gctx)
    ctx_base = axis.tower.prepare_full_tokens(px_base, grid, gctx)
    n_rows = ctx_full["seq_len"]
    n_groups = n_rows // unit

    # the tower pass: base image, full depth, eager attention collection (the streaming arm's
    # own `_approx_base`, so the vector is the arm's, not a re-derivation)
    _, cache = axis._approx_base(ctx_base, {}, collect_attn=True)
    rows_all = axis._rows_of_groups(ctx_full, torch.arange(n_groups, device=dev))
    a = axis._attn_layermean(cache).to(torch.float32)
    a = a[rows_all.to(a.device)].reshape(n_groups, unit).mean(dim=1)
    a = a / a.mean().clamp_min(1e-12)

    d = px_full.float() - px_base.float()
    mse_raw = d.pow(2).mean(dim=-1).reshape(n_groups, unit).mean(dim=1)
    mse = mse_raw / mse_raw.mean().clamp_min(1e-12)
    std = axis._pixel_std(int(px_full.shape[-1]), dev, torch.float32)
    d = d if std is None else d * std
    rms = d.pow(2).mean(dim=-1).reshape(n_groups, unit).mean(dim=1).sqrt()

    band = torch.zeros(n_groups, dtype=torch.int16)
    for r, (g0, g1) in enumerate(axis._bands(groups, n_groups)):
        band[g0:g1] = r
    return (rms.cpu().numpy().astype(np.float32), mse.cpu().numpy().astype(np.float32),
            mse_raw.cpu().numpy().astype(np.float32), a.cpu().numpy().astype(np.float32),
            band.numpy(), std is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen25vl", "qwen35"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--degrade-filter", choices=["bicubic", "box", "pyr"], default="box",
                    help="MUST match the campaign arm being calibrated (the 2026-09-11 "
                         "convention: box for realworldqa/chartqa/mmvp/vsr/cvbench, pyr for "
                         "the rest) -- the score is a property of the degradation")
    ap.add_argument("--samples", type=int, default=36, help="0 = full split")
    ap.add_argument("--contiguous", action="store_true")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="analysis/results/pscore_dump")
    a = ap.parse_args()

    from transformers import AutoProcessor
    from qwen_vl_prefill.datasets_eval import get_spec
    from datasets import load_dataset
    from appcorr.models.vision_only import load_vision_only

    proc = AutoProcessor.from_pretrained(a.model)
    model = load_vision_only(a.model, device=a.device)
    axis = make_axis(a.family, model, proc)
    ip = proc.image_processor
    print(f"processor {type(ip).__name__}: image_mean={list(ip.image_mean)} "
          f"image_std={list(ip.image_std)} rescale={float(ip.rescale_factor):.8f} "
          f"-> one gray level = "
          f"{[round(float(ip.rescale_factor) / float(s), 6) for s in ip.image_std]} "
          f"normalised units", flush=True)
    tmpl_kw = {"think": True} if (a.think and a.family == "qwen35") else {}

    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    n = len(ds) if a.samples == 0 else min(a.samples, len(ds))
    idxs = list(range(n)) if a.samples == 0 else \
        (list(range(n)) if a.contiguous else list(range(0, len(ds), max(1, len(ds) // n)))[:n])

    cols = {k: [] for k in ("rms", "mse", "mse_raw", "attn", "band")}
    img_id, n_groups, offs, hw, raw_units = [], [], [0], [], True
    t0 = time.perf_counter()
    for j, i in enumerate(idxs):
        img, q, _ = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        if img.mode != "RGB":
            img = img.convert("RGB")
        base = degrade(img, a.level, a.degrade_filter)
        inputs = axis.build_inputs(img, q, **tmpl_kw).to(a.device)
        px_base = axis.build_inputs(base, q, **tmpl_kw)["pixel_values"].to(a.device)
        rms, mse, mse_raw, attn, band, ru = score_one(axis, inputs, px_base, a.groups)
        raw_units = raw_units and ru
        for k, v in zip(("rms", "mse", "mse_raw", "attn", "band"),
                        (rms, mse, mse_raw, attn, band)):
            cols[k].append(v)
        img_id.append(int(i))
        n_groups.append(int(rms.size))
        offs.append(offs[-1] + int(rms.size))
        hw.append([int(v) for v in inputs["image_grid_thw"][0].tolist()])
        if (j + 1) % 10 == 0:
            print(f"  {j + 1}/{len(idxs)} images, {offs[-1]} groups "
                  f"({(j + 1) / (time.perf_counter() - t0):.2f} img/s)", flush=True)

    slug = a.model.split("/")[-1].lower()
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    name = (f"{a.dataset}_{slug}_g{a.groups}_l{a.level}{a.degrade_filter}"
            f"_n{len(idxs)}.npz")
    path = os.path.join(ROOT, a.out, name)
    meta = {"dataset": a.dataset, "model": a.model, "family": a.family, "groups": a.groups,
            "level": a.level, "degrade_filter": a.degrade_filter, "n_images": len(idxs),
            "image_mean": [float(v) for v in ip.image_mean],
            "image_std": [float(v) for v in ip.image_std],
            "rescale_factor": float(ip.rescale_factor),
            "rms_units": "raw01" if raw_units else "normalised",
            "attn": "full-depth eager received-attention layer mean, N x mean (mean 1)",
            "pscore_note": "band 0 of the deferred arm ranks on the ENERGY factor alone"}
    np.savez_compressed(path, offsets=np.asarray(offs, np.int64),
                        image_id=np.asarray(img_id, np.int64),
                        n_groups=np.asarray(n_groups, np.int64),
                        grid_thw=np.asarray(hw, np.int64),
                        meta=np.asarray(json.dumps(meta)),
                        **{k: np.concatenate(v) for k, v in cols.items()})
    print(f"wrote {path}: {len(idxs)} images, {offs[-1]} merge groups, "
          f"rms units {meta['rms_units']}, {time.perf_counter() - t0:.1f}s")
    print("PSCORE_DUMP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
