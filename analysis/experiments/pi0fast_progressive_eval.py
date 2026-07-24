"""
pi0fast_progressive_eval.py

Offline evaluation of pi0-FAST progressive vision (appcorr/models/pi0fast/progressive_model.py) on
real LIBERO frames from a lerobot dataset. Mirrors the OFT/OpenVLA progressive evals, adapted to
pi0-FAST's specifics:

  - The CORE technique for pi0-FAST is progressive *vision* (SigLIP low-res base + per-group patch
    correct); the bidirectional PaliGemma prefix + FAST decode run in full via lerobot (a causal /
    block-causal chunked LLM prefill is not lossless for a bidirectional prefix -- measured
    +12% CE). So this driver sweeps the vision compression, not an LLM schedule.

  - Metrics per frame, per config:
      * action-parity L1 vs the exact (full-correct) prediction -- the lossless check
        (progressive @ 100% correct must equal exact, and equal stock lerobot: |diff|=0).
      * recompute rate = fraction of patches actually corrected (the compute the vision tower spends
        beyond the base approx).
      * teacher-forced next-action-token cross-entropy / argmax accuracy against the dataset's GT
        action chunk -- the model-quality signal that does NOT depend on the (greedy, and on this
        checkpoint fairly degenerate) free-running decode.

Configs swept: exact (100% correct), and progressive with a `--keep` fraction of patches corrected
(top-to-bottom groups) over a low-res base (`--base-factor`).

Run (pi0fast conda env; disable dynamo to avoid a triton inductor cache crash):
    TORCHDYNAMO_DISABLE=1 python analysis/experiments/pi0fast_progressive_eval.py \
        --checkpoint lerobot/pi0fast-libero --dataset HuggingFaceVLA/libero \
        --num-frames 20 --keep 1.0,0.5,0.25 --results-json out.json
"""

import argparse
import json
import sys
from pathlib import Path

APPCORR_ROOT = Path(__file__).resolve().parents[2]
if str(APPCORR_ROOT) not in sys.path:
    sys.path.insert(0, str(APPCORR_ROOT))

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="lerobot/pi0fast-libero")
    p.add_argument("--dataset", type=str, default="HuggingFaceVLA/libero")
    p.add_argument("--episodes", type=str, default="0", help="Comma-separated episode ids.")
    p.add_argument("--num-frames", type=int, default=20, help="Frames sampled (strided) per episode.")
    p.add_argument("--num-groups", type=int, default=4, help="Top-to-bottom correction groups per image.")
    p.add_argument("--base-factor", type=int, default=4, help="Low-res base downsample factor.")
    p.add_argument("--keep", type=str, default="1.0,0.5,0.25",
                   help="Comma-separated fractions of patches to correct (1.0 = lossless).")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--results-json", type=str, default=None)
    return p.parse_args()


def obs_from_sample(s):
    # Keys MUST match config.image_features (this checkpoint: image/image2 + an auto-padded
    # empty_camera_0). Using the wrong names silently pads the real cameras as "missing" (mask=0),
    # so the model sees no vision and collapses to a constant action -- verified failure mode.
    return {
        "observation.images.image": s["observation.images.image"][None],
        "observation.images.image2": s["observation.images.image2"][None],
        "observation.state": s["observation.state"][None],
        "task": [s["task"]],
    }


@torch.inference_mode()
def teacher_forced_ce(M, obs, gt_action_chunk, features=None):
    """Next-action-token cross-entropy against the GT action chunk (robust to the greedy decode's
    occasional detokenizer spikes). If `features` is given, the (progressive) vision features are
    injected via embed_image so this measures the approximation's effect on the model directly."""
    o = dict(obs)
    o["action"] = gt_action_chunk[None]
    b = M.pre(o)
    if features is None:
        out = M.pol.forward(b)
    else:
        queue = list(features)
        M.m.paligemma_with_expert.embed_image = lambda img: queue.pop(0)
        try:
            out = M.pol.forward(b)
        finally:
            M.m.paligemma_with_expert.embed_image = M._orig_embed_image
    return float(out[0]) if isinstance(out, tuple) else float(out["loss"])


def main():
    args = parse_args()
    from appcorr.models.pi0fast.progressive_model import Pi0FastProgressiveModel
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = torch.device(args.device)
    M = Pi0FastProgressiveModel(args.checkpoint, device)
    H = M.pol.config.n_action_steps
    keeps = [float(x) for x in args.keep.split(",")]
    episodes = [int(x) for x in args.episodes.split(",")]

    results = {}
    for ep in episodes:
        ds = LeRobotDataset(args.dataset, episodes=[ep])
        n = len(ds)
        frame_ids = list(range(0, n, max(1, n // args.num_frames)))[: args.num_frames]
        for fi in frame_ids:
            s = ds[fi]
            obs = obs_from_sample(s)
            gt_chunk = torch.stack([ds[min(fi + k, n - 1)]["action"] for k in range(H)])

            a_exact = M.predict_action_exact(obs)
            gt_l1 = float(np.abs(a_exact - gt_chunk.numpy()).mean())

            npatch = M.tokens_per_image or 256   # SigLIP So400m/224 -> 256 patches/image
            for keep_frac in keeps:
                # compute the (progressive) features ONCE at this compression, reuse for the free
                # decode AND the (robust) teacher-forced CE.
                M.cache_feature = {}
                pixel_list = M._pixel_list(obs)
                if keep_frac >= 1.0:
                    feats = M._features_progressive(pixel_list, args.num_groups, args.base_factor, None)
                    kept = 1.0
                else:
                    k = int(round(keep_frac * npatch))
                    keep_idx = [torch.arange(k, device=device)] * len(pixel_list)
                    feats = M._features_progressive(pixel_list, args.num_groups, args.base_factor, keep_idx)
                    kept = k / npatch
                a = M._run_with_injected(obs, feats)
                ce_k = teacher_forced_ce(M, obs, gt_chunk, features=feats)
                # clip before diffing: the FAST detokenizer occasionally emits a single garbage token
                # -> an absurd action value that would dominate a raw L1 (real actions are ~[-1,1]).
                # The teacher-forced CE is the outlier-free primary metric; this keeps L1 readable.
                l1 = float(np.abs(np.clip(a, -3, 3) - np.clip(a_exact, -3, 3)).mean())
                key = f"keep={keep_frac:.2f}"
                r = results.setdefault(key, {"l1_vs_exact": [], "recompute_rate": [], "tf_ce": [], "exact_l1_vs_gt": []})
                r["l1_vs_exact"].append(l1)
                r["recompute_rate"].append(kept)
                r["tf_ce"].append(ce_k)
                r["exact_l1_vs_gt"].append(gt_l1)

    print("\n=== pi0-FAST progressive vision eval ===")
    print(f"checkpoint={args.checkpoint} dataset={args.dataset} frames={sum(1 for _ in results.get(list(results)[0], {}).get('l1_vs_exact', []))}")
    print(f"{'config':14s} | recompute | L1 vs exact | exact L1 vs GT | teacher-forced CE")
    summary = {}
    for key, r in results.items():
        rc = float(np.mean(r["recompute_rate"]))
        l1 = float(np.mean(r["l1_vs_exact"]))
        ce = float(np.mean(r["tf_ce"]))
        gt = float(np.mean(r["exact_l1_vs_gt"]))
        print(f"{key:14s} | {rc:8.3f}  | {l1:10.6f}  | {gt:12.4f}  | {ce:.3f}")
        summary[key] = {"recompute_rate": rc, "l1_vs_exact": l1, "tf_ce": ce,
                        "exact_l1_vs_gt": gt, "n_frames": len(r["l1_vs_exact"])}

    if args.results_json:
        with open(args.results_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.results_json}")


if __name__ == "__main__":
    main()
