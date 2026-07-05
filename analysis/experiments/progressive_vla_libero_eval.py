"""
progressive_vla_libero_eval.py

Phase 3/5 real evaluation: runs the progressive VLA prefill system (appcorr/models/openvla/
progressive_model.py) *inside* an actual LIBERO control loop (real env stepping, real success/failure
detection) rather than comparing a single decoded action's L2 distance against ground truth -- this is
what actually answers "does partial correction still let the robot complete the task."

Modes compared, per control step (each step gets a fresh camera frame, matching real deployment --
no state carried across steps):
  - stock:          the real, unmodified predict_action() on the full-resolution frame (baseline).
  - approx_only:    approx_forward() on a blurred (1/4-res) frame only, 0% correction (worst case --
                    "what if the high-res residual never arrives in time").
  - corrected_50:   approx_forward(blur) then correct_forward() on 50% of patches, in *sequential*
                    (causally-monotonic) order -- per Phase 3's finding that scattered/residual-rank
                    order leaves permanent staleness across multiple correction rounds, sequential
                    prefix order was confirmed to reach exact convergence at 100%, so it's the correct
                    choice for a partial, still-improving-over-time schedule too.

Run (from repo root, in the `openvla` conda env, with LIBERO's software-EGL env vars set):
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 MUJOCO_EGL_ALLOW_ANY_DEVICE=1 USE_TF=0 USE_TORCH=1 \
    python analysis/experiments/progressive_vla_libero_eval.py \
        --checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --task-suite libero_spatial --task-id 0 --num-trials 2 --max-steps 150
"""

import argparse
import os
import sys
import time
from pathlib import Path

APPCORR_ROOT = Path(__file__).resolve().parents[2]
OPENVLA_ROOT = Path("/NHNHOME/share/cjpark/openvla")
for p in (APPCORR_ROOT, OPENVLA_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from appcorr.models.openvla.progressive_model import OpenVLAProgressiveModel
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_dummy_action
from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--task-suite", type=str, default="libero_spatial")
    parser.add_argument("--task-ids", type=str, default="0", help="Comma-separated task ids, or 'all' for every task in the suite.")
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=150, help="Cap on rollout length (real suite max is higher; capped for turnaround time).")
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--blur-factor", type=int, default=4)
    parser.add_argument("--correction-ratio", type=float, default=0.5)
    parser.add_argument("--modes", type=str, default="stock,approx_only,corrected_50")
    parser.add_argument("--exit-metric", type=str, default="neighbor_mass",
                        help="Early-exit confidence metric: max_prob | top2_margin | entropy | neighbor_mass.")
    parser.add_argument("--exit-threshold", type=float, default=0.95,
                        help="Exit when worst-token confidence >= threshold (or <= for entropy).")
    parser.add_argument("--calib-log", type=str, default=None,
                        help="JSONL output path for exitcalib_* modes (per-step stats + bin agreement).")
    return parser.parse_args()


def blur_downup(x: torch.Tensor, factor: int) -> torch.Tensor:
    B, C, H, W = x.shape
    low = F.interpolate(x, size=(H // factor, W // factor), mode="bilinear", align_corners=False)
    return F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)


def per_patch_residual_rms(true_px: torch.Tensor, blur_px: torch.Tensor, patch: int = 14) -> torch.Tensor:
    """AppCorr's `residual_rms` mobile pscore: per-patch RMS of the pixel residual between the true
    image and the blurred base layer (DINOv2 channel group; both towers share the 16x16 grid)."""
    r = (true_px[:, :3] - blur_px[:, :3]).float()
    B, C, H, W = r.shape
    gh, gw = H // patch, W // patch
    r = r.view(B, C, gh, patch, gw, patch)
    return r.pow(2).mean(dim=(1, 3, 5)).sqrt().view(B, gh * gw)  # [B, 256]


def parse_ratio(mode: str, prefix: str, fallback: float) -> float:
    suffix = mode[len(prefix):].lstrip("_")
    return float(suffix) / 100.0 if suffix else fallback


def fused_pscore(model, true_px, blur_px) -> torch.Tensor:
    """AppCorr's fused pscore: layer-averaged CLS attention (server pscore) x per-patch pixel
    residual RMS (mobile pscore), pscore_fusion='multiply'. [1, 256]."""
    attn_score = model.cache_feature["dino_cls_attn_layermean"].float()
    resid_score = per_patch_residual_rms(true_px.float(), blur_px.float()).to(attn_score.device)
    return attn_score * resid_score


def attnres_patch_idx(model, true_px, blur_px, ratio: float) -> torch.Tensor:
    """Top-k selection on the fused pscore (AppCorr's token_keep_ratio path, block.py's
    _select_patch_keep_mask). Single correction round, so importance-scattered selection carries
    no multi-round staleness (see Phase 3 audit)."""
    k = max(1, int(round(256 * ratio)))
    fused = fused_pscore(model, true_px, blur_px)
    return fused.topk(k, dim=1).indices.squeeze(0)


def attnres_threshold_patch_idx(model, true_px, blur_px, threshold: float) -> torch.Tensor:
    """Hard-threshold selection on the fused pscore -- AppCorr's `token_keep_thres` path
    (block.py's _select_patch_keep_mask: `combined_patch_scores >= token_keep_thres`, mutually
    exclusive with the ratio/top-k path, NO min-keep floor in the original). Patch count is
    data-dependent here, unlike the fixed-ratio modes."""
    fused = fused_pscore(model, true_px, blur_px)
    idx = (fused[0] >= threshold).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        idx = fused.topk(1, dim=1).indices.squeeze(0)  # AppCorr has no explicit floor; avoid a
        # fully-empty correction round only as a degenerate-input safety net, not the normal path.
    return idx


def aggregate_confidence(stats: dict, metric: str) -> float:
    """Worst-token aggregation across the 7 action tokens (min for confidence-like metrics,
    max for entropy) -- the conservative analog of decide_exit's per-sample thresholds."""
    vals = [t[metric] for t in stats["per_token"]]
    return max(vals) if metric == "entropy" else min(vals)


def confidence_passes(conf: float, metric: str, threshold: float) -> bool:
    return conf <= threshold if metric == "entropy" else conf >= threshold


def decide_action(model: OpenVLAProgressiveModel, mode: str, image: Image.Image, task_description: str,
                   correction_ratio: float, blur_factor: int, exit_cfg: dict = None,
                   calib_records: list = None):
    """Returns (action, info) where info may carry {'exited': bool} for earlyexit modes."""
    info = {}
    if mode == "stock":
        model.start_session(image, task_description, center_crop=True)
        px = model.reference_pixel_values.to(device=model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            action = model.vla.predict_action(
                input_ids=model.input_ids, pixel_values=px, unnorm_key=model.unnorm_key, do_sample=False
            )
        return action, info

    model.start_session(image, task_description, center_crop=True)
    true_px = model.reference_pixel_values.to(device=model.device, dtype=torch.bfloat16)
    blur_px = blur_downup(true_px.float(), factor=blur_factor).to(dtype=torch.bfloat16)
    with torch.no_grad():
        model.approx_forward(blur_px)
        if mode == "approx_only":
            return model.decode_action(), info

        if mode.startswith("corrected"):
            ratio = parse_ratio(mode, "corrected", correction_ratio)
            k = max(1, int(round(256 * ratio)))
            patch_idx = torch.arange(0, k, device=model.device)  # sequential/prefix order
            model.correct_forward(true_px, patch_idx)
            return model.decode_action(), info

        if mode.startswith("attnres"):
            ratio = parse_ratio(mode, "attnres", correction_ratio)
            patch_idx = attnres_patch_idx(model, true_px, blur_px, ratio)
            model.correct_forward(true_px, patch_idx)
            return model.decode_action(), info

        if mode.startswith("attnthresh"):
            # Mode name carries the threshold scaled by 1e6, e.g. "attnthresh_200" -> 0.0002 --
            # AppCorr's own token_keep_thres configs (0.0001-0.002) live in this same integer range,
            # though on a differently-scaled/differently-fused pscore (see module docstring context
            # in progressive_vla_libero_eval's fused_pscore -- ours is attn x residual_rms,
            # multiply fusion, calibrated separately; only the mechanism transfers 1:1).
            suffix = mode[len("attnthresh"):].lstrip("_")
            threshold = float(suffix) * 1e-6
            patch_idx = attnres_threshold_patch_idx(model, true_px, blur_px, threshold)
            info["num_patches"] = int(patch_idx.numel())
            model.correct_forward(true_px, patch_idx)
            return model.decode_action(), info

        if mode.startswith("exitcalib"):
            # Calibration: compute BOTH the approx-decode (with stats) and the corrected decode,
            # log confidence-vs-agreement, and ACT with the corrected action (so the visited state
            # distribution matches the attnres_* policy the exit would fall back to).
            ratio = parse_ratio(mode, "exitcalib", correction_ratio)
            approx_action, approx_stats = model.decode_action(return_stats=True)
            patch_idx = attnres_patch_idx(model, true_px, blur_px, ratio)
            model.correct_forward(true_px, patch_idx)
            corr_action, corr_stats = model.decode_action(return_stats=True)
            bins_a, bins_c = np.array(approx_stats["bins"]), np.array(corr_stats["bins"])
            if calib_records is not None:
                calib_records.append({
                    "confidence": {m: aggregate_confidence(approx_stats, m)
                                    for m in ("max_prob", "top2_margin", "entropy", "neighbor_mass")},
                    "max_bin_dist": int(np.abs(bins_a - bins_c).max()),
                    "exact_agree": bool((bins_a == bins_c).all()),
                })
            return corr_action, info

        if mode.startswith("earlyexit"):
            ratio = parse_ratio(mode, "earlyexit", correction_ratio)
            approx_action, approx_stats = model.decode_action(return_stats=True)
            conf = aggregate_confidence(approx_stats, exit_cfg["metric"])
            if confidence_passes(conf, exit_cfg["metric"], exit_cfg["threshold"]):
                info["exited"] = True
                return approx_action, info
            info["exited"] = False
            patch_idx = attnres_patch_idx(model, true_px, blur_px, ratio)
            model.correct_forward(true_px, patch_idx)
            return model.decode_action(), info

    raise ValueError(f"Unknown mode: {mode}")


def run_episode(model, mode, task_suite_name, task_id, max_steps, num_steps_wait, correction_ratio,
                blur_factor, exit_cfg=None, calib_records=None, exit_counts=None, patch_counts=None):
    if exit_counts is None:
        exit_counts = [0, 0]  # [exits, decisions]
    if patch_counts is None:
        patch_counts = []
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    env, task_description = get_libero_env(task, "openvla", resolution=256)
    env.reset()
    obs = env.set_init_state(initial_states[0])

    t = 0
    done = False
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue

        img = get_libero_image(obs, 224)
        image = Image.fromarray(img).convert("RGB")

        action, step_info = decide_action(model, mode, image, task_description, correction_ratio,
                                          blur_factor, exit_cfg=exit_cfg, calib_records=calib_records)
        if "exited" in step_info:
            exit_counts[0] += int(step_info["exited"])
            exit_counts[1] += 1
        if "num_patches" in step_info:
            patch_counts.append(step_info["num_patches"])
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)

        obs, reward, done, info = env.step(action.tolist())
        if done:
            break
        t += 1

    env.close()
    return done, t


def resolve_task_ids(task_suite_name: str, task_ids_arg: str) -> list[int]:
    if task_ids_arg.strip().lower() == "all":
        from libero.libero import benchmark

        task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
        return list(range(task_suite.n_tasks))
    return [int(x) for x in task_ids_arg.split(",")]


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modes = args.modes.split(",")
    task_ids = resolve_task_ids(args.task_suite, args.task_ids)
    print(f"[eval] Task ids: {task_ids}")

    print(f"[eval] Loading {args.checkpoint}...")
    model = OpenVLAProgressiveModel(args.checkpoint, device, unnorm_key=args.task_suite)

    exit_cfg = {"metric": args.exit_metric, "threshold": args.exit_threshold}
    calib_records: list = []
    results = {mode: [] for mode in modes}  # mode -> list of (task_id, success)
    exit_totals = {mode: [0, 0] for mode in modes}  # mode -> [exits, decisions]
    patch_totals = {mode: [] for mode in modes}  # mode -> list of per-step patch counts (attnthresh)
    for mode in modes:
        print(f"\n[eval] === Mode: {mode} ===")
        if mode.startswith("earlyexit"):
            print(f"    exit metric={args.exit_metric} threshold={args.exit_threshold}")
        for task_id in task_ids:
            for trial in range(args.num_trials):
                t0 = time.time()
                success, steps = run_episode(
                    model, mode, args.task_suite, task_id, args.max_steps, args.num_steps_wait,
                    args.correction_ratio, args.blur_factor,
                    exit_cfg=exit_cfg, calib_records=calib_records, exit_counts=exit_totals[mode],
                    patch_counts=patch_totals[mode],
                )
                dt = time.time() - t0
                results[mode].append((task_id, success))
                extra = ""
                if exit_totals[mode][1] > 0:
                    extra = f" exit_frac_so_far={exit_totals[mode][0] / exit_totals[mode][1]:.2f}"
                print(f"    task {task_id} trial {trial + 1}/{args.num_trials}: success={success} steps={steps} wall={dt:.1f}s{extra}")
                sys.stdout.flush()

    if args.calib_log and calib_records:
        import json

        with open(args.calib_log, "w") as f:
            for rec in calib_records:
                f.write(json.dumps(rec) + "\n")
        print(f"\n[eval] Wrote {len(calib_records)} calibration records to {args.calib_log}")

    print("\n[eval] === Summary: success rate by mode (aggregated across all tasks/trials) ===")
    for mode in modes:
        successes = [s for _, s in results[mode]]
        rate = sum(successes) / len(successes) if successes else float("nan")
        exits, decisions = exit_totals[mode]
        exit_str = f"  exit_frac={exits / decisions:.2f} ({exits}/{decisions} steps)" if decisions else ""
        patches = patch_totals[mode]
        patch_str = f"  mean_patches={np.mean(patches):.1f}/256 ({np.mean(patches)/256:.1%})" if patches else ""
        print(f"    {mode:16s}  {sum(successes)}/{len(successes)}  ({rate:.0%}){exit_str}{patch_str}")

    print("\n[eval] === Per-task breakdown ===")
    for mode in modes:
        per_task: dict[int, list[bool]] = {}
        for task_id, success in results[mode]:
            per_task.setdefault(task_id, []).append(success)
        line = "  ".join(f"t{tid}={sum(v)}/{len(v)}" for tid, v in sorted(per_task.items()))
        print(f"    {mode:16s}  {line}")


if __name__ == "__main__":
    main()
