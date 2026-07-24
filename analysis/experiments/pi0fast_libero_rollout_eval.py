"""
pi0fast_libero_rollout_eval.py

T1-style REAL LIBERO rollout evaluation for pi0-FAST progressive vision: steps the actual LIBERO
sim (via lerobot's LiberoEnv, so the env->policy preprocessing -- 180-degree image rotation, the
[eef_pos, quat->axisangle, gripper] state -- exactly matches training) and reports task SUCCESS
RATE, comparing:
  - stock : full-resolution vision (baseline; embed_image untouched).
  - prog  : progressive vision -- SigLIP low-res base approx + correct only `--keep` fraction of
            patches (top-to-bottom), injected transparently by overriding embed_image, so the whole
            lerobot select_action / FAST-decode pipeline runs unchanged on the approximated features.

Requires the double-scaling fix (installed by importing the progressive model) and EGL rendering on
the working device:
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 MUJOCO_EGL_ALLOW_ANY_DEVICE=1 TORCHDYNAMO_DISABLE=1 \
    python analysis/experiments/pi0fast_libero_rollout_eval.py \
        --task-suite libero_spatial --task-ids 0,1 --num-trials 10 --modes stock,prog --keep 0.5

Note: the FAST detokenizer occasionally emits a garbage token -> an absurd action; actions are
clipped to [-1, 1] (the LIBERO delta-control range) before stepping, matching how these actions are
consumed in practice.
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
    p.add_argument("--task-suite", type=str, default="libero_spatial")
    p.add_argument("--task-ids", type=str, default="0", help="Comma-separated task ids, or 'all'.")
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--trial-offset", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None, help="Default: the suite's own max.")
    p.add_argument("--modes", type=str, default="stock,prog")
    p.add_argument("--keep", type=float, default=0.5, help="Corrected-patch fraction for 'prog'.")
    p.add_argument("--base-factor", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--results-json", type=str, default=None)
    return p.parse_args()


def batch_robot_state(o):
    rs = o.get("observation.robot_state")
    if rs is None:
        return o
    def b(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            return x.unsqueeze(0)
        if isinstance(x, dict):
            return {k: b(v) for k, v in x.items()}
        return x
    o["observation.robot_state"] = b(rs)
    return o


def make_progressive_embed_image(M, keep, base_factor):
    """embed_image override: run SigLIP base approx + keep-fraction patch correct on each image.
    Called once per image (3x) per predict; tags roll0/1/2, cache cleared at the start of each."""
    ctr = [0]
    def f(img):
        i = ctr[0] % 3
        if i == 0:
            M.cache_feature = {}
        tag = f"roll{i}"
        M.siglip.approx_forward(M._base_pixel(img, base_factor), M.cache_feature, tag)
        npatch = M.cache_feature[f"{tag}_layer0_kv"].shape[2]
        k = max(1, int(round(keep * npatch)))
        feat, _ = M.siglip.correct_forward(img, torch.arange(k, device=M.device), M.cache_feature, tag)
        ctr[0] += 1
        return M._project(feat)
    return f


def run_trial(M, env_pre, task_suite, suite_name, task_id, task_str, trial, max_steps):
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation

    env = LiberoEnv(task_suite, task_id, suite_name, obs_type="pixels_agent_pos",
                    episode_index=trial, episode_length=max_steps)
    obs, _ = env.reset()
    M.pol.reset()
    success = False
    try:
        for _ in range(max_steps):
            o = preprocess_observation(obs)
            o = batch_robot_state(o)
            o = env_pre(o)
            o["task"] = [task_str]
            o = M.pre(o)
            with torch.inference_mode():
                action = M.post(M.pol.select_action(o))
            a = action.to("cpu").numpy()
            if a.ndim == 2:
                a = a[0]
            a = np.clip(a, -1.0, 1.0)   # guard against FAST-detokenizer spikes
            obs, _, terminated, _, info = env.step(a)
            if info.get("is_success"):
                success = True
                break
            if terminated:
                break
    finally:
        env.close()
    return success


def main():
    args = parse_args()
    from libero.libero import benchmark
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.env_processor import LiberoProcessorStep
    from appcorr.models.pi0fast.progressive_model import Pi0FastProgressiveModel

    device = torch.device(args.device)
    M = Pi0FastProgressiveModel(args.checkpoint, device)
    env_pre = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    modes = args.modes.split(",")
    if args.task_ids == "all":
        task_ids = list(range(task_suite.n_tasks))
    else:
        task_ids = [int(x) for x in args.task_ids.split(",")]
    max_steps = args.max_steps or getattr(M, "_max_steps", None) or 300

    results = {m: {"successes": 0, "trials": 0, "per_task": {}} for m in modes}
    for task_id in task_ids:
        task_str = task_suite.get_task(task_id).language
        for mode in modes:
            if mode == "prog":
                M.m.paligemma_with_expert.embed_image = make_progressive_embed_image(
                    M, args.keep, args.base_factor)
            else:
                M.m.paligemma_with_expert.embed_image = M._orig_embed_image
            succ = 0
            for tr in range(args.trial_offset, args.trial_offset + args.num_trials):
                s = run_trial(M, env_pre, task_suite, args.task_suite, task_id, task_str, tr, max_steps)
                succ += int(s)
                print(f"[{mode}] task{task_id} trial{tr}: {'SUCCESS' if s else 'fail'} "
                      f"(running {succ}/{tr - args.trial_offset + 1})", flush=True)
            M.m.paligemma_with_expert.embed_image = M._orig_embed_image
            results[mode]["successes"] += succ
            results[mode]["trials"] += args.num_trials
            results[mode]["per_task"][task_id] = succ / args.num_trials

    print("\n=== pi0-FAST LIBERO rollout eval ===")
    print(f"suite={args.task_suite} tasks={task_ids} trials/task={args.num_trials} "
          f"max_steps={max_steps} keep={args.keep}")
    for mode in modes:
        r = results[mode]
        sr = r["successes"] / max(1, r["trials"])
        print(f"  {mode:6s}: success {r['successes']}/{r['trials']} = {sr*100:.1f}%   per-task={r['per_task']}")

    if args.results_json:
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.results_json}")


if __name__ == "__main__":
    main()
