"""
Evaluate the direct-loaded OpenVLA-OFT model on real LIBERO rollouts.

This is the OFT counterpart of ``openvla_offload_libero_eval.py``.  It keeps one
model resident on one GPU and evaluates three inference modes:

* ``full``: stock-resolution images through the exact OFT forward.
* ``pipelined``: 1/4-resolution bases, full vision correction, and four-group
  causal LLM prefill, followed by OFT's bidirectional action block.
* ``approx``: 1/4-resolution images without vision correction.

OFT predicts an eight-action chunk.  The chunk is queued and executed open-loop,
matching the released OFT LIBERO evaluation policy.  Results are appended to a
JSONL file after every episode, so an interrupted shard can be resumed safely.
"""

import argparse
import glob
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


APPCORR_ROOT = Path(__file__).resolve().parents[2]
OPENVLA_ROOT = Path("/NHNHOME/share/cjpark/openvla")
for path in (APPCORR_ROOT, OPENVLA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="moojink/openvla-7b-oft-finetuned-libero-spatial",
    )
    parser.add_argument("--snapshot-dir", default=None)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--unnorm-key", default="libero_spatial_no_noops")
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--num-trials", type=int, default=25)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--base-factor", type=int, default=4)
    parser.add_argument("--schedules", default="full,pipelined,approx")
    parser.add_argument("--result-jsonl", type=Path, required=True)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip episode keys already present in result-jsonl.",
    )
    return parser.parse_args()


def resolve_snapshot(checkpoint: str, snapshot_dir: Optional[str]) -> str:
    if snapshot_dir:
        return snapshot_dir
    repo_dir = checkpoint.replace("/", "--")
    matches = sorted(
        glob.glob(f"/NHNHOME/huggingface/hub/models--{repo_dir}/snapshots/*/")
    )
    if not matches:
        raise FileNotFoundError(
            f"No cached snapshot for {checkpoint}; pass --snapshot-dir explicitly"
        )
    return matches[-1]


def low_res_base(image: np.ndarray, factor: int) -> np.ndarray:
    pil = Image.fromarray(image)
    width, height = pil.size
    down = pil.resize(
        (max(width // factor, 1), max(height // factor, 1)),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(
        down.resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def observation_inputs(obs, quat2axisangle) -> Tuple[List[np.ndarray], np.ndarray]:
    # Matches the already validated direct-OFT rollout path.
    agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(
        obs["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    proprio = np.concatenate(
        [
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]
    )
    return [agent, wrist], proprio


def predict_chunk(model, schedule: str, images, proprio, args):
    started = time.perf_counter()
    if schedule == "full":
        actions = model.predict_action_exact(images, proprio)
    elif schedule == "pipelined":
        bases = [low_res_base(image, args.base_factor) for image in images]
        actions = model.predict_action_progressive(
            images,
            proprio,
            base_images_np=bases,
            num_groups=args.num_groups,
        )
    elif schedule == "approx":
        bases = [low_res_base(image, args.base_factor) for image in images]
        actions = model.predict_action_exact(bases, proprio)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return actions, time.perf_counter() - started


def completed_keys(path: Path) -> set:
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("record_type") == "episode":
            keys.add(
                (
                    row.get("schedule"),
                    int(row.get("task_id")),
                    int(row.get("init_state_idx")),
                )
            )
    return keys


def append_jsonl(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def run_episode(model, task_suite, task_id: int, init_state_idx: int,
                schedule: str, args, quat2axisangle):
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
    )
    from experiments.robot.robot_utils import (
        invert_gripper_action,
        normalize_gripper_action,
    )

    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    wall_started = time.perf_counter()
    inference_s = 0.0
    inference_calls = 0
    try:
        env.reset()
        obs = env.set_init_state(
            initial_states[init_state_idx % len(initial_states)]
        )
        model.start_session(task_description)
        action_queue = deque()
        step = 0
        success = False
        while step < args.max_steps + args.num_steps_wait:
            if step < args.num_steps_wait:
                obs, _, success, _ = env.step(
                    get_libero_dummy_action("openvla")
                )
                step += 1
                continue
            if not action_queue:
                images, proprio = observation_inputs(obs, quat2axisangle)
                chunk, elapsed = predict_chunk(
                    model, schedule, images, proprio, args
                )
                inference_s += elapsed
                inference_calls += 1
                action_queue.extend(chunk)
            action = normalize_gripper_action(
                np.asarray(action_queue.popleft()), binarize=True
            )
            action = invert_gripper_action(action)
            obs, _, success, _ = env.step(action.tolist())
            if success:
                break
            step += 1
    finally:
        env.close()
    return {
        "success": bool(success),
        "steps": int(step),
        "wall_s": time.perf_counter() - wall_started,
        "inference_s": inference_s,
        "inference_calls": inference_calls,
    }


def main():
    args = parse_args()
    schedules = [value.strip() for value in args.schedules.split(",") if value]
    valid_schedules = {"full", "pipelined", "approx"}
    unknown = set(schedules) - valid_schedules
    if unknown:
        raise ValueError(f"Unknown schedules: {sorted(unknown)}")

    import torch
    from libero.libero import benchmark

    from appcorr.models.openvla.progressive_model_oft import (
        OFTProgressiveModel,
        quat2axisangle,
    )

    snapshot = resolve_snapshot(args.checkpoint, args.snapshot_dir)
    print(f"[driver] checkpoint={args.checkpoint}", flush=True)
    print(f"[driver] snapshot={snapshot}", flush=True)
    print(
        f"[driver] suite={args.task_suite} schedules={schedules} "
        f"trials={args.trial_start}..{args.trial_start + args.num_trials - 1}",
        flush=True,
    )
    model = OFTProgressiveModel(
        args.checkpoint,
        torch.device("cuda:0"),
        oft_snapshot_dir=snapshot,
        unnorm_key=args.unnorm_key,
    )
    print("[driver] model loaded", flush=True)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    if args.task_ids.strip().lower() == "all":
        task_ids = list(range(task_suite.n_tasks))
    else:
        task_ids = [int(value) for value in args.task_ids.split(",")]

    done_keys = set() if args.no_resume else completed_keys(args.result_jsonl)
    total_requested = len(schedules) * len(task_ids) * args.num_trials
    total_pending = sum(
        (schedule, task_id, args.trial_start + trial) not in done_keys
        for schedule in schedules
        for task_id in task_ids
        for trial in range(args.num_trials)
    )
    print(
        f"[driver] requested={total_requested} already_done="
        f"{total_requested - total_pending} pending={total_pending}",
        flush=True,
    )

    run_started = time.perf_counter()
    completed_now = 0
    summaries = {}
    for schedule in schedules:
        successes = 0
        episodes = 0
        wall_total = 0.0
        for task_id in task_ids:
            for trial in range(args.num_trials):
                init_state_idx = args.trial_start + trial
                key = (schedule, task_id, init_state_idx)
                if key in done_keys:
                    continue
                result = run_episode(
                    model,
                    task_suite,
                    task_id,
                    init_state_idx,
                    schedule,
                    args,
                    quat2axisangle,
                )
                row = {
                    "record_type": "episode",
                    "schedule": schedule,
                    "task_id": task_id,
                    "trial": trial + 1,
                    "init_state_idx": init_state_idx,
                    "base_factor": args.base_factor,
                    "num_groups": args.num_groups,
                    **result,
                }
                append_jsonl(args.result_jsonl, row)
                successes += int(result["success"])
                episodes += 1
                wall_total += result["wall_s"]
                completed_now += 1
                elapsed = time.perf_counter() - run_started
                mean_s = elapsed / completed_now
                eta_h = mean_s * (total_pending - completed_now) / 3600
                print(
                    f"[{schedule}] task {task_id} init {init_state_idx}: "
                    f"success={result['success']} steps={result['steps']} "
                    f"wall={result['wall_s']:.1f}s "
                    f"infer={result['inference_s']:.1f}s/"
                    f"{result['inference_calls']} "
                    f"progress={completed_now}/{total_pending} eta={eta_h:.1f}h",
                    flush=True,
                )
        summaries[schedule] = {
            "successes": successes,
            "episodes_run_now": episodes,
            "wall_s": wall_total,
        }

    for schedule, summary in summaries.items():
        episodes = summary["episodes_run_now"]
        rate = summary["successes"] / episodes if episodes else float("nan")
        print(
            f"[summary] {schedule}: {summary['successes']}/{episodes} "
            f"({rate:.1%}) wall={summary['wall_s'] / 3600:.2f}h",
            flush=True,
        )


if __name__ == "__main__":
    main()
