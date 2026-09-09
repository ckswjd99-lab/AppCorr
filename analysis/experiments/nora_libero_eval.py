"""Resume-safe standalone NORA T1 evaluation on real LIBERO rollouts.

The three schedules isolate the prefill experiment:

* ``stock``: released full-resolution NORA inference.
* ``pipelined``: 1/4-resolution Qwen vision base, four progressive vision
  corrections interleaved with append-only causal Qwen prefill.
* ``approx``: released NORA inference on the 1/4-resolution base image.

NORA-long's stock FAST+ decoder emits five actions.  Those actions are queued
and executed open-loop; the action generation implementation is unchanged.
"""

import argparse
import glob
import json
import time
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from nora_libero_runtime import configure_nora_libero_runtime


configure_nora_libero_runtime()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="declare-lab/nora-long-finetuned-libero-spatial",
    )
    parser.add_argument("--snapshot-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--num-trials", type=int, default=25)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--base-factor", type=int, default=4)
    parser.add_argument("--schedules", default="stock,pipelined,approx")
    parser.add_argument("--result-jsonl", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument(
        "--video-every",
        type=int,
        default=0,
        help="Save every Nth episode per task/schedule; zero disables videos.",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def resolve_snapshot(checkpoint: str, snapshot_dir: str | None) -> str:
    if snapshot_dir:
        return snapshot_dir
    name = checkpoint.replace("/", "--")
    matches = sorted(
        glob.glob(f"/NHNHOME/huggingface/hub/models--{name}/snapshots/*")
    )
    return matches[-1] if matches else checkpoint


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
                    row["schedule"],
                    int(row["task_id"]),
                    int(row["init_state_idx"]),
                )
            )
    return keys


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def predict(model, schedule: str, image, instruction: str, args):
    if schedule == "stock":
        return model.predict_stock(image, instruction)
    if schedule == "pipelined":
        return model.predict_pipelined(
            image,
            instruction,
            num_groups=args.num_groups,
            base_factor=args.base_factor,
        )
    if schedule == "approx":
        return model.predict_approx(
            image,
            instruction,
            base_factor=args.base_factor,
        )
    raise ValueError(f"unknown NORA schedule: {schedule}")


def maybe_write_video(args, frames, schedule, task_id, init_state_idx, success):
    if (
        args.video_dir is None
        or args.video_every <= 0
        or init_state_idx % args.video_every != 0
    ):
        return None
    args.video_dir.mkdir(parents=True, exist_ok=True)
    path = args.video_dir / (
        f"{schedule}_task{task_id:02d}_init{init_state_idx:03d}"
        f"_success{int(success)}.mp4"
    )
    imageio.mimwrite(path, frames, fps=20)
    return str(path)


def run_episode(model, suite, task_id, init_state_idx, schedule, args):
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
    )
    from experiments.robot.robot_utils import (
        invert_gripper_action,
        normalize_gripper_action,
    )

    task = suite.get_task(task_id)
    states = suite.get_task_init_states(task_id)
    env, instruction = get_libero_env(task, "openvla", resolution=256)
    queue = deque()
    frames = []
    inference_s = 0.0
    inference_calls = 0
    action_tokens = 0
    started = time.perf_counter()
    success = False
    step = 0
    try:
        env.reset()
        obs = env.set_init_state(states[init_state_idx % len(states)])
        while step < args.max_steps + args.num_steps_wait:
            if step < args.num_steps_wait:
                obs, _, success, _ = env.step(
                    get_libero_dummy_action("openvla")
                )
                step += 1
                continue
            image = np.ascontiguousarray(
                obs["agentview_image"][::-1, ::-1]
            )
            frames.append(image)
            if not queue:
                call_started = time.perf_counter()
                output = predict(model, schedule, image, instruction, args)
                inference_s += time.perf_counter() - call_started
                inference_calls += 1
                action_tokens += int(output.action_token_ids.numel())
                queue.extend(output.actions)
            action = normalize_gripper_action(
                np.asarray(queue.popleft()),
                binarize=True,
            )
            action = invert_gripper_action(action)
            # This extra released-NORA convention maps nonnegative gripper
            # outputs to fully closed before the LIBERO step.
            action[-1] = 1.0 if action[-1] >= 0.0 else action[-1]
            obs, _, success, _ = env.step(action.tolist())
            step += 1
            if success:
                break
    finally:
        env.close()
    video = maybe_write_video(
        args,
        frames,
        schedule,
        task_id,
        init_state_idx,
        success,
    )
    return {
        "success": bool(success),
        "steps": int(step),
        "wall_s": time.perf_counter() - started,
        "inference_s": inference_s,
        "inference_calls": inference_calls,
        "action_tokens": action_tokens,
        "video": video,
    }


def main():
    args = parse_args()
    schedules = [value for value in args.schedules.split(",") if value]
    unknown = set(schedules) - {"stock", "pipelined", "approx"}
    if unknown:
        raise ValueError(f"unknown schedules: {sorted(unknown)}")

    from libero.libero import benchmark

    from appcorr.models.nora import NoraProgressiveModel

    snapshot = resolve_snapshot(args.checkpoint, args.snapshot_dir)
    print(f"[driver] checkpoint={snapshot}", flush=True)
    model = NoraProgressiveModel(snapshot, device=args.device)
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task_ids = (
        list(range(suite.n_tasks))
        if args.task_ids == "all"
        else [int(value) for value in args.task_ids.split(",") if value]
    )
    done = set() if args.no_resume else completed_keys(args.result_jsonl)
    requested = [
        (schedule, task_id, args.trial_start + trial)
        for schedule in schedules
        for task_id in task_ids
        for trial in range(args.num_trials)
    ]
    pending = [key for key in requested if key not in done]
    print(
        f"[driver] requested={len(requested)} done={len(requested)-len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )

    run_started = time.perf_counter()
    completed = 0
    for schedule, task_id, init_state_idx in pending:
        result = run_episode(
            model,
            suite,
            task_id,
            init_state_idx,
            schedule,
            args,
        )
        row = {
            "record_type": "episode",
            "checkpoint": args.checkpoint,
            "task_suite": args.task_suite,
            "schedule": schedule,
            "task_id": task_id,
            "init_state_idx": init_state_idx,
            "base_factor": args.base_factor,
            "num_groups": args.num_groups,
            "action_horizon": model.action_horizon,
            **result,
        }
        append_jsonl(args.result_jsonl, row)
        completed += 1
        elapsed = time.perf_counter() - run_started
        eta_h = (
            elapsed / completed * (len(pending) - completed) / 3600
            if completed
            else float("nan")
        )
        print(
            f"[{schedule}] task={task_id} init={init_state_idx} "
            f"success={result['success']} steps={result['steps']} "
            f"infer={result['inference_s']:.1f}s/{result['inference_calls']} "
            f"progress={completed}/{len(pending)} eta={eta_h:.1f}h",
            flush=True,
        )


if __name__ == "__main__":
    main()
