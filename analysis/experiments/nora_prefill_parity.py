"""Exactness gates for NORA vision correction and grouped causal prefill.

By default this captures one initial observation from each requested real
LIBERO task.  Rollout evaluation must not start unless exact vision,
all-token correction, and stock-vs-grouped FAST token checks pass.
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

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
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--base-factor", type=int, default=4)
    return parser.parse_args()


def resolve_snapshot(checkpoint: str, snapshot_dir: str | None) -> str:
    if snapshot_dir:
        return snapshot_dir
    name = checkpoint.replace("/", "--")
    matches = sorted(
        glob.glob(f"/NHNHOME/huggingface/hub/models--{name}/snapshots/*")
    )
    return matches[-1] if matches else checkpoint


def capture_frames(task_suite, task_ids):
    from experiments.robot.libero.libero_utils import get_libero_env

    frames = []
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        states = task_suite.get_task_init_states(task_id)
        env, instruction = get_libero_env(task, "openvla", resolution=256)
        try:
            env.reset()
            obs = env.set_init_state(states[0])
            frame = np.ascontiguousarray(
                obs["agentview_image"][::-1, ::-1]
            )
            frames.append((task_id, frame, instruction))
        finally:
            env.close()
    return frames


def main():
    args = parse_args()
    from libero.libero import benchmark

    from appcorr.models.nora import NoraProgressiveModel

    task_ids = [int(value) for value in args.task_ids.split(",") if value]
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    frames = capture_frames(suite, task_ids)
    model = NoraProgressiveModel(
        resolve_snapshot(args.checkpoint, args.snapshot_dir),
        device=args.device,
    )

    # Vision gates use a real frame and compare the exact tensors, not just the
    # downstream action.
    _, frame, instruction = frames[0]
    full = model.prepare_inputs(frame, instruction)
    pixel_values = full["pixel_values"].to(model.model.visual.dtype)
    grid = full["image_grid_thw"]
    stock_vision = model.model.visual(pixel_values, grid_thw=grid)
    approx_vision, _ = model.vision.approx(pixel_values, grid)
    if not torch.equal(stock_vision, approx_vision):
        raise AssertionError(
            "exact-input approx vision differs from stock: "
            f"max={float((stock_vision - approx_vision).abs().max())}"
        )

    base = model.prepare_inputs(
        model.low_res_base(frame, args.base_factor),
        instruction,
    )
    _, cache = model.vision.approx(
        base["pixel_values"].to(model.model.visual.dtype),
        base["image_grid_thw"],
    )
    corrected, _ = model.vision.correct(
        pixel_values,
        grid,
        torch.arange(stock_vision.shape[0], device=model.device),
        cache,
    )
    if not torch.equal(stock_vision, corrected):
        raise AssertionError(
            "one-shot all-cell correction differs from stock: "
            f"max={float((stock_vision - corrected).abs().max())}"
        )
    print("[PASS] vision exact and one-shot all-cell correction", flush=True)

    for task_id, image, task_instruction in frames:
        stock = model.predict_stock(image, task_instruction)
        grouped = model.predict_grouped_full(
            image,
            task_instruction,
            num_groups=args.num_groups,
        )
        same = torch.equal(
            stock.action_token_ids,
            grouped.action_token_ids,
        )
        max_action = float(
            np.max(np.abs(stock.actions - grouped.actions))
        )
        print(
            f"[parity] task={task_id} tokens={same} "
            f"n={stock.action_token_ids.numel()} max_action={max_action:.3g}",
            flush=True,
        )
        if not same or max_action != 0.0:
            raise AssertionError(f"grouped-full parity failed on task {task_id}")
    print(f"[PASS] grouped-full parity on {len(frames)} real frames", flush=True)


if __name__ == "__main__":
    main()
