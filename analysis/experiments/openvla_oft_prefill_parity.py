"""Parity and execution-order gates for interleaved OpenVLA-OFT prefill."""

import argparse
import glob
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np


APPCORR_ROOT = Path(__file__).resolve().parents[2]
for path in (
    APPCORR_ROOT,
    APPCORR_ROOT.parent / "openvla",
):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
if importlib.util.find_spec("libero") is None:
    libero_root = APPCORR_ROOT.parent / "openvla_deps" / "LIBERO"
    if (libero_root / "libero" / "libero").is_dir():
        sys.path.insert(0, str(libero_root))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "2")
os.environ.setdefault("MUJOCO_EGL_ALLOW_ANY_DEVICE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# The local OpenVLA stack imports both TensorFlow and PyTorch unless these are
# fixed before robosuite creates its EGL context; that combination segfaults in
# eglMakeCurrent on the B200 host.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="moojink/openvla-7b-oft-finetuned-libero-spatial",
    )
    parser.add_argument("--snapshot-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--base-factor", type=int, default=4)
    parser.add_argument(
        "--max-grouped-action-error",
        type=float,
        default=0.02,
        help="BF16 grouped-SDPA action tolerance against the full OFT forward.",
    )
    return parser.parse_args()


def resolve_snapshot(checkpoint: str, snapshot_dir: str | None) -> str:
    if snapshot_dir:
        return snapshot_dir
    name = checkpoint.replace("/", "--")
    matches = sorted(
        glob.glob(f"/NHNHOME/huggingface/hub/models--{name}/snapshots/*")
    )
    if not matches:
        raise FileNotFoundError(f"checkpoint is not cached: {checkpoint}")
    return matches[-1]


def low_res_base(image: np.ndarray, factor: int) -> np.ndarray:
    from PIL import Image

    pil = Image.fromarray(image)
    width, height = pil.size
    base = pil.resize(
        (max(width // factor, 1), max(height // factor, 1)),
        Image.Resampling.BILINEAR,
    ).resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(base, dtype=np.uint8)


def capture_frames(suite, task_ids):
    from experiments.robot.libero.libero_utils import get_libero_env

    from appcorr.models.openvla.progressive_model_oft import quat2axisangle

    samples = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        env, instruction = get_libero_env(task, "openvla", resolution=256)
        try:
            env.reset()
            obs = env.set_init_state(states[0])
            images = [
                np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                ),
            ]
            proprio = np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            )
            samples.append((task_id, instruction, images, proprio))
        finally:
            env.close()
    return samples


def assert_interleaved_trace(trace, num_groups: int):
    expected = []
    for image_idx in range(2):
        expected.append(("vision_base", image_idx, None))
        for group_idx in range(num_groups):
            expected.extend(
                [
                    ("vision_correct", image_idx, group_idx),
                    ("llm_prefill", image_idx, group_idx),
                ]
            )
    actual = [
        (row["op"], row["image"], row.get("group"))
        for row in trace
    ]
    if actual != expected:
        raise AssertionError(
            f"vision/LLM operations are not interleaved as required:\n{actual}"
        )
    for row in trace:
        if row["op"] == "vision_correct" and row["patches"] != 64:
            raise AssertionError(f"wrong vision group size: {row}")
        if row["op"] == "llm_prefill" and row["tokens"] != 64:
            raise AssertionError(f"wrong LLM group size: {row}")


def main():
    args = parse_args()
    import torch
    from libero.libero import benchmark

    from appcorr.models.openvla.progressive_model_oft import (
        OFTProgressiveModel,
    )

    task_ids = [int(value) for value in args.task_ids.split(",") if value]
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    samples = capture_frames(suite, task_ids)
    model = OFTProgressiveModel(
        args.checkpoint,
        torch.device(args.device),
        resolve_snapshot(args.checkpoint, args.snapshot_dir),
        unnorm_key=f"{args.task_suite}_no_noops",
    )

    maximum = 0.0
    for task_id, instruction, images, proprio in samples:
        model.start_session(instruction)
        exact = model.predict_action_exact(images, proprio)
        grouped = model.predict_action_progressive(
            images,
            proprio,
            base_images_np=None,
            num_groups=args.num_groups,
        )
        error = float(np.max(np.abs(exact - grouped)))
        maximum = max(maximum, error)
        print(
            f"[grouped-full] task={task_id} max_action_error={error:.6f}",
            flush=True,
        )
        if error > args.max_grouped_action_error:
            raise AssertionError(
                f"grouped-full parity failed on task {task_id}: {error}"
            )

    task_id, instruction, images, proprio = samples[0]
    model.start_session(instruction)
    model.predict_action_progressive(
        images,
        proprio,
        base_images_np=[
            low_res_base(image, args.base_factor) for image in images
        ],
        num_groups=args.num_groups,
    )
    assert_interleaved_trace(model.last_progressive_trace, args.num_groups)
    print(
        "[PASS] actual order is agent groups 0..3 then wrist groups 0..3, "
        "with every vision correction immediately followed by its LLM append",
        flush=True,
    )
    print(
        f"[PASS] grouped-full action parity on {len(samples)} real frames; "
        f"overall max error={maximum:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
