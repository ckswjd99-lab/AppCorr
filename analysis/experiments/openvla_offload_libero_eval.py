"""
openvla_offload_libero_eval.py

Runs the OpenVLA progressive prefill through the REAL offload pipeline (SchedulerModule +
WorkerModule via multiprocessing queues -- everything from offload/server/main.py minus the TCP
transport), stepping actual LIBERO episodes. The driver plays the mobile side: per control step it
center-crops the camera frame, encodes it with VLAPatchCanvasPolicy (base layer + residual-ranked
patch groups), pushes the patches, and consumes the InferenceResult -- which carries the full
server_events CUDA-event timeline, giving per-op (APPROX/CORRECT/HEAD/...) timing through the
existing monitoring machinery.

Schedules (scheduler_kwargs['schedule'] on VLAInterleavedStatic): interleaved | sequential | full.

Run (from repo root, `openvla` conda env, LIBERO software-EGL env vars set):
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 MUJOCO_EGL_ALLOW_ANY_DEVICE=1 USE_TF=0 USE_TORCH=1 \
    python analysis/experiments/openvla_offload_libero_eval.py --task-ids all --schedules interleaved,sequential,full
"""

import argparse
import math
import multiprocessing
import queue as queue_mod
import sys
import time
from collections import defaultdict
from pathlib import Path

APPCORR_ROOT = Path(__file__).resolve().parents[2]
OPENVLA_ROOT = Path("/NHNHOME/share/cjpark/openvla")
for p in (APPCORR_ROOT, OPENVLA_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--task-suite", type=str, default="libero_spatial")
    parser.add_argument("--task-ids", type=str, default="0")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--coverage", type=float, default=1.0)
    parser.add_argument("--base-factor", type=int, default=4)
    parser.add_argument("--schedules", type=str, default="interleaved,sequential,full")
    parser.add_argument("--result-timeout", type=float, default=180.0)
    return parser.parse_args()


def center_crop_resize(img_np: np.ndarray, crop_scale: float = 0.9) -> np.ndarray:
    image = Image.fromarray(img_np).convert("RGB")
    w, h = image.size
    nh, nw = h * math.sqrt(crop_scale), w * math.sqrt(crop_scale)
    image = image.crop(((w - nw) / 2, (h - nh) / 2, (w + nw) / 2, (h + nh) / 2)).resize((224, 224), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def make_config(args, schedule: str):
    from offload.common.protocol import ExperimentConfig

    return ExperimentConfig(
        exp_id=f"vla_{schedule}",
        model_name="openvla",
        device="cuda:0",
        dataset_name=args.task_suite,
        dataset_kwargs={"checkpoint": args.checkpoint, "unnorm_key": args.task_suite},
        batch_size=1,
        image_shape=(224, 224, 3),
        patch_size=(14, 14),
        scheduler_policy_name="VLAInterleavedStatic",
        transmission_policy_name="VLAPatchCanvas",
        scheduler_kwargs={"schedule": schedule, "total_layers": 32},
        transmission_kwargs={
            "num_groups": args.num_groups,
            "coverage": args.coverage,
            "base_factor": args.base_factor,
            "text": "",
        },
    )


def request_action(encoder, config, frame_np, text, sched_q, result_q, timeout):
    """Mobile side of one control step: encode + push all groups, wait for the InferenceResult."""
    config.transmission_kwargs["text"] = text
    now = time.time()
    for group_patches in encoder.encode(frame_np, config):
        for p in group_patches:
            p.arrival_time = now
        for p in group_patches:
            sched_q.put(p)
    result = result_q.get(timeout=timeout)
    action = np.array(result.output[0], dtype=np.float64)
    return action, result.server_events


def run_episode(task_suite_name, task_id, args, encoder, config, sched_q, result_q, op_times):
    from libero.libero import benchmark

    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
    from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action

    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    env.reset()
    obs = env.set_init_state(initial_states[0])

    t, done = 0, False
    while t < args.max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue
        frame = center_crop_resize(get_libero_image(obs, 224))
        action, server_events = request_action(
            encoder, config, frame, task_description, sched_q, result_q, args.result_timeout
        )
        for ev in server_events:
            op_times[ev["type"]].append(ev["end"] - ev["start"])
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        t += 1
    env.close()
    return done, t


def main():
    args = parse_args()
    schedules = args.schedules.split(",")

    from offload.policies import get_transmission
    from offload.server.scheduler import SchedulerModule
    from offload.server.worker import WorkerModule

    sched_q = multiprocessing.Queue()
    worker_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    control_q = multiprocessing.Queue()
    feedback_q = multiprocessing.Queue()
    scheduler = SchedulerModule(sched_q, worker_q, control_q, feedback_q)
    worker = WorkerModule(worker_q, result_q, feedback_q)
    scheduler.start()
    worker.start()

    if args.task_ids.strip().lower() == "all":
        from libero.libero import benchmark

        task_ids = list(range(benchmark.get_benchmark_dict()[args.task_suite]().n_tasks))
    else:
        task_ids = [int(x) for x in args.task_ids.split(",")]
    print(f"[driver] Task ids: {task_ids}, schedules: {schedules}")

    encoder = get_transmission("VLAPatchCanvas")
    results = {}
    timing = {}
    try:
        for schedule in schedules:
            config = make_config(args, schedule)
            control_q.put(("CONFIG", config))
            time.sleep(1.0)  # let CONFIG reach the worker ahead of the first patches

            op_times = defaultdict(list)
            outcomes = []
            print(f"\n[driver] === Schedule: {schedule} ===")
            for task_id in task_ids:
                for trial in range(args.num_trials):
                    t0 = time.time()
                    try:
                        success, steps = run_episode(
                            args.task_suite, task_id, args, encoder, config, sched_q, result_q, op_times
                        )
                    except queue_mod.Empty:
                        print(f"    task {task_id}: TIMEOUT waiting for InferenceResult -- aborting schedule")
                        raise
                    outcomes.append((task_id, success))
                    print(f"    task {task_id} trial {trial + 1}: success={success} steps={steps} wall={time.time() - t0:.1f}s")
                    sys.stdout.flush()
            results[schedule] = outcomes
            timing[schedule] = {k: (float(np.mean(v)), len(v)) for k, v in op_times.items()}
    finally:
        control_q.put(("STOP", None))
        result_q.cancel_join_thread()
        for proc in (scheduler, worker):
            proc.join(timeout=15)
            if proc.is_alive():
                proc.terminate()

    print("\n[driver] === Summary: success rate by schedule ===")
    for schedule, outcomes in results.items():
        succ = [s for _, s in outcomes]
        print(f"    {schedule:12s}  {sum(succ)}/{len(succ)}  ({sum(succ) / len(succ):.0%})")

    print("\n[driver] === Mean per-op GPU time (ms) from server_events ===")
    for schedule, ops in timing.items():
        parts = "  ".join(f"{k}={v[0] * 1000:.1f}({v[1]})" for k, v in sorted(ops.items()))
        print(f"    {schedule:12s}  {parts}")

    print("\n[driver] === Per-task breakdown ===")
    for schedule, outcomes in results.items():
        line = "  ".join(f"t{tid}={'1' if s else '0'}" for tid, s in outcomes)
        print(f"    {schedule:12s}  {line}")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
