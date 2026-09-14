"""Where does one OpenVLA control step go? Per-step wall breakdown of the offload driver
(`openvla_offload_libero_eval.py`, same processes, same config), asked after the LIBERO campaign
showed ~290-360 ms/step while the server-side op events (CUDA-event durations) sum to ~200 ms.

Per control step the driver clock is split into
  sim        env.step (MuJoCo physics + offscreen render)
  image      get_libero_image + center_crop_resize (JPEG round trip, LANCZOS, crop)
  encode     VLAPatchCanvas.encode + multiprocessing puts of every Patch
  wait       result_q.get -- everything between the last push and the action coming back
and the server events of that step (absolute timestamps, same host clock) into
  server_span   first SERVER_RECEIVE -> last op end
  server_ops    sum of op durations (LOAD_INPUT/PREPARE_TOKENS/CORRECT_FORWARD/HEAD_INFERENCE/...)
  server_gaps   span - ops (job dispatch, per-op event sync, queue hops inside the server)
  transit       wait - (last op end - t_last_push)   (result pipe back + driver-side pickup)
so wall = sim + image + encode + wait, and wait = (server work after the last push) + transit.

Run (openvla env, same env vars as the campaign driver; ONE short episode, discarded warmup first):
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 MUJOCO_EGL_ALLOW_ANY_DEVICE=1 \
  USE_TF=0 USE_TORCH=1 PYTHONPATH=.:/NHNHOME/share/cjpark/openvla \
  python analysis/experiments/openvla_step_profile.py --task-suite libero_spatial \
      --checkpoint openvla/openvla-7b-finetuned-libero-spatial --max-steps 60 \
      --num-groups 4 --grouping sequential --schedules chunked --vision-correction new_only
"""
import copy
import json
import multiprocessing
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import analysis.experiments.openvla_offload_libero_eval as base  # noqa: E402


def profiled_episode(task_suite_name, task_id, args, encoder, config, sched_q, result_q, rows):
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_image
    from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action

    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(task_id)
    env, task_description = base.make_libero_env(task, resolution=256)
    env.reset()
    obs = env.set_init_state(task_suite.get_task_init_states(task_id)[0])
    t, done = 0, False
    while t < args.max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue
        t0 = time.time()
        frame = base.center_crop_resize(get_libero_image(obs, 224))
        t1 = time.time()
        config.transmission_kwargs["text"] = task_description
        now = time.time()
        n_patches = 0
        for group_patches in encoder.encode(frame, config):
            for p in group_patches:
                p.arrival_time = now
                sched_q.put(p)
                n_patches += 1
        t2 = time.time()
        result = result_q.get(timeout=args.result_timeout)
        t3 = time.time()
        action = np.array(result.output[0], dtype=np.float64)
        evs = result.server_events
        ops = [e for e in evs if e["type"] not in ("SERVER_RECEIVE",)]
        span_start = min(e["start"] for e in evs)
        span_end = max(e["end"] for e in evs)
        by_type = defaultdict(float)
        for e in ops:
            by_type[e["type"]] += e["end"] - e["start"]
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        t4 = time.time()
        obs, _, done, _ = env.step(action.tolist())
        t5 = time.time()
        rows.append({
            "image": t1 - t0, "encode": t2 - t1, "wait": t3 - t2, "post": t4 - t3, "sim": t5 - t4,
            "server_span": span_end - span_start, "server_ops": sum(by_type.values()),
            "server_after_push": span_end - t2, "transit": t3 - span_end,
            "server_first_recv_lag": span_start - now, "n_patches": n_patches,
            **{f"op:{k}": v for k, v in by_type.items()},
        })
        if done:
            break
        t += 1
    env.close()
    return done, t


def main():
    args = base.parse_args()
    from offload.policies import get_transmission
    from offload.server.scheduler import SchedulerModule
    from offload.server.worker import WorkerModule

    sched_q, worker_q, result_q = multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Queue()
    control_q, feedback_q = multiprocessing.Queue(), multiprocessing.Queue()
    scheduler = SchedulerModule(sched_q, worker_q, control_q, feedback_q)
    worker = WorkerModule(worker_q, result_q, feedback_q)
    scheduler.start()
    worker.start()
    encoder = get_transmission("VLAPatchCanvas")
    schedule = args.schedules.split(",")[0]
    task_id = 0 if args.task_ids.strip().lower() == "all" else int(args.task_ids.split(",")[0])
    try:
        config = base.make_config(args, schedule)
        control_q.put(("CONFIG", config))
        time.sleep(1.0)
        warm = copy.copy(args)
        warm.max_steps = 15
        profiled_episode(args.task_suite, task_id, warm, encoder, config, sched_q, result_q, [])
        rows = []
        done, steps = profiled_episode(args.task_suite, task_id, args, encoder, config, sched_q, result_q, rows)
    finally:
        control_q.put(("STOP", None))
        result_q.cancel_join_thread()
        for proc in (scheduler, worker):
            proc.join(timeout=15)
            if proc.is_alive():
                proc.terminate()

    keys = sorted({k for r in rows for k in r})
    mean = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
    wall = mean["image"] + mean["encode"] + mean["wait"] + mean["post"] + mean["sim"]
    print(f"\n[profile] {args.task_suite} task {task_id} schedule={schedule} steps={len(rows)} "
          f"success={done}  mean wall/step = {wall * 1e3:.1f} ms")
    print("  driver clock:")
    for k in ("sim", "image", "encode", "wait", "post"):
        print(f"    {k:<10s} {mean[k] * 1e3:7.1f} ms  ({100 * mean[k] / wall:4.1f}%)")
    print("  inside wait:")
    for k in ("server_first_recv_lag", "server_after_push", "transit"):
        print(f"    {k:<22s} {mean[k] * 1e3:7.1f} ms")
    print("  server (this step's events):")
    print(f"    span {mean['server_span'] * 1e3:7.1f} ms = ops {mean['server_ops'] * 1e3:7.1f} ms "
          f"+ gaps {(mean['server_span'] - mean['server_ops']) * 1e3:7.1f} ms")
    for k in keys:
        if k.startswith("op:"):
            print(f"      {k[3:]:<18s} {mean[k] * 1e3:7.1f} ms")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mean": mean, "rows": rows, "args": vars(args)}, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
