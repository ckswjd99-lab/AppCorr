"""
clip_zeroshot_offload_eval.py

Drives the REAL offload pipeline (SchedulerModule + WorkerModule via multiprocessing queues --
everything from offload/server/main.py minus the TCP transport) against ImageNet-1k validation
images, using the CLIP-ViT-bigG/14 zero-shot classifier executor (OpenCLIPExecutor). Mirrors
`dinov3_classifier_offload_eval.py`'s structure/conventions exactly (same deterministic strided
sampling, same CUDA-event per-op latency, same nr=10 sanity -> scale-up discipline). Compares
three conditions, selected by --config:

  1. "approx-only"  -- offload/config/imagenet_clip_bigg_approx_only_l2.json (Laplacian, single
     heavily downsampled level, BatchCountBasedPolicy -> FULL_INFERENCE on the degraded input only).
  2. "full baseline" -- offload/config/imagenet_clip_bigg_sequential.json (FullImageCompression,
     lossless PNG, BatchCountBasedPolicy -> FULL_INFERENCE == stock model call).
  3. "interleaved correction" -- offload/config/imagenet_clip_bigg_interleaved_g4.json
     (ProgressiveLaplacian + GroupTriggerPolicy, real approx/correct pipeline).
     --grouping-strategy/--num-groups override transmission_kwargs for sweeping.

Reports top-1/top-5 accuracy (reusing offload.mobile.dataset.ImageNetLoader's bookkeeping) and
per-op GPU latency (CUDA-event based, from server_events -- see worker.py's execute_pipeline).

Run (from repo root, appcorr conda env):
    python analysis/experiments/clip_zeroshot_offload_eval.py \\
        --config offload/config/imagenet_clip_bigg_interleaved_g4.json --num-samples 10
"""

import argparse
import json
import multiprocessing
import queue as queue_mod
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to offload config JSON.")
    parser.add_argument("--grouping-strategy", type=str, default=None,
                         help="Override transmission_kwargs['grouping_strategy']. Only meaningful "
                              "for ProgressiveLaplacian configs.")
    parser.add_argument("--num-groups", type=int, default=None,
                         help="Override transmission_kwargs['num_groups'].")
    parser.add_argument("--token-keep-thres", type=float, default=None,
                         help="Override appcorr_kwargs['token_keep_thres']. Also forces "
                              "appcorr_kwargs['mobile_pscore']='residual_energy' if not already set.")
    parser.add_argument("--data-root", type=str, default="/NHNHOME/share/cjpark/data/imagenet_val")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--full", action="store_true", help="Run the FULL dataset (ignores --num-samples).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--result-timeout", type=float, default=300.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--label", type=str, default=None, help="Tag for this run in the printed summary.")
    return parser.parse_args()


def load_config(args):
    from offload.common import ExperimentConfig

    with open(args.config, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["batch_size"] = 1  # one image at a time: clean per-image latency + simple result indexing
    raw["device"] = args.device
    if args.grouping_strategy is not None:
        raw.setdefault("transmission_kwargs", {})["grouping_strategy"] = args.grouping_strategy
    if args.num_groups is not None:
        raw.setdefault("transmission_kwargs", {})["num_groups"] = args.num_groups
    if args.token_keep_thres is not None:
        appcorr = raw.setdefault("appcorr_kwargs", {})
        appcorr["token_keep_thres"] = args.token_keep_thres
        appcorr.setdefault("mobile_pscore", "residual_energy")
    return ExperimentConfig(**raw), raw


def main():
    args = parse_args()
    config, raw_config = load_config(args)
    label = args.label or Path(args.config).stem

    from offload.mobile.dataset import get_dataset_loader
    from offload.policies import get_transmission
    from offload.server.scheduler import SchedulerModule
    from offload.server.worker import WorkerModule

    print(f"[driver] === Run: {label} ===")
    print(f"[driver] config={args.config}  transmission_kwargs={raw_config.get('transmission_kwargs', {})}")

    dataset_loader = get_dataset_loader(
        "imagenet-1k", args.data_root, batch_size=1,
        image_size=config.image_shape[0], num_workers=args.num_workers,
    )
    full_loader = dataset_loader.get_loader()
    from torch.utils.data import Subset

    full_dataset = full_loader.dataset
    n_total = len(full_dataset)
    if args.full:
        indices = list(range(n_total))
        loader = torch.utils.data.DataLoader(
            full_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        )
        print(f"[driver] running FULL dataset: {n_total} images")
    else:
        n_samples = min(args.num_samples, n_total)
        stride = max(n_total // n_samples, 1)
        indices = list(range(0, n_total, stride))[:n_samples]
        subset = Subset(full_dataset, indices)
        loader = torch.utils.data.DataLoader(
            subset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        )
        print(f"[driver] sampling {len(indices)} images strided across {n_total} "
              f"(stride={stride}, spans ~{len(indices)} distinct classes)")

    sched_q = multiprocessing.Queue()
    worker_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    control_q = multiprocessing.Queue()
    feedback_q = multiprocessing.Queue()
    scheduler = SchedulerModule(sched_q, worker_q, control_q, feedback_q)
    worker = WorkerModule(worker_q, result_q, feedback_q)
    scheduler.start()
    worker.start()

    encoder = get_transmission(config.transmission_policy_name)
    op_times = defaultdict(list)
    per_sample = []
    t_start = time.time()

    total_target = len(indices)
    print_every = 1 if total_target <= 50 else max(total_target // 200, 50)
    kept_total = 0.0
    full_total = 0.0

    try:
        control_q.put(("CONFIG", config))
        time.sleep(1.0)  # let CONFIG reach the worker before the first patches

        processed = 0
        for images, labels in loader:
            if processed >= total_target:
                break
            image_np = images[0].permute(1, 2, 0).contiguous().numpy()  # [H,W,3] uint8
            label_val = int(labels[0].item())

            t0 = time.time()
            for group_patches in encoder.encode(image_np[None], config):
                now = time.time()
                for p in group_patches:
                    p.arrival_time = now
                for p in group_patches:
                    sched_q.put(p)
            try:
                result = result_q.get(timeout=args.result_timeout)
            except queue_mod.Empty:
                print(f"    sample {processed}: TIMEOUT waiting for InferenceResult -- aborting")
                raise
            wall = time.time() - t0

            pred_top5 = result.output[0] if result.output else []
            batch_metrics = dataset_loader.evaluate_batch([pred_top5], [label_val])
            for ev in result.server_events:
                op_times[ev["type"]].append(ev["end"] - ev["start"])
            kept_total += float(getattr(result, "token_prune_kept_patch", 0.0) or 0.0)
            full_total += float(getattr(result, "token_prune_full_patch", 0.0) or 0.0)

            top1 = bool(pred_top5 and pred_top5[0] == label_val)
            top5 = label_val in pred_top5
            per_sample.append({"label": label_val, "pred_top5": pred_top5, "top1": top1, "top5": top5, "wall": wall})
            processed += 1
            if processed % print_every == 0 or processed == total_target:
                summ = dataset_loader.get_summary()
                keep_pct = 100.0 * kept_total / full_total if full_total > 0 else 100.0
                print(f"    [{processed}/{total_target}] running top1={summ.get('top1_acc', 0.0):.2f}% "
                      f"top5={summ.get('top5_acc', 0.0):.2f}% keep_rate={keep_pct:.1f}% "
                      f"elapsed={time.time() - t_start:.0f}s")
                sys.stdout.flush()
    finally:
        control_q.put(("STOP", None))
        result_q.cancel_join_thread()
        for proc in (scheduler, worker):
            proc.join(timeout=15)
            if proc.is_alive():
                proc.terminate()

    summary = dataset_loader.get_summary()
    total_wall = time.time() - t_start
    keep_pct = 100.0 * kept_total / full_total if full_total > 0 else 100.0

    print(f"\n[driver] === Summary: {label} ===")
    print(f"    samples: {summary.get('total_samples', 0)}")
    print(f"    top1_acc: {summary.get('top1_acc', 0.0):.2f}%   top5_acc: {summary.get('top5_acc', 0.0):.2f}%")
    print(f"    patch keep_rate: {keep_pct:.2f}%  (kept={kept_total:.0f} / full={full_total:.0f})")
    print(f"    total wall time: {total_wall:.1f}s ({total_wall / max(processed, 1):.2f}s/sample avg)")

    print(f"\n[driver] === Mean per-op GPU time (ms), {label} ===")
    for op_name in sorted(op_times):
        vals = op_times[op_name]
        print(f"    {op_name:20s} mean={np.mean(vals) * 1000:8.3f}ms  n={len(vals)}")

    return {
        "label": label,
        "config": args.config,
        "grouping_strategy": raw_config.get("transmission_kwargs", {}).get("grouping_strategy"),
        "summary": summary,
        "keep_rate_pct": keep_pct,
        "op_times_ms": {k: float(np.mean(v) * 1000) for k, v in op_times.items()},
        "total_wall_sec": total_wall,
        "per_sample": per_sample,
    }


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
