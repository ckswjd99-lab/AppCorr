"""
dinov3_classifier_offload_eval.py

Drives the REAL offload pipeline (SchedulerModule + WorkerModule via multiprocessing queues --
everything from offload/server/main.py minus the TCP transport) against ImageNet-1k validation
images, using the DINOv3-7B classifier executor. Compares three conditions, selected by --config:

  1. "approx-only"  -- offload/config/imnet_approx_only_l2.json (Laplacian, single heavily
     downsampled level, BatchCountBasedPolicy -> FULL_INFERENCE on the degraded input only).
  2. "full baseline" -- offload/config/imnet_sequential.json (FullImageCompression, lossless PNG
     of the complete image, BatchCountBasedPolicy -> FULL_INFERENCE == stock model call).
  3. "interleaved correction" -- offload/config/imnet_interleaved_g4.json (ProgressiveLaplacian +
     GroupTriggerPolicy, real approx/correct pipeline). --grouping-strategy overrides
     transmission_kwargs['grouping_strategy'] (grid/uniform_diff/energy_asc/energy_desc/...) for
     the grouping sweep without needing a separate config file per strategy.

Reports top-1/top-5 accuracy (reusing offload.mobile.dataset.ImageNetLoader's bookkeeping) and
per-op GPU latency (CUDA-event based, from server_events -- see worker.py's execute_pipeline).

Run (from repo root, appropriate conda env):
    python analysis/experiments/dinov3_classifier_offload_eval.py \
        --config offload/config/imnet_interleaved_g4.json --grouping-strategy energy_desc \
        --num-samples 10
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
                         help="Override transmission_kwargs['grouping_strategy'] (e.g. grid, "
                              "uniform_diff, energy_asc, energy_desc). Only meaningful for "
                              "ProgressiveLaplacian configs.")
    parser.add_argument("--num-groups", type=int, default=None,
                         help="Override transmission_kwargs['num_groups'].")
    parser.add_argument("--token-keep-ratio", type=float, default=None,
                         help="Override appcorr_kwargs['token_keep_ratio'].")
    parser.add_argument("--sdpa-query-bucket-size", type=int, default=None,
                         help="Override appcorr_kwargs['sdpa_query_bucket_size'] -- pads the "
                              "correction query set to a bucket multiple, avoiding cuBLAS/SDPA's "
                              "per-shape dispatch cost under variable group sizes (uniform_diff/"
                              "energy_asc/energy_desc). 0/unset = disabled (existing behavior).")
    parser.add_argument("--data-root", type=str, default="/NHNHOME/share/cjpark/data/imagenet_val")
    parser.add_argument("--num-samples", type=int, default=10)
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
    if args.token_keep_ratio is not None:
        raw.setdefault("appcorr_kwargs", {})["token_keep_ratio"] = args.token_keep_ratio
    if args.sdpa_query_bucket_size is not None:
        raw.setdefault("appcorr_kwargs", {})["sdpa_query_bucket_size"] = args.sdpa_query_bucket_size
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
    print(f"[driver] appcorr_kwargs={raw_config.get('appcorr_kwargs', {})}")

    dataset_loader = get_dataset_loader(
        "imagenet-1k", args.data_root, batch_size=1,
        image_size=config.image_shape[0], num_workers=args.num_workers,
    )
    full_loader = dataset_loader.get_loader()
    # ImageFolder is not shuffled, and this dataset is 1000 classes x 50 images each, sorted by
    # class -- a plain `for images, labels in loader: break after N` would only ever see the
    # first N images of class 0000, giving a trivial/meaningless accuracy signal (and no real
    # test of grouping behavior on varied content). Take a deterministic STRIDED subset spanning
    # the whole validation set instead, so a small --num-samples still touches many classes, and
    # -- just as important -- the SAME exact images are used across every grouping-strategy run
    # for a fair A/B (stride/subset is a pure function of num_samples and dataset size, no RNG).
    from torch.utils.data import Subset

    full_dataset = full_loader.dataset
    n_total = len(full_dataset)
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

    try:
        control_q.put(("CONFIG", config))
        time.sleep(1.0)  # let CONFIG reach the worker before the first patches

        processed = 0
        for images, labels in loader:
            if processed >= args.num_samples:
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

            top1 = bool(pred_top5 and pred_top5[0] == label_val)
            top5 = label_val in pred_top5
            per_sample.append({"label": label_val, "pred_top5": pred_top5, "top1": top1, "top5": top5, "wall": wall})
            print(f"    sample {processed + 1}/{args.num_samples}: label={label_val} pred={pred_top5} "
                  f"top1={top1} top5={top5} wall={wall:.2f}s")
            sys.stdout.flush()
            processed += 1
    finally:
        control_q.put(("STOP", None))
        result_q.cancel_join_thread()
        for proc in (scheduler, worker):
            proc.join(timeout=15)
            if proc.is_alive():
                proc.terminate()

    summary = dataset_loader.get_summary()
    total_wall = time.time() - t_start

    print(f"\n[driver] === Summary: {label} ===")
    print(f"    samples: {summary.get('total_samples', 0)}")
    print(f"    top1_acc: {summary.get('top1_acc', 0.0):.2f}%   top5_acc: {summary.get('top5_acc', 0.0):.2f}%")
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
