"""
realworldqa_offload_eval.py

Drives the REAL offload pipeline (SchedulerModule + WorkerModule, local multiprocess, no TCP --
same convention as every prior driver this session) against the RealWorldQA benchmark, using the
Qwen2.5-VL AppCorr executor (OpenCLIPExecutor's sibling for this task: OpenCLIPExecutor -> here
Qwen25VLExecutor). Compares approx-only / full baseline / interleaved correction, selected by
--config.

**Native resolution**: unlike every prior driver (which resized to one fixed shape), each image
here is resized ONLY via Qwen's own `smart_resize` (aspect-ratio-preserving, patch(28)-aligned,
clamped to the processor's own configured pixel-count window -- this IS "native resolution" for
this model family, see Phase 1/plan notes). Since each image gets a DIFFERENT target shape, a
FRESH `ExperimentConfig` (with that image's own `image_shape`) is built and CONFIG is resent via
control_q before every image -- `WorkerModule.config` is a mutable global updated on each CONFIG
message, while `context` (session state) is keyed per-request (`request_id`), so this is safe:
each image's patches are always decoded/executed against ITS OWN just-sent config, and by the time
the next image's CONFIG is sent, the previous image's full pipeline (through SEND_RESPONSE) has
already completed (the driver blocks on `result_q.get()` before moving on).

Run (appcorr env):
    python analysis/experiments/realworldqa_offload_eval.py \\
        --config offload/config/realworldqa_qwen25vl_32b_interleaved_g4.json --num-samples 20
"""

import argparse
import json
import multiprocessing
import queue as queue_mod
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def extract_letter(text: str):
    m = re.search(r"\b([ABCD])\b", text.strip())
    return m.group(1) if m else None


def normalize_freeform(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def is_mcq(question: str) -> bool:
    return bool(re.search(r"\n\s*A\.\s", question))


def score_answer(question: str, pred_text: str, gt_answer: str) -> bool:
    if is_mcq(question):
        return extract_letter(pred_text) == gt_answer.strip().upper()
    pred_norm = normalize_freeform(pred_text)
    gt_norm = normalize_freeform(gt_answer)
    return pred_norm == gt_norm or (gt_norm != "" and gt_norm in pred_norm.split())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--grouping-strategy", type=str, default=None)
    p.add_argument("--num-groups", type=int, default=None)
    p.add_argument("--keep-rate", type=float, default=None)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--full", action="store_true", help="Run all 765 RealWorldQA examples (ignores --num-samples).")
    p.add_argument("--result-timeout", type=float, default=600.0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--label", type=str, default=None)
    return p.parse_args()


def load_base_config_dict(args):
    with open(args.config, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["batch_size"] = 1
    raw["device"] = args.device
    if args.grouping_strategy is not None:
        raw.setdefault("transmission_kwargs", {})["grouping_strategy"] = args.grouping_strategy
    if args.num_groups is not None:
        raw.setdefault("transmission_kwargs", {})["num_groups"] = args.num_groups
    if args.keep_rate is not None:
        raw.setdefault("transmission_kwargs", {})["keep_rate"] = args.keep_rate
    return raw


def main():
    args = parse_args()
    raw_config = load_base_config_dict(args)
    label = args.label or Path(args.config).stem
    model_path = raw_config["dataset_kwargs"]["model_path"]

    from offload.common import ExperimentConfig
    from offload.policies import get_transmission
    from offload.server.scheduler import SchedulerModule
    from offload.server.worker import WorkerModule
    from transformers import AutoProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset
    from PIL import Image

    print(f"[driver] === Run: {label} ===")
    print(f"[driver] config={args.config}  transmission_kwargs={raw_config.get('transmission_kwargs', {})}")

    print(f"[driver] loading processor for {model_path} (cheap -- tokenizer/image_processor config only)...")
    processor = AutoProcessor.from_pretrained(model_path)
    ip = processor.image_processor
    min_pixels = ip.size["shortest_edge"]
    max_pixels = ip.size["longest_edge"]
    # The Laplacian pyramid needs EVERY level to be independently patch-aligned, i.e. H/W must be
    # divisible by patch_size * pyramid_scale, not just patch_size -- smart_resize's own default
    # factor (patch_size * merge_size = 28) only guarantees the FULL-res image is 28-aligned, which
    # is not strict enough once a level divides that by 4 (pyramid_levels up to 2 => scale 4 => need
    # multiples of 112). Use factor=112 for ALL conditions (not just the pyramid ones) so every
    # condition sees the exact same per-image resolution -- a fair apples-to-apples comparison.
    base_factor = ip.patch_size * ip.merge_size
    factor = base_factor * 4
    print(f"[driver] smart_resize params: factor={factor} (base={base_factor}, aligned for pyramid scale 4) "
          f"min_pixels={min_pixels} max_pixels={max_pixels}")

    print("[driver] loading lmms-lab/RealWorldQA ...")
    ds = load_dataset("lmms-lab/RealWorldQA", split="test")
    n_total = len(ds)
    if args.full:
        indices = list(range(n_total))
        print(f"[driver] running FULL dataset: {n_total} examples")
    else:
        n_samples = min(args.num_samples, n_total)
        stride = max(n_total // n_samples, 1)
        indices = list(range(0, n_total, stride))[:n_samples]
        print(f"[driver] sampling {len(indices)} examples strided across {n_total} (stride={stride})")

    sched_q = multiprocessing.Queue()
    worker_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    control_q = multiprocessing.Queue()
    feedback_q = multiprocessing.Queue()
    scheduler = SchedulerModule(sched_q, worker_q, control_q, feedback_q)
    worker = WorkerModule(worker_q, result_q, feedback_q)
    scheduler.start()
    worker.start()

    encoder = get_transmission(raw_config["transmission_policy_name"])
    op_times = defaultdict(list)
    per_sample = []
    t_start = time.time()
    correct = 0
    processed = 0
    print_every = 1 if len(indices) <= 20 else max(len(indices) // 100, 5)

    try:
        for idx in indices:
            ex = ds[idx]
            image = ex["image"].convert("RGB")
            question = ex["question"]
            gt_answer = ex["answer"]

            target_h, target_w = smart_resize(image.height, image.width, factor=factor,
                                               min_pixels=min_pixels, max_pixels=max_pixels)
            resized = image.resize((target_w, target_h), Image.BILINEAR)
            image_np = np.array(resized, dtype=np.uint8)  # [H,W,3]

            image_config = dict(raw_config)
            image_config["image_shape"] = [target_h, target_w, 3]
            config = ExperimentConfig(**image_config)

            control_q.put(("CONFIG", config))
            time.sleep(2.0)  # let CONFIG propagate before this image's patches arrive

            t0 = time.time()
            for group_patches in encoder.encode(image_np[None], config):
                now = time.time()
                for p in group_patches:
                    p.arrival_time = now
                    p.text_payload = question
                for p in group_patches:
                    sched_q.put(p)
            try:
                result = result_q.get(timeout=args.result_timeout)
            except queue_mod.Empty:
                print(f"    sample {processed}: TIMEOUT waiting for InferenceResult -- aborting")
                raise
            wall = time.time() - t0

            pred_text = result.output[0] if result.output else ""
            for ev in result.server_events:
                op_times[ev["type"]].append(ev["end"] - ev["start"])

            ok = score_answer(question, pred_text, gt_answer)
            correct += int(ok)
            processed += 1
            per_sample.append({"idx": idx, "gt": gt_answer, "pred": pred_text, "correct": ok, "wall": wall,
                                "grid_hw": (target_h, target_w)})
            if processed % print_every == 0 or processed == len(indices):
                acc = 100.0 * correct / processed
                print(f"    [{processed}/{len(indices)}] idx={idx} grid={target_h}x{target_w} "
                      f"gt={gt_answer!r} pred={pred_text!r} correct={ok} running_acc={acc:.2f}% "
                      f"wall={wall:.2f}s elapsed={time.time()-t_start:.0f}s")
                sys.stdout.flush()
    finally:
        control_q.put(("STOP", None))
        result_q.cancel_join_thread()
        for proc in (scheduler, worker):
            proc.join(timeout=15)
            if proc.is_alive():
                proc.terminate()

    total_wall = time.time() - t_start
    acc = 100.0 * correct / max(processed, 1)

    print(f"\n[driver] === Summary: {label} ===")
    print(f"    samples: {processed}")
    print(f"    accuracy: {acc:.2f}%  ({correct}/{processed})")
    print(f"    total wall time: {total_wall:.1f}s ({total_wall / max(processed, 1):.2f}s/sample avg)")

    print(f"\n[driver] === Mean per-op GPU time (ms), {label} ===")
    for op_name in sorted(op_times):
        vals = op_times[op_name]
        print(f"    {op_name:20s} mean={np.mean(vals) * 1000:8.3f}ms  n={len(vals)}")

    return {
        "label": label,
        "config": args.config,
        "accuracy": acc,
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
