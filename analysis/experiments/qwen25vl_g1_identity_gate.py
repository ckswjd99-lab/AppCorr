"""
qwen25vl_g1_identity_gate.py

The gate that would have caught the mm_token_type_ids bug (docs/memo/qwen25vl_baseline_mrope_bug.md)
and the rule-3 increment-persist bug before either shipped a number. Per
docs/memo/interleaved_correction_contract.md's rule 2 ("g=1 identity"): driving the interleaved
path with a single group at 100% coverage must reproduce the sequential baseline EXACTLY, because
it is architecturally the same computation.

Drives the REAL offload pipeline (SchedulerModule + WorkerModule, same as
`realworldqa_offload_eval.py`) for both the sequential config and the interleaved_g4 config forced
to `num_groups=1`, on the same strided sample indices, and asserts PER-SAMPLE prediction equality
-- never the aggregate. 70%==70% with two cancelling flips is exactly how the mm_token_type_ids bug
hid for a full investigation; an aggregate-only gate would have called the broken version healthy.

`Qwen25VLExecutor.full_inference`'s own independence assertion (HF-internally-derived
position_ids == the fork's `context["position_ids"]`) fires unconditionally on every sequential
sample already -- this script does not duplicate that, it adds the cross-arm check the assertion
alone cannot make (the assertion only proves the baseline's OWN position derivation is internally
consistent; it says nothing about whether the interleaved arm agrees with it).

Run (appcorr env):
    python analysis/experiments/qwen25vl_g1_identity_gate.py \\
        --model-path Qwen/Qwen2.5-VL-32B-Instruct --num-samples 10

Exit code 0 = every sample matched. Exit code 1 = at least one sample diverged (a real bug, not
noise -- see the contract memo's gate #2: "expect equality, not close enough").
"""

import argparse
import json
import multiprocessing
import queue as queue_mod
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEQUENTIAL_CONFIG = REPO_ROOT / "offload/config/realworldqa_qwen25vl_32b_sequential.json"
INTERLEAVED_CONFIG = REPO_ROOT / "offload/config/realworldqa_qwen25vl_32b_interleaved_g4.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def run_arm(config_path, num_groups_override, indices, model_path, device):
    """Drives one arm through the real SchedulerModule/WorkerModule pipeline, returns
    {idx: pred_text} for the given indices, in order."""
    with open(config_path) as f:
        raw_config = json.load(f)
    raw_config["batch_size"] = 1
    raw_config["device"] = device
    if num_groups_override is not None:
        raw_config.setdefault("transmission_kwargs", {})["num_groups"] = num_groups_override

    from offload.common import ExperimentConfig
    from offload.policies import get_transmission
    from offload.server.scheduler import SchedulerModule
    from offload.server.worker import WorkerModule
    from transformers import AutoProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset
    from PIL import Image

    processor = AutoProcessor.from_pretrained(model_path)
    ip = processor.image_processor
    min_pixels, max_pixels = ip.size["shortest_edge"], ip.size["longest_edge"]
    factor = ip.patch_size * ip.merge_size * 4

    ds = load_dataset("lmms-lab/RealWorldQA", split="test")

    sched_q, worker_q, result_q = multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Queue()
    control_q, feedback_q = multiprocessing.Queue(), multiprocessing.Queue()
    scheduler = SchedulerModule(sched_q, worker_q, control_q, feedback_q)
    worker = WorkerModule(worker_q, result_q, feedback_q)
    scheduler.start()
    worker.start()

    encoder = get_transmission(raw_config["transmission_policy_name"])
    preds = {}

    try:
        for idx in indices:
            ex = ds[idx]
            image = ex["image"].convert("RGB")
            question = ex["question"]

            target_h, target_w = smart_resize(image.height, image.width, factor=factor,
                                               min_pixels=min_pixels, max_pixels=max_pixels)
            resized = image.resize((target_w, target_h), Image.BILINEAR)
            image_np = np.array(resized, dtype=np.uint8)

            image_config = dict(raw_config)
            image_config["image_shape"] = [target_h, target_w, 3]
            config = ExperimentConfig(**image_config)

            result = None
            for attempt in range(3):
                control_q.put(("CONFIG", config))
                time.sleep(2.0)
                for group_patches in encoder.encode(image_np[None], config):
                    now = time.time()
                    for p in group_patches:
                        p.arrival_time = now
                        p.text_payload = question
                    for p in group_patches:
                        sched_q.put(p)
                try:
                    result = result_q.get(timeout=150.0)
                    break
                except queue_mod.Empty:
                    print(f"    idx={idx}: TIMEOUT (attempt {attempt+1}/3)", flush=True)
            if result is None:
                raise RuntimeError(f"idx={idx} failed after 3 attempts")
            preds[idx] = result.output[0] if result.output else ""
            print(f"    [{config_path.name}] idx={idx} pred={preds[idx]!r}", flush=True)
    finally:
        control_q.put(("STOP", None))
        result_q.cancel_join_thread()
        for proc in (scheduler, worker):
            proc.join(timeout=15)
            if proc.is_alive():
                proc.terminate()

    return preds


def main():
    args = parse_args()

    from datasets import load_dataset
    n_total = len(load_dataset("lmms-lab/RealWorldQA", split="test"))
    n_samples = min(args.num_samples, n_total)
    stride = max(n_total // n_samples, 1)
    indices = list(range(0, n_total, stride))[:n_samples]
    print(f"[gate] indices: {indices}", flush=True)

    print(f"\n[gate] === sequential (baseline) ===", flush=True)
    seq_preds = run_arm(SEQUENTIAL_CONFIG, None, indices, args.model_path, args.device)

    print(f"\n[gate] === interleaved_g4 forced to num_groups=1 ===", flush=True)
    g1_preds = run_arm(INTERLEAVED_CONFIG, 1, indices, args.model_path, args.device)

    print(f"\n[gate] === per-sample comparison ===")
    mismatches = []
    for idx in indices:
        seq_p, g1_p = seq_preds[idx], g1_preds[idx]
        match = seq_p == g1_p
        print(f"    idx={idx}  seq={seq_p!r}  g1={g1_p!r}  {'OK' if match else 'MISMATCH'}")
        if not match:
            mismatches.append(idx)

    n = len(indices)
    print(f"\n[gate] {n - len(mismatches)}/{n} samples match exactly.")
    if mismatches:
        print(f"[gate] FAIL: {len(mismatches)} sample(s) diverged: {mismatches}")
        print(f"[gate] Per the interleaved correction contract (docs/memo/interleaved_correction_contract.md), "
              f"g=1 must reproduce baseline bit-for-bit. A residual difference means something is still wrong "
              f"-- do not attribute it to float reassociation without proof.")
        sys.exit(1)
    else:
        print(f"[gate] PASS: g=1 identity holds, every sample, exact text match.")
        sys.exit(0)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
