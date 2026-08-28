"""Spec-generic Qwen2.5-VL 32B eval over the real-photo benches (CV-Bench, MMVP, WildVision, ...).

Reuses `refcoco_gqa_batched_eval.py`'s validated machinery (build_first_token_context with its
no_grad + canvas-reconstruction invariants, batched_generate_fallback with its mm_token_type_ids
handling) but selects datasets via `qwen_vl_prefill.datasets_eval.get_spec` -- the same registry
every other model's numbers went through -- instead of that driver's refcoco/gqa-specific loading
and scoring. One driver, five arms:

    --arm ceiling    lossless transmission, stock forward           (sequential config)
    --arm floor      level-2 degraded, approx-only                  (approx_only config)
    --arm k0.25      interleaved g=4 + vision keep 0.25             (interleaved config)
    --arm k0.50      interleaved g=4 + vision keep 0.50
    --arm streaming  exact chunked prefill, k=1.0                   (llm_schedule=streaming)

WildVision has no reference answers (its spec's score() raises, deliberately): pass --dump-only to
skip scoring and write {idx, pred} jsonl for the later pairwise judge. Resume and OOM-skip
semantics are inherited from the shared helpers' conventions: --log-jsonl appends, same-label rows
fold in on restart, label mismatch aborts.

Run (appcorr env):
    python analysis/experiments/qwen25vl_bench_eval.py --dataset cvbench --arm ceiling \\
        --batch-size 8 --log-jsonl out/cvbench_ceiling.jsonl
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

from offload.common import ExperimentConfig
from offload.policies import get_transmission
from offload.server.model.qwen25vl_executor import Qwen25VLExecutor
from analysis.experiments.refcoco_gqa_batched_eval import (
    build_first_token_context, batched_generate_fallback,
)
from qwen_vl_prefill.datasets_eval import get_spec

ARM_CONFIGS = {
    "ceiling":   ("offload/config/realworldqa_qwen25vl_32b_sequential.json", {}),
    "floor":     ("offload/config/realworldqa_qwen25vl_32b_approx_only.json", {}),
    "k0.25":     ("offload/config/realworldqa_qwen25vl_32b_interleaved_g4.json",
                  {"token_keep_ratio": 0.25}),
    "k0.50":     ("offload/config/realworldqa_qwen25vl_32b_interleaved_g4.json",
                  {"token_keep_ratio": 0.50}),
    "streaming": ("offload/config/realworldqa_qwen25vl_32b_interleaved_g4.json",
                  {"llm_schedule": "streaming"}),
}
BASELINE_ARMS = {"ceiling", "floor"}  # floor is approx-only but NOT stock -- see below


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--arm", choices=sorted(ARM_CONFIGS), required=True)
    p.add_argument("--num-samples", type=int, default=None,
                   help="Strided subset size; omit for the full split.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--label", default=None)
    p.add_argument("--log-jsonl", default=None)
    p.add_argument("--dump-only", action="store_true",
                   help="Skip scoring (WildVision: no reference answers); write {idx, pred} only.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path, extra_appcorr = ARM_CONFIGS[args.arm]
    label = args.label or f"{args.dataset}_{args.arm}"

    with open(REPO_ROOT / cfg_path) as f:
        raw_config = json.load(f)
    raw_config["batch_size"] = 1
    raw_config["device"] = args.device
    if args.arm in ("k0.25", "k0.50", "streaming"):
        raw_config["transmission_kwargs"]["grouping_strategy"] = "sequential"
        raw_config["transmission_kwargs"]["num_groups"] = 4
    for k, v in extra_appcorr.items():
        raw_config.setdefault("appcorr_kwargs", {})[k] = v

    is_baseline = raw_config["transmission_policy_name"] in {"FullImageCompression", "Raw"}

    from datasets import load_dataset
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    spec = get_spec(args.dataset)
    ds = spec.load(load_dataset)
    n_total = len(ds)
    if args.num_samples:
        n = min(args.num_samples, n_total)
        indices = list(range(0, n_total, max(n_total // n, 1)))[:n]
    else:
        indices = list(range(n_total))
    print(f"[bench] === {label} === arm={args.arm} n={len(indices)}/{n_total} "
          f"baseline={is_baseline} dump_only={args.dump_only}", flush=True)

    executor = Qwen25VLExecutor(torch.device(args.device))
    executor.load_model(raw_config["model_name"], ExperimentConfig(**raw_config))
    proc = executor.processor
    ip = proc.image_processor
    min_px, max_px = ip.size["shortest_edge"], ip.size["longest_edge"]
    factor = ip.patch_size * ip.merge_size * 4
    encoder = None if is_baseline else get_transmission(raw_config["transmission_policy_name"])

    correct = 0
    processed = 0
    score_sum = 0.0
    already = {}
    if args.log_jsonl and Path(args.log_jsonl).exists():
        with open(args.log_jsonl) as f:
            for line in f:
                r = json.loads(line)
                if r.get("label") != label:
                    raise SystemExit(f"label mismatch in {args.log_jsonl}: {r.get('label')!r} != {label!r}")
                already[r["idx"]] = r
        for r in already.values():
            if not args.dump_only:
                correct += int(r["correct"]); score_sum += float(r.get("score", 0.0))
            processed += 1
        if already:
            print(f"[bench] RESUME: {len(already)} rows folded in", flush=True)
    log_f = open(args.log_jsonl, "a", encoding="utf-8") if args.log_jsonl else None

    t0 = time.time()
    print_every = max(len(indices) // 100, 1)
    oom_skipped = []
    batch_items, batch_meta = [], []

    def flush():
        nonlocal correct, processed, score_sum
        if not batch_items:
            return
        texts = batched_generate_fallback(executor.model, proc.tokenizer, batch_items,
                                          args.max_new_tokens, args.device, executor.image_token_id)
        for (idx, gold), pred in zip(batch_meta, texts):
            row = {"idx": idx, "label": label, "pred": pred}
            if not args.dump_only:
                ok, sc = spec.score(pred, gold)
                correct += int(ok); score_sum += float(sc)
                row.update(correct=bool(ok), score=float(sc))
            processed += 1
            if log_f:
                log_f.write(json.dumps(row) + "\n"); log_f.flush()
            if processed % print_every == 0 or processed == len(indices):
                extra = (f" acc={100*correct/max(processed,1):.2f}%" if not args.dump_only else "")
                print(f"    [{processed}/{len(indices)}] idx={idx} pred={pred[:40]!r}{extra} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        batch_items.clear(); batch_meta.clear()

    for idx in indices:
        if idx in already:
            continue
        img, prompt, gold = spec.prepare(ds[idx], smart_resize, factor, min_px, max_px)
        image_np = np.array(img, dtype=np.uint8)
        cfg = ExperimentConfig(**{**raw_config,
                                  "image_shape": [image_np.shape[0], image_np.shape[1], 3]})
        try:
            first_token, context = build_first_token_context(
                executor, encoder, raw_config, cfg, image_np, prompt, is_baseline)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not isinstance(e, torch.cuda.OutOfMemoryError) and "mha_graph" not in str(e):
                raise
            print(f"    [SKIP-OOM] idx={idx}", flush=True)
            oom_skipped.append(idx)
            gc.collect(); torch.cuda.empty_cache()
            continue
        batch_items.append({"input_ids": context["input_ids"], "first_token": first_token,
                            "pixel_values": context["pixel_values"],
                            "image_grid_thw": context["image_grid_thw"]})
        batch_meta.append((idx, gold))
        del context
        torch.cuda.empty_cache()
        if len(batch_items) >= args.batch_size:
            flush()
    flush()
    if log_f:
        log_f.close()

    print(f"\n[bench] === Summary: {label} ===")
    print(f"    samples: {processed}")
    if not args.dump_only:
        print(f"    accuracy: {100*correct/max(processed,1):.2f}%  ({correct}/{processed})  "
              f"mean_score: {score_sum/max(processed,1):.4f}")
    print(f"    wall: {time.time()-t0:.0f}s")
    if oom_skipped:
        print(f"    SKIPPED (OOM): {len(oom_skipped)} -- {oom_skipped}")


if __name__ == "__main__":
    main()
