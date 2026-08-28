"""Measured (not analytic) FLOPs for Qwen2.5-VL 32B: ceiling + interleaved g=4 keep-rate arms.

Supersedes `flops_report_qwen25vl.py`'s ceiling-only measurement (its `_note` in
inprocess_flops.json: "no interleaved driver") -- the interleaved driver now exists
(`Qwen25VLExecutor` + vision keep rate, commits 28be3cf..5cda320), so the arms are measured by
driving the REAL mechanism in-process: the same encode -> decode -> preprocess -> prepare_tokens
-> approx_forward / correct_forward path the accuracy numbers came from, under `flops.session`.

Critical/overlappable split follows the counter's one rule (an op is critical iff it runs after
the final arrival): every transmission group opens `fl.arrival(gid)`, so round 0's full approx
pass and rounds 1..g-1 overlap transmission, and only the last group's correction is critical.
Ceiling never opens an arrival -> 100% critical, matching the table's denominator convention.

Two stage exclusions, both per counter.py's documented policy:
  * PSCORE -- `incoming_attention` re-runs the fused qkv `nn.Linear` (attention.py:120), the same
    double-count Gemma 3 had (its `_incoming_attention` re-ran q/k projections, 5.8%). Wrapped
    here by patching the method with a `fl.stage("PSCORE")` scope -- driver-local, zero cost to
    the production path when accounting is off.
  * PREPARE_TOKENS -- `correct_forward` re-runs `prepare_full_tokens` (patch_embed + rope) every
    round because the canvas changes; the offload worker already labels this op PREPARE_TOKENS and
    the counter already excludes it (ADE20K precedent). Same label reused so by_stage shows it.

Output: per dataset, `full` (ceiling critical GFLOPs/instruction), `k0.25`/`k0.50` (arm critical),
`total_k0.25`/`total_k0.50` (arm total) -- the inprocess_flops.json key convention.

    python analysis/experiments/flops_report_qwen25vl_arms.py \
        [--datasets refcoco gqa realworldqa] [--samples 6] [--keeps 0.25 0.50]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from appcorr import flops
from qwen_vl_prefill.datasets_eval import get_spec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_CONFIG = os.path.join(REPO_ROOT, "offload/config/realworldqa_qwen25vl_32b_interleaved_g4.json")
OUT_JSON = os.path.join(REPO_ROOT, "analysis/results/flops/inprocess_flops.json")

GFLOP = 1e9


@torch.no_grad()
def run_arm(executor, encoder, raw_config, image_np, prompt, keep, fl, req_id, llm_schedule=None):
    """One request through the real mechanism. Returns nothing -- fl accumulates."""
    from offload.common import ExperimentConfig, Task

    cfg = dict(raw_config)
    cfg["image_shape"] = [int(image_np.shape[0]), int(image_np.shape[1]), 3]
    cfg["transmission_kwargs"]["grouping_strategy"] = "sequential"
    cfg["transmission_kwargs"]["num_groups"] = 4
    cfg.setdefault("appcorr_kwargs", {})["token_keep_ratio"] = keep
    if llm_schedule is not None:
        cfg["appcorr_kwargs"]["llm_schedule"] = llm_schedule
    config = ExperimentConfig(**cfg)

    context = {}
    patch_buffer = []
    canvas = None
    with fl.request(req_id):
        for group_patches in encoder.encode(image_np[None], config):
            gid = group_patches[0].group_id
            for p in group_patches:
                p.text_payload = prompt
            patch_buffer.extend(group_patches)
            canvas = encoder.decode(patch_buffer, config, canvas=canvas)
            task = Task(task_id=0, request_id=0, payload=group_patches, instructions=[])
            with fl.arrival(gid):
                # preprocess/prepare_tokens run inside the arrival too: their cost (patch_embed,
                # prompt embed) lands on the arrival that triggered them, and prepare_full_tokens
                # is excluded by stage regardless (see module docstring).
                executor.preprocess(canvas, task, context, config)
                executor.prepare_tokens(task, context, config)
                if gid == 0:
                    with fl.stage("approx"):
                        executor.approx_forward({"layers": (0, executor.num_llm_layers)}, context, config)
                else:
                    with fl.stage("correct"):
                        executor.correct_forward(
                            {"layers": (0, executor.num_llm_layers), "group_id": gid}, context, config)
    del context


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-32B-Instruct")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--keeps", type=float, nargs="+", default=[0.25, 0.50])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", nargs="+", default=["refcoco", "gqa", "realworldqa"])
    ap.add_argument("--llm-schedule", choices=["interleaved", "streaming"], default=None,
                    help="appcorr_kwargs.llm_schedule for the arm runs; use 'streaming' with "
                         "--keeps 1.0 to measure the table's Streaming(k=1.0) column "
                         "(json keys k1.00/total_k1.00).")
    ap.add_argument("--write-json", action="store_true",
                    help="Merge results into analysis/results/flops/inprocess_flops.json")
    a = ap.parse_args()

    from datasets import load_dataset
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from offload.common import ExperimentConfig
    from offload.policies import get_transmission
    from offload.server.model.qwen25vl_executor import Qwen25VLExecutor
    from appcorr.models.qwen25vl.vision.attention import ApproxCorrectQwen25VLVisionAttention

    with open(BASE_CONFIG) as f:
        raw_config = json.load(f)
    raw_config["batch_size"] = 1
    raw_config["device"] = a.device

    executor = Qwen25VLExecutor(torch.device(a.device))
    print(f"[qwen-arms] loading {a.model_path}", flush=True)
    executor.load_model("qwen25vl_realworldqa", ExperimentConfig(**raw_config))
    proc = executor.processor
    ip = proc.image_processor
    min_px, max_px = ip.size["shortest_edge"], ip.size["longest_edge"]
    factor = ip.patch_size * ip.merge_size * 4
    encoder = get_transmission(raw_config["transmission_policy_name"])

    results = {}
    for ds_name in a.datasets:
        spec = get_spec(ds_name)
        ds = spec.load(load_dataset)
        idxs = list(range(0, len(ds), max(1, len(ds) // a.samples)))[:a.samples]
        row = {}

        # ---- ceiling: stock forward, no arrivals -> 100% critical ------------------------------ #
        with flops.session(executor.model.model.visual, executor.model.model.language_model,
                           enabled=True) as fl:
            for i in idxs:
                img, prompt, _ = spec.prepare(ds[i], smart_resize, factor, min_px, max_px)
                msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                     {"type": "text", "text": prompt}]}]
                text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                enc = proc(text=[text], images=[img], return_tensors="pt").to(a.device)
                if "mm_token_type_ids" not in enc:
                    # Older processor versions do not emit it; without it compute_3d_position_ids
                    # silently degrades to 1D positions (docs/memo/qwen25vl_baseline_mrope_bug.md).
                    enc["mm_token_type_ids"] = (enc["input_ids"] == executor.image_token_id).long()
                with fl.request(i):
                    with fl.stage("full"):
                        executor.model(**enc, use_cache=False)
        n = len(fl.requests)
        row["full"] = round(sum(r.critical for r in fl.requests) / n / GFLOP, 1)
        print(f"[{ds_name}] full (ceiling): {row['full']} GFLOPs/instr (n={n})", flush=True)

        # ---- arms: real mechanism, PSCORE excluded via a driver-local method wrap -------------- #
        for keep in a.keeps:
            with flops.session(executor.model.model.visual, executor.model.model.language_model,
                               enabled=True) as fl:
                orig_ia = ApproxCorrectQwen25VLVisionAttention.incoming_attention
                orig_pft = executor.vision_tower.prepare_full_tokens

                def ia_scoped(self_, *args, _fl=fl, _orig=orig_ia, **kw):
                    with _fl.stage("PSCORE"):
                        return _orig(self_, *args, **kw)

                def pft_scoped(*args, _fl=fl, _orig=orig_pft, **kw):
                    with _fl.stage("PREPARE_TOKENS"):
                        return _orig(*args, **kw)

                ApproxCorrectQwen25VLVisionAttention.incoming_attention = ia_scoped
                executor.vision_tower.prepare_full_tokens = pft_scoped
                try:
                    for i in idxs:
                        img, prompt, _ = spec.prepare(ds[i], smart_resize, factor, min_px, max_px)
                        image_np = np.array(img, dtype=np.uint8)
                        try:
                            run_arm(executor, encoder, raw_config, image_np, prompt, keep, fl, i,
                                    llm_schedule=a.llm_schedule)
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            if not isinstance(e, torch.cuda.OutOfMemoryError) and "mha_graph" not in str(e):
                                raise
                            print(f"    [skip-oom] {ds_name} idx={i} keep={keep}", flush=True)
                            torch.cuda.empty_cache()
                finally:
                    ApproxCorrectQwen25VLVisionAttention.incoming_attention = orig_ia
                    executor.vision_tower.prepare_full_tokens = orig_pft
                torch.cuda.empty_cache()

            done = [r for r in fl.requests if r.buckets]
            n = len(done)
            crit = sum(r.critical for r in done) / n / GFLOP
            tot = sum(r.total for r in done) / n / GFLOP
            key = f"k{keep:.2f}"  # "k0.25"/"k0.50" -- json convention; bare f"{keep}" would drop the trailing zero
            row[key] = round(crit, 1)
            row[f"total_{key}"] = round(tot, 1)
            by_stage = {}
            for r in done:
                for s, v in r.by_stage().items():
                    by_stage[s] = by_stage.get(s, 0) + v
            stg = {s: round(v / n / GFLOP, 1) for s, v in by_stage.items()}
            print(f"[{ds_name}] keep={keep}: critical={crit:.1f} total={tot:.1f} GFLOPs/instr "
                  f"(n={n}) by_stage={stg}", flush=True)

        results[ds_name] = row

    print("\n== summary ==")
    print(json.dumps(results, indent=2))

    if a.write_json:
        with open(OUT_JSON) as f:
            db = json.load(f)
        entry = db.setdefault("qwen25vl_32b", {})
        entry.pop("_note", None)  # ceiling-only caveat no longer true
        entry["_samples"] = a.samples
        for ds_name, row in results.items():
            entry.setdefault(ds_name, {}).update(row)
        with open(OUT_JSON, "w") as f:
            json.dump(db, f, indent=2)
        print(f"[qwen-arms] merged into {OUT_JSON}")


if __name__ == "__main__":
    main()
