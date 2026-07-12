"""
refcoco_gqa_batched_eval.py

Batched re-implementation of the RefCOCO/GQA full-dataset eval, built to cut the multi-day
full-dataset runtime the offload pipeline's batch_size=1 convention implies. Bypasses the
SchedulerModule/WorkerModule multiprocess pipeline entirely (no queues, no network-offload
simulation) and calls Qwen25VLExecutor's methods directly, in-process -- this is safe to do here
because we only need the model's answers for a plain accuracy sweep, not the transmission-latency
simulation the full offload pipeline is for.

What gets batched and what doesn't, and why:
  - The AppCorr vision-tower fork (`appcorr/models/qwen25vl/vision/`) has NO batch dimension at
    all (`hidden_states` is `[seq_length, dim]`, one image's patches, by design -- see that
    module's docstring). The LLM fork's `.correct()`/`.approx()` do carry a batch axis but every
    call site in this investigation has only ever driven them with B=1. Adding real batch support
    to the fork itself is a substantial, invasive change -- out of scope given the ask was to
    batch "the easy parts" efficiently, not rewrite the fork.
  - What IS easy and safe to batch: (a) baseline, which uses zero fork code (pure stock forward +
    stock generate()) -- batched end-to-end; (b) the `.generate()` FALLBACK call every condition
    (baseline and keep_rate alike) makes for any answer longer than one token -- this is 100%
    stock HF `generate()`, batches trivially with left-padding, and is often the single largest
    per-sample cost (up to 63 autoregressive steps vs one correction forward pass).
  - So: keep_rate conditions still run their correction (approx_forward/correct_forward) and
    first-token decode ONE IMAGE AT A TIME, unchanged from the existing offload-pipeline drivers
    (no risk to the validated correction logic) -- but instead of immediately falling back to a
    per-image generate() call, first-tokens are accumulated across a batch of `--batch-size`
    images, then ONE batched generate() call produces all their continuations together.
  - Baseline batches the correction-free first-forward pass too (a real stock `model(...)` call
    over the whole batch), since there's no per-image fork state to preserve there.

Validated before trusting at scale: run with --batch-size 1 first and diff against the existing
offload-pipeline driver's per-sample answers on the same strided indices (should match closely,
mod ordinary bf16 batch-size-invariance noise); then increase --batch-size and confirm accuracy is
stable (not a regression) before launching a full-dataset run.

Run (appcorr env):
    python analysis/experiments/refcoco_gqa_batched_eval.py --dataset refcoco \\
        --config offload/config/realworldqa_qwen25vl_32b_sequential.json \\
        --device cuda:0 --batch-size 8 --num-samples 40 --label smoketest
    python analysis/experiments/refcoco_gqa_batched_eval.py --dataset gqa \\
        --config offload/config/realworldqa_qwen25vl_32b_interleaved_g4.json \\
        --grouping-strategy top_energy --num-groups 1 --keep-rate 0.4 \\
        --device cuda:0 --batch-size 8 --full
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offload.common import Task, ExperimentConfig
from offload.common.protocol import Patch
from offload.policies import get_transmission
from offload.server.model.qwen25vl_executor import Qwen25VLExecutor

from analysis.experiments.refcoco_offload_eval import (
    GROUNDING_PROMPT_TMPL, score_answer as refcoco_score_answer,
)
from analysis.experiments.gqa_offload_eval import (
    GQA_PROMPT_SUFFIX, score_answer as gqa_score_answer,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["refcoco", "gqa"], required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--grouping-strategy", type=str, default=None)
    p.add_argument("--num-groups", type=int, default=None)
    p.add_argument("--keep-rate", type=float, default=None)
    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument("--full", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=63)
    return p.parse_args()


def load_base_config_dict(args):
    with open(args.config, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["batch_size"] = 1  # per-image correction loop is still batch=1; only generate() batches
    raw["device"] = args.device
    if args.grouping_strategy is not None:
        raw.setdefault("transmission_kwargs", {})["grouping_strategy"] = args.grouping_strategy
    if args.num_groups is not None:
        raw.setdefault("transmission_kwargs", {})["num_groups"] = args.num_groups
    if args.keep_rate is not None:
        raw.setdefault("transmission_kwargs", {})["keep_rate"] = args.keep_rate
    return raw


def build_first_token_context(executor, encoder, raw_config, config, image_np, prompt, is_baseline):
    """Runs preprocess+prepare_tokens+[approx/correct]_forward for ONE image directly (no
    scheduler/worker), then decodes the first token. Mirrors head_inference's/full_inference's
    first-token logic exactly, just stopping before the generate() fallback so it can be batched
    by the caller. Returns (first_token[1] tensor, context dict).

    CRITICAL (bug found by the user's suspicion, then a controlled approx-only smoke test): the
    keep_rate path must feed `preprocess` the CANVAS RECONSTRUCTED FROM THE TRANSMITTED PATCHES
    (blurred base + whatever correction groups have arrived), exactly as the offload pipeline's
    WorkerModule does via `policy.decode(patch_buffer, config, canvas=prev)` (worker.py:177-188).
    An earlier version of this script passed the raw full-resolution `image_np` to `preprocess`
    for every group instead -- which made `pixel_values` (and therefore the vision tower's inputs
    AND the batched generate() fallback) full-resolution regardless of keep_rate, silently turning
    every keep_rate/approx-only condition into ~baseline + fork numerical noise. All keep_rate
    results produced by that version are INVALID (baseline results are unaffected: baseline is
    genuinely full-resolution stock computation). Likewise, the per-image `config` must carry the
    image's own `image_shape` (as `refcoco_offload_eval.py:187` does) so encode/decode grids match
    the actual image rather than the config-default 896x896."""
    context = {}

    if is_baseline:
        total_layers = config.transmission_kwargs.get("total_layers", executor.num_llm_layers)
        task = Task(task_id=0, request_id=0, payload=[
            Patch(image_idx=0, spatial_idx=0, data=b"", text_payload=prompt)
        ], instructions=[])
        executor.preprocess(image_np[None], task, context, config)
        executor.prepare_tokens(task, context, config)
        with torch.no_grad():
            outputs = executor.model(
                input_ids=context["input_ids"], attention_mask=context["attention_mask"],
                pixel_values=context["pixel_values"], image_grid_thw=context["image_grid_thw"],
                use_cache=False,
            )
            first_token = outputs.logits[:, -1, :].argmax(dim=-1)
    else:
        image_config = dict(raw_config)
        image_config["image_shape"] = [int(image_np.shape[0]), int(image_np.shape[1]), 3]
        config = ExperimentConfig(**image_config)
        total_layers = config.transmission_kwargs.get("total_layers", executor.num_llm_layers)
        patch_buffer = []
        canvas = None
        for group_patches in encoder.encode(image_np[None], config):
            now = time.time()
            for p in group_patches:
                p.arrival_time = now
                p.text_payload = prompt
            group_id = group_patches[0].group_id
            # Reconstruct the current-fidelity canvas from ALL patches received so far --
            # identical accumulate-then-decode semantics to WorkerModule (worker.py:177-188).
            patch_buffer.extend(group_patches)
            canvas = encoder.decode(patch_buffer, config, canvas=canvas)
            task = Task(task_id=0, request_id=0, payload=group_patches, instructions=[])
            executor.preprocess(canvas, task, context, config)
            executor.prepare_tokens(task, context, config)
            if group_id == 0:
                executor.approx_forward({"layers": (0, total_layers)}, context, config)
            else:
                executor.correct_forward({"layers": (0, total_layers), "group_id": group_id}, context, config)
        with torch.no_grad():
            x_full = context["llm_current_feature"]
            hidden = executor.model.model.language_model.norm(x_full)
            logits_last = executor.model.lm_head(hidden[:, -1, :].to(executor.model.lm_head.weight.dtype))
            first_token = logits_last.argmax(dim=-1)

    return first_token, context


def batched_generate_fallback(model, tokenizer, items, max_new_tokens, device):
    """items: list of dicts with input_ids[1,T], attention_mask[1,T], first_token[1],
    pixel_values[P,C], image_grid_thw[1,3]. One batched model.generate() call, left-padded.
    Returns list of decoded continuation strings, one per item, in the same order."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    eos_id = tokenizer.eos_token_id
    seqs, orig_lens = [], []
    for it in items:
        if it["first_token"].item() == eos_id:
            seqs.append(it["input_ids"][0])  # no fallback needed; will be trimmed to "" below
            orig_lens.append(-1)  # sentinel: eos already, skip decoding
            continue
        ext = torch.cat([it["input_ids"][0], it["first_token"]], dim=0)
        seqs.append(ext)
        orig_lens.append(it["input_ids"].shape[1])

    max_len = max(s.shape[0] for s in seqs)
    B = len(seqs)
    padded_ids = torch.full((B, max_len), pad_id, dtype=seqs[0].dtype, device=device)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        padded_ids[i, max_len - L:] = s
        attn_mask[i, max_len - L:] = 1

    pixel_values = torch.cat([it["pixel_values"] for it in items], dim=0)
    image_grid_thw = torch.cat([it["image_grid_thw"] for it in items], dim=0)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=padded_ids, attention_mask=attn_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id,
        )

    results = []
    for i, orig_len in enumerate(orig_lens):
        if orig_len == -1:
            results.append("")
            continue
        # max_len-1 (not max_len) since every item's padded prompt ends with first_token at
        # position max_len-1 (left-padding right-aligns all items to the same max_len) -- the
        # answer text is first_token + whatever generate() appended after it, so trimming from
        # max_len would silently drop the first token (confirmed via a real smoke-test bug: a
        # generated "483" came back as "83").
        trimmed = gen_ids[i, max_len - 1:]
        results.append(tokenizer.decode(trimmed, skip_special_tokens=True))
    return results


def main():
    args = parse_args()
    raw_config = load_base_config_dict(args)
    label = args.label or f"{args.dataset}_batched"
    model_path = raw_config["dataset_kwargs"]["model_path"]
    is_baseline = raw_config["transmission_policy_name"] != "ProgressiveLaplacian"

    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset

    print(f"[batched] === Run: {label} === dataset={args.dataset} baseline={is_baseline} batch_size={args.batch_size}")
    print(f"[batched] config={args.config} transmission_kwargs={raw_config.get('transmission_kwargs', {})}")

    config = ExperimentConfig(**raw_config)
    executor = Qwen25VLExecutor(torch.device(args.device))
    executor.load_model(config.model_name, config)
    processor = executor.processor
    ip = processor.image_processor
    min_pixels, max_pixels = ip.size["shortest_edge"], ip.size["longest_edge"]
    factor = ip.patch_size * ip.merge_size * 4
    encoder = get_transmission(raw_config["transmission_policy_name"]) if not is_baseline else None

    if args.dataset == "refcoco":
        ds = load_dataset("lmms-lab/RefCOCO", split="val")
        n_total = len(ds)
    else:
        instr = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
        imgs = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
        image_by_id = {ex["id"]: ex["image"] for ex in imgs}
        n_total = len(instr)

    if args.full:
        indices = list(range(n_total))
    else:
        n_samples = min(args.num_samples, n_total)
        stride = max(n_total // n_samples, 1)
        indices = list(range(0, n_total, stride))[:n_samples]
    print(f"[batched] {len(indices)} examples (of {n_total})")

    def load_example(idx):
        if args.dataset == "refcoco":
            ex = ds[idx]
            image = ex["image"].convert("RGB")
            expr = ex["answer"][0] if isinstance(ex["answer"], list) and ex["answer"] else str(ex["answer"])
            bx, by, bw, bh = ex["bbox"]
            gt_box_orig = (bx, by, bx + bw, by + bh)
            prompt = GROUNDING_PROMPT_TMPL.format(expr=expr)
            gt = gt_box_orig
        else:
            ex = instr[idx]
            image = image_by_id[ex["imageId"]].convert("RGB")
            prompt = ex["question"] + GQA_PROMPT_SUFFIX
            gt = ex["answer"]
        orig_w, orig_h = image.width, image.height
        target_h, target_w = smart_resize(orig_h, orig_w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
        resized = image.resize((target_w, target_h), Image.BILINEAR)
        image_np = np.array(resized, dtype=np.uint8)
        sx, sy = target_w / orig_w, target_h / orig_h
        if args.dataset == "refcoco":
            gt = (gt[0] * sx, gt[1] * sy, gt[2] * sx, gt[3] * sy)
        return image_np, prompt, gt, (target_h, target_w)

    correct = 0
    processed = 0
    iou_sum = 0.0
    t_start = time.time()
    print_every = 1 if len(indices) <= 40 else max(len(indices) // 200, 10)

    batch_items = []
    batch_meta = []  # (idx, gt, grid_hw)

    def flush_batch():
        nonlocal correct, processed, iou_sum
        if not batch_items:
            return
        texts = batched_generate_fallback(executor.model, processor.tokenizer, batch_items,
                                           args.max_new_tokens, args.device)
        for (idx, gt, grid_hw), pred_text in zip(batch_meta, texts):
            if args.dataset == "refcoco":
                ok, iou = refcoco_score_answer(pred_text, gt)
                iou_sum += iou
            else:
                ok = gqa_score_answer(pred_text, gt)
                iou = 0.0
            correct += int(ok)
            processed += 1
            if processed % print_every == 0 or processed == len(indices):
                acc = 100.0 * correct / processed
                extra = f" mean_iou={iou_sum/processed:.3f}" if args.dataset == "refcoco" else ""
                print(f"    [{processed}/{len(indices)}] idx={idx} grid={grid_hw[0]}x{grid_hw[1]} "
                      f"pred={pred_text[:50]!r} correct={ok} running_acc={acc:.2f}%{extra} "
                      f"elapsed={time.time()-t_start:.0f}s")
                sys.stdout.flush()
        batch_items.clear()
        batch_meta.clear()

    for idx in indices:
        image_np, prompt, gt, grid_hw = load_example(idx)
        first_token, context = build_first_token_context(executor, encoder, raw_config, config, image_np, prompt, is_baseline)
        batch_items.append({
            "input_ids": context["input_ids"], "first_token": first_token,
            "pixel_values": context["pixel_values"], "image_grid_thw": context["image_grid_thw"],
        })
        batch_meta.append((idx, gt, grid_hw))
        if len(batch_items) >= args.batch_size:
            flush_batch()
    flush_batch()

    total_wall = time.time() - t_start
    acc = 100.0 * correct / max(processed, 1)
    print(f"\n[batched] === Summary: {label} ===")
    print(f"    samples: {processed}")
    if args.dataset == "refcoco":
        print(f"    accuracy@0.5: {acc:.2f}%  ({correct}/{processed})  mean_iou: {iou_sum/max(processed,1):.4f}")
    else:
        print(f"    accuracy: {acc:.2f}%  ({correct}/{processed})")
    print(f"    total wall time: {total_wall:.1f}s ({total_wall/max(processed,1):.2f}s/sample avg)")


if __name__ == "__main__":
    main()
