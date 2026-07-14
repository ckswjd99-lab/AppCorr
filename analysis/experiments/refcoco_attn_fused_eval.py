"""
refcoco_attn_fused_eval.py

Real-accuracy pipeline test of "residual_energy x early-layer-attention" fused patch selection,
following up on the AUC-vs-GT-bbox diagnostic (qwen25vl_attn_pscore_diagnostic.py) that found
attention layer 7 (AUC=0.5763) clearly beats pure residual_energy (AUC=0.5376) at localizing
RefCOCO's referring-expression target region -- and following the already-validated DINOv3/OpenVLA
precedent in this repo (commits 26384fa/46d1016/7a4eb55/1f33a1f: mobile_pscore=residual_energy x
server_pscore=attention, multiply-fused, +5-6pp accuracy there).

Mechanism (no changes to offload/policies/transmission/progressive.py or the vision/LLM forks):
  1. Run `encoder.encode(..., keep_rate=1.0)` so `top_energy`'s grouping assigns EVERY merge-group
     to group_id=1 -- `_process_image_group_residuals` then compresses/creates a Patch for every
     single merge-group (each carrying `pscore_hint` = residual energy, already computed CPU-side,
     no fork changes needed).
  2. Process group 0 (base/blurred layer) exactly as every other driver this session does:
     preprocess -> prepare_tokens -> approx_forward. This gives us `context["vision_ctx"]`, the
     already-validated per-image RoPE/window-index bundle needed to compute attention scores.
  3. Compute the "patch_attn_prob"-style attention-received score for the chosen early layer(s)
     (default: layer 7 alone, the best single-layer AUC found) from this SAME approx pass -- no
     extra forward pass needed, reuses the hidden state already computed in step 2.
  4. fused_score = pscore_hint (residual) * attn_score (attention), computed per ALL-groups Patch
     from step 1. Rank descending, keep the top `--keep-rate` fraction (fixed-count, matching
     `top_energy`'s semantics exactly, for direct comparability against the existing full-dataset
     top_energy curve at the same keep_rate).
  5. Reconstruct the canvas from base+selected patches (same `encoder.decode(patch_buffer, ...,
     canvas=prev)` pattern validated in refcoco_gqa_batched_eval.py's bugfix), preprocess again,
     and run `correct_forward(group_id=1)` -- context["group_map"] only marks the SELECTED
     merge-groups as group_id=1, so only those get corrected, exactly like a smaller keep_rate
     would under plain top_energy, just with a smarter selection rule.

Run (appcorr env):
    python analysis/experiments/refcoco_attn_fused_eval.py --keep-rate 0.35 --attn-layers 7 \\
        --device cuda:0 --num-samples 64 --label smoke_attnfused_kr035
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offload.common import Task, ExperimentConfig
from offload.common.protocol import Patch
from offload.policies import get_transmission
from offload.server.model.qwen25vl_executor import Qwen25VLExecutor

from analysis.experiments.refcoco_offload_eval import GROUNDING_PROMPT_TMPL, score_answer
from analysis.experiments.refcoco_gqa_batched_eval import batched_generate_fallback
from analysis.experiments.qwen25vl_attn_pscore_diagnostic import _manual_attention_received
from analysis.experiments.qwen25vl_llm_attn_pscore_diagnostic import compute_llm_text_to_image_attention


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="offload/config/realworldqa_qwen25vl_32b_interleaved_g4.json")
    p.add_argument("--keep-rate", type=float, required=True, help="Fraction of merge-groups to correct, "
                    "selected by fused (residual x attention) score instead of residual alone.")
    p.add_argument("--attn-layers", type=int, nargs="+", default=[7], help="Full-attention layer index/indices "
                    "to use for the attention score (mean if multiple). Default: layer 7 alone (best single-layer "
                    "AUC found in qwen25vl_attn_pscore_diagnostic.py).")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--full", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=63)
    p.add_argument("--log-jsonl", type=str, default=None)
    p.add_argument("--weighted-model", type=str, default=None, help="Path to an .npz fitted by "
                    "the logistic-regression weight search (feature_order/scaler_mean/scaler_scale/"
                    "coef/intercept over [log1p_residual, attn_L...]). When set, overrides the plain "
                    "residual x mean-attention fusion with this learned linear combination, and "
                    "--attn-layers is ignored in favor of the layers baked into the model file.")
    return p.parse_args()


class WeightedFusionModel:
    """Loads a StandardScaler + logistic-regression fit over [log1p_residual, attn_L...,
    attn_L_hH..., or llmattn_L...] (see the weight-fitting work in
    qwen25vl_attn_pscore_diagnostic.py / qwen25vl_llm_attn_pscore_diagnostic.py) and applies it as
    a fused score. Supports vision layer-mean (attn_<L>), vision per-head (attn_<L>_h<H>), and LLM
    text->image layer-mean (llmattn_<L>) features in the same fit, parsed from feature_order."""

    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.feature_order = [str(f) for f in d["feature_order"]]
        self.scaler_mean = d["scaler_mean"].astype(np.float64)
        self.scaler_scale = d["scaler_scale"].astype(np.float64)
        self.coef = d["coef"].astype(np.float64)
        intercept = d["intercept"]
        self.intercept = float(intercept.reshape(-1)[0])
        self.attn_layers = set()       # vision layers needed as layer-mean (attn_<L>)
        self.per_head_layers = set()   # vision layers needed with per-head breakdown (attn_<L>_h<H>)
        self.llm_layers = set()        # LLM layers needed as layer-mean (llmattn_<L>)
        for name in self.feature_order:
            if name == "log1p_residual":
                continue
            if name.startswith("llmattn_"):
                self.llm_layers.add(int(name.split("_")[1]))
                continue
            parts = name.split("_")
            layer = int(parts[1])
            if len(parts) == 2:
                self.attn_layers.add(layer)
            else:
                self.per_head_layers.add(layer)

    def score(self, residual, attn_by_layer, attn_by_layer_head=None, llm_attn_by_layer=None):
        """residual: np.array [N]. attn_by_layer: dict layer_idx -> np.array [N] (vision
        layer-mean, native order, gathered per-patch). attn_by_layer_head: dict layer_idx ->
        np.array [num_heads, N] (vision per-head, native order, gathered per-patch).
        llm_attn_by_layer: dict layer_idx -> np.array [N] (LLM text->image layer-mean, native
        order, gathered per-patch). Returns fused score np.array [N] (higher = keep)."""
        feats = []
        for name in self.feature_order:
            if name == "log1p_residual":
                feats.append(np.log1p(residual))
                continue
            if name.startswith("llmattn_"):
                feats.append(llm_attn_by_layer[int(name.split("_")[1])])
                continue
            parts = name.split("_")
            layer = int(parts[1])
            if len(parts) == 2:
                feats.append(attn_by_layer[layer])
            else:
                head = int(parts[2][1:])  # "h12" -> 12
                feats.append(attn_by_layer_head[layer][head])
        X = np.stack(feats, axis=1)
        Xs = (X - self.scaler_mean) / self.scaler_scale
        return Xs @ self.coef + self.intercept


def compute_attn_scores_native_order(executor, vctx, layer_indices, device, reduce="mean"):
    """Attention-received score(s), returned indexed by NATIVE merge-group order (matching
    Patch.spatial_idx), via one forward pass through the shared prefix.
    reduce="mean": single array, mean over layer_indices (legacy behavior).
    reduce="none": dict {layer_idx: np.array} with each requested layer kept separate."""
    tower = executor.vision_tower
    x = vctx["hidden_states"]
    cache = {}
    unit = tower.spatial_merge_unit
    inv_window_index = vctx["inv_window_index"]
    layer_set = set(layer_indices)
    max_layer = max(layer_set)
    per_layer_native = {}
    for i in range(max_layer + 1):
        if i in layer_set:
            received = _manual_attention_received(tower.blocks[i], x, vctx, i, tower)
            n_groups = received.shape[0] // unit
            received_per_group_permuted = received.view(n_groups, unit).mean(dim=1)
            per_layer_native[i] = received_per_group_permuted[inv_window_index]
        x, cache = tower.approx_forward(x, i, i + 1, vctx, cache, tag_prefix="attnfused")
    if reduce == "none":
        return {k: v.cpu().numpy() for k, v in per_layer_native.items()}
    return torch.stack([per_layer_native[l] for l in layer_indices]).mean(dim=0).cpu().numpy()


def compute_all_attn_native_order(executor, vctx, layer_mean_layers, per_head_layers, device):
    """Single forward pass through the shared prefix, capturing BOTH layer-mean and per-head
    attention-received scores as needed per layer (avoids a redundant second pass when a
    WeightedFusionModel needs per-head features). Returns
    (layer_mean_dict {layer: np.array[N]}, per_head_dict {layer: np.array[num_heads, N]}),
    both in NATIVE merge-group order."""
    tower = executor.vision_tower
    x = vctx["hidden_states"]
    cache = {}
    unit = tower.spatial_merge_unit
    inv_window_index = vctx["inv_window_index"]
    all_layers = set(layer_mean_layers) | set(per_head_layers)
    layer_mean_out, per_head_out = {}, {}
    if not all_layers:
        return layer_mean_out, per_head_out
    max_layer = max(all_layers)
    for i in range(max_layer + 1):
        if i in all_layers:
            need_per_head = i in per_head_layers
            received = _manual_attention_received(tower.blocks[i], x, vctx, i, tower, collapse_heads=not need_per_head)
            if need_per_head:
                n_groups = received.shape[1] // unit
                permuted = received.view(received.shape[0], n_groups, unit).mean(dim=2)
                per_head_out[i] = permuted[:, inv_window_index].cpu().numpy()
                if i in layer_mean_layers:
                    layer_mean_out[i] = per_head_out[i].mean(axis=0)
            else:
                n_groups = received.shape[0] // unit
                permuted = received.view(n_groups, unit).mean(dim=1)
                layer_mean_out[i] = permuted[inv_window_index].cpu().numpy()
        x, cache = tower.approx_forward(x, i, i + 1, vctx, cache, tag_prefix="attnfused")
    return layer_mean_out, per_head_out


def build_attn_fused_context(executor, encoder, raw_config, image_np, prompt, keep_rate, attn_layers, device,
                              weighted_model=None):
    """Mirrors refcoco_gqa_batched_eval.py's build_first_token_context, but selects the corrected
    merge-groups by (residual x attention) fused score instead of pure top_energy."""
    image_config = dict(raw_config)
    image_config["image_shape"] = [int(image_np.shape[0]), int(image_np.shape[1]), 3]
    image_config.setdefault("transmission_kwargs", {})
    image_config["transmission_kwargs"] = dict(raw_config.get("transmission_kwargs", {}),
                                                 grouping_strategy="top_energy", num_groups=1,
                                                 keep_rate=1.0, mobile_pscore="residual_energy")
    config = ExperimentConfig(**image_config)
    total_layers = config.transmission_kwargs.get("total_layers", executor.num_llm_layers)

    context = {}
    patch_buffer = []
    canvas = None
    all_group1_patches = None
    for group_patches in encoder.encode(image_np[None], config):
        now = time.time()
        for p in group_patches:
            p.arrival_time = now
            p.text_payload = prompt
        group_id = group_patches[0].group_id
        if group_id == 0:
            patch_buffer.extend(group_patches)
            canvas = encoder.decode(patch_buffer, config, canvas=canvas)
            task = Task(task_id=0, request_id=0, payload=group_patches, instructions=[])
            executor.preprocess(canvas, task, context, config)
            executor.prepare_tokens(task, context, config)
            executor.approx_forward({"layers": (0, total_layers)}, context, config)
        else:
            all_group1_patches = group_patches  # ALL merge-groups (keep_rate=1.0), not yet applied

    # Attention score(s) from the JUST-COMPLETED approx pass (group 0's vision_ctx / llm_input_embeds).
    if weighted_model is not None:
        with torch.no_grad():
            layer_mean_scores, per_head_scores = compute_all_attn_native_order(
                executor, context["vision_ctx"], weighted_model.attn_layers,
                weighted_model.per_head_layers, device)
            llm_layer_scores = {}
            if weighted_model.llm_layers:
                llm_received = compute_llm_text_to_image_attention(
                    executor, context, weighted_model.llm_layers, per_head=False)
                llm_layer_scores = {l: v.cpu().numpy() for l, v in llm_received.items()}
        residual_arr = np.array([p.pscore_hint for p in all_group1_patches], dtype=np.float64)
        spatial_idxs = np.array([p.spatial_idx for p in all_group1_patches])
        attn_for_patches = {l: arr[spatial_idxs] for l, arr in layer_mean_scores.items()}
        attn_head_for_patches = {l: arr[:, spatial_idxs] for l, arr in per_head_scores.items()}
        llm_attn_for_patches = {l: arr[spatial_idxs] for l, arr in llm_layer_scores.items()}
        fused_scores = weighted_model.score(residual_arr, attn_for_patches, attn_head_for_patches, llm_attn_for_patches)
        fused = list(zip(all_group1_patches, fused_scores))
    else:
        with torch.no_grad():
            attn_scores = compute_attn_scores_native_order(executor, context["vision_ctx"], attn_layers, device)
        fused = [(p, p.pscore_hint * float(attn_scores[p.spatial_idx])) for p in all_group1_patches]
    fused.sort(key=lambda t: t[1], reverse=True)
    keep_n = int(round(keep_rate * len(fused)))
    selected = [p for p, _ in fused[:keep_n]]

    if selected:
        patch_buffer.extend(selected)
        canvas = encoder.decode(patch_buffer, config, canvas=canvas)
        task = Task(task_id=0, request_id=0, payload=selected, instructions=[])
        executor.preprocess(canvas, task, context, config)
        executor.prepare_tokens(task, context, config)
        executor.correct_forward({"layers": (0, total_layers), "group_id": 1}, context, config)

    with torch.no_grad():
        first_token = executor.decode_first_token(context["llm_current_feature"])
    return first_token, context


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    raw_config["batch_size"] = 1
    raw_config["device"] = args.device
    label = args.label or f"attnfused_kr{args.keep_rate}"

    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset
    from PIL import Image

    weighted_model = WeightedFusionModel(args.weighted_model) if args.weighted_model else None
    if weighted_model is not None:
        print(f"[attnfused] === Run: {label} === keep_rate={args.keep_rate} "
              f"weighted_model={args.weighted_model} (layers={weighted_model.attn_layers})")
    else:
        print(f"[attnfused] === Run: {label} === keep_rate={args.keep_rate} attn_layers={args.attn_layers}")
    config = ExperimentConfig(**raw_config)
    executor = Qwen25VLExecutor(torch.device(args.device))
    executor.load_model(config.model_name, config)
    processor = executor.processor
    ip = processor.image_processor
    min_pixels, max_pixels = ip.size["shortest_edge"], ip.size["longest_edge"]
    factor = ip.patch_size * ip.merge_size * 4
    encoder = get_transmission(raw_config["transmission_policy_name"])

    ds = load_dataset("lmms-lab/RefCOCO", split="val")
    n_total = len(ds)
    if args.full:
        indices = list(range(n_total))
    else:
        n_samples = min(args.num_samples, n_total)
        stride = max(n_total // n_samples, 1)
        indices = list(range(0, n_total, stride))[:n_samples]
    print(f"[attnfused] {len(indices)} examples (of {n_total})")

    def load_example(idx):
        ex = ds[idx]
        image = ex["image"].convert("RGB")
        expr = ex["answer"][0] if isinstance(ex["answer"], list) and ex["answer"] else str(ex["answer"])
        bx, by, bw, bh = ex["bbox"]
        gt_box_orig = (bx, by, bx + bw, by + bh)
        prompt = GROUNDING_PROMPT_TMPL.format(expr=expr)
        target_h, target_w = smart_resize(image.height, image.width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
        resized = image.resize((target_w, target_h), Image.BILINEAR)
        image_np = np.array(resized, dtype=np.uint8)
        sx, sy = target_w / image.width, target_h / image.height
        gt = (gt_box_orig[0] * sx, gt_box_orig[1] * sy, gt_box_orig[2] * sx, gt_box_orig[3] * sy)
        return image_np, prompt, gt, (target_h, target_w)

    correct = 0
    processed = 0
    iou_sum = 0.0
    t_start = time.time()
    print_every = 1 if len(indices) <= 40 else max(len(indices) // 200, 10)
    log_f = open(args.log_jsonl, "a", encoding="utf-8") if args.log_jsonl else None

    batch_items, batch_meta = [], []

    def flush_batch():
        nonlocal correct, processed, iou_sum
        if not batch_items:
            return
        texts = batched_generate_fallback(executor.model, processor.tokenizer, batch_items, args.max_new_tokens, args.device)
        for (idx, gt, grid_hw), pred_text in zip(batch_meta, texts):
            ok, iou = score_answer(pred_text, gt)
            iou_sum += iou
            correct += int(ok)
            processed += 1
            if log_f is not None:
                log_f.write(json.dumps({"idx": idx, "label": label, "pred": pred_text, "correct": bool(ok), "iou": iou}) + "\n")
            if processed % print_every == 0 or processed == len(indices):
                acc = 100.0 * correct / processed
                print(f"    [{processed}/{len(indices)}] idx={idx} grid={grid_hw[0]}x{grid_hw[1]} "
                      f"pred={pred_text[:50]!r} correct={ok} running_acc={acc:.2f}% mean_iou={iou_sum/processed:.3f} "
                      f"elapsed={time.time()-t_start:.0f}s")
                sys.stdout.flush()
        batch_items.clear()
        batch_meta.clear()

    for idx in indices:
        image_np, prompt, gt, grid_hw = load_example(idx)
        first_token, context = build_attn_fused_context(executor, encoder, raw_config, image_np, prompt,
                                                          args.keep_rate, args.attn_layers, args.device,
                                                          weighted_model=weighted_model)
        batch_items.append({"input_ids": context["input_ids"], "first_token": first_token,
                             "pixel_values": context["pixel_values"], "image_grid_thw": context["image_grid_thw"]})
        batch_meta.append((idx, gt, grid_hw))
        if len(batch_items) >= args.batch_size:
            flush_batch()
    flush_batch()
    if log_f is not None:
        log_f.close()

    total_wall = time.time() - t_start
    acc = 100.0 * correct / max(processed, 1)
    print(f"\n[attnfused] === Summary: {label} ===")
    print(f"    samples: {processed}")
    print(f"    accuracy@0.5: {acc:.2f}%  ({correct}/{processed})  mean_iou: {iou_sum/max(processed,1):.4f}")
    print(f"    total wall time: {total_wall:.1f}s ({total_wall/max(processed,1):.2f}s/sample avg)")


if __name__ == "__main__":
    main()
