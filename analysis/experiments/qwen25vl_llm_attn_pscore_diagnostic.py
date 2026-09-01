"""
qwen25vl_llm_attn_pscore_diagnostic.py

Query-CONDITIONED pscore diagnostic (task #56), following up on the vision-tower-only
`qwen25vl_attn_pscore_diagnostic.py` (content-only, no awareness of the actual question/expression
text). This one extracts, from the LLM decoder's OWN causal self-attention, how much attention
mass the REAL post-image text (the RefCOCO referring expression + assistant-prompt prefix) sends to
each image token -- a signal that changes depending on what's actually being asked, unlike the
vision tower's self-attention.

Mechanism (no changes to appcorr/models/qwen25vl/llm/decoder_layer.py or the executor):
  1. Build the REAL prompt (GROUNDING_PROMPT_TMPL with the actual referring expression) via
     `executor.preprocess`/`prepare_tokens`, so `context["input_ids"]`/`position_ids` reflect the
     genuine question, not a placeholder.
  2. Run the vision tower to full depth + splice (matching every other driver's "first call"
     convention), then walk LLM layers 0..max_candidate_layer via the validated `.approx()`, capturing
     a manual attention readout at each requested layer's real INPUT (mirrors
     `compute_multi_layer_attention_received`'s one-shared-pass pattern).
  3. At each candidate layer: manually replicate `self_attn.approx()`'s q/k/v projection + M-RoPE
     (since `F.scaled_dot_product_attention` never exposes probabilities), restrict QUERY rows to
     the text positions strictly AFTER the image (the real question + assistant-prompt prefix -- text
     positions before the image, e.g. a system prompt, are causally forbidden from seeing the image
     at all and would just dilute the average with structural zeros), keep the FULL key set (all N
     positions, causal-masked) so the softmax is correctly normalized against the SAME denominator
     the model actually used (not renormalized over just the image subset), then read off the
     image-token columns. Because `get_merged_output` already un-permutes to NATIVE merge-group order
     before splicing (see backbone.py's `inv_window_index` un-permute), `image_token_positions[g]`
     already corresponds directly to merge-group `g` -- no window/unit-index gymnastics needed here,
     unlike the vision-tower diagnostic.

Run (appcorr env, needs 1 free GPU):
    python analysis/experiments/qwen25vl_llm_attn_pscore_diagnostic.py --device cuda:0 --num-samples 400
"""

import argparse
import json
import sys
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

from analysis.experiments.refcoco_offload_eval import GROUNDING_PROMPT_TMPL
from appcorr.models.qwen25vl.llm.decoder_layer import apply_multimodal_rotary_pos_emb, repeat_kv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="offload/config/realworldqa_qwen25vl_32b_interleaved_g4.json")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-samples", type=int, default=400)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 16, 32, 48, 63],
                    help="Which LLM decoder layer indices to extract text->image attention from.")
    p.add_argument("--per-head", action="store_true", help="Keep per-head breakdown instead of the "
                    "head-mean (mirrors qwen25vl_attn_pscore_diagnostic.py's --per-head).")
    p.add_argument("--save-npz", type=str, default=None,
                    help="If set, dump raw per-merge-group [image_idx, label, residual, "
                         "llmattn_<layer>...] rows to this .npz path.")
    return p.parse_args()


def _manual_llm_attention_received(self_attn, x_normed, position_ids, text_query_idx, image_idx,
                                    collapse_heads=True):
    """x_normed: [1, N, C] (already input_layernorm'd). Computes causal QK^T*scale softmax for
    QUERY rows = text_query_idx (post-image text positions) against the FULL N-length causal key
    set, then reads off the image-key columns -- the correctly-normalized 'how much attention mass
    does the (post-image) text send to each image token'. Returns [n_image] (collapse_heads=True)
    or [num_heads, n_image] (collapse_heads=False)."""
    q, k, v = self_attn._project_heads(x_normed)  # [1, H(_kv), N, Dh]
    cos, sin = self_attn.rotary_emb(v, position_ids)
    q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self_attn.mrope_section)
    k_full = repeat_kv(k, self_attn.num_key_value_groups)  # [1, H, N, Dh]
    scale = self_attn.head_dim ** -0.5

    N = k_full.shape[2]
    q_text = q[:, :, text_query_idx, :]  # [1, H, Qtext, Dh]
    scores = (q_text.float() @ k_full.float().transpose(-2, -1)) * scale  # [1, H, Qtext, N]

    key_positions = torch.arange(N, device=q.device).view(1, 1, 1, N)
    query_positions = text_query_idx.view(1, 1, -1, 1)
    scores = scores.masked_fill(key_positions > query_positions, float("-inf"))
    attn_prob = scores.softmax(dim=-1)  # [1, H, Qtext, N], normalized over the REAL causal key set

    received_image = attn_prob[:, :, :, image_idx]  # [1, H, Qtext, n_image]
    if collapse_heads:
        return received_image.mean(dim=(0, 1, 2))  # [n_image]
    return received_image.mean(dim=(0, 2))  # [H, n_image]


def compute_llm_text_to_image_attention(executor, context, layer_indices, per_head=False):
    """Single shared-prefix pass through llm_layers[0..max_layer] (via the validated `.approx()`),
    capturing the manual text->image attention-received readout at each requested layer's real
    INPUT. Returns {layer_idx: tensor}, already in NATIVE merge-group order (== image_token_positions'
    order -- get_merged_output already un-permutes before splicing, see module docstring)."""
    layer_set = set(layer_indices)
    max_layer = max(layer_set)
    x = context["llm_input_embeds"]
    cache = {}
    permanent_idx = context["permanent_group_idx"]
    image_idx = context["image_token_positions"]
    text_query_idx = permanent_idx[permanent_idx > image_idx.max()]
    results = {}
    for i in range(max_layer + 1):
        layer = executor.llm_layers[i]
        if i in layer_set:
            x_normed = layer.input_layernorm(x)
            results[i] = _manual_llm_attention_received(
                layer.self_attn, x_normed, context["position_ids"], text_query_idx, image_idx,
                collapse_heads=not per_head,
            )
        x, cache = layer.approx(x, context["position_ids"], cache, tag=f"llmdiag_layer{i}")
    return results


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    raw_config["batch_size"] = 1
    raw_config["device"] = args.device
    raw_config.setdefault("transmission_kwargs", {})["grouping_strategy"] = "top_energy"
    raw_config["transmission_kwargs"]["num_groups"] = 1
    raw_config["transmission_kwargs"]["keep_rate"] = 1.0

    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset
    from PIL import Image

    print("[llmdiag] loading model...")
    config = ExperimentConfig(**raw_config)
    executor = Qwen25VLExecutor(torch.device(args.device))
    executor.load_model(config.model_name, config)
    ip = executor.processor.image_processor
    min_pixels, max_pixels = ip.size["shortest_edge"], ip.size["longest_edge"]
    factor = ip.patch_size * ip.merge_size * 4
    encoder = get_transmission(raw_config["transmission_policy_name"])
    total_llm_layers = executor.num_llm_layers
    layers = sorted(args.layers)
    num_heads = executor.llm_layers[0].self_attn.num_heads
    print(f"[llmdiag] num_llm_layers={total_llm_layers} candidate layers={layers} per_head={args.per_head} (num_heads={num_heads})")

    ds = load_dataset("lmms-lab/RefCOCO", split="val")
    n_total = len(ds)
    n_samples = args.num_samples
    stride = max(n_total // n_samples, 1)
    indices = list(range(0, n_total, stride))[:n_samples]
    print(f"[llmdiag] {len(indices)} examples (of {n_total})")

    per_layer_inside = {i: [] for i in layers}
    per_layer_outside = {i: [] for i in layers}
    per_head_inside = {(l, h): [] for l in layers for h in range(num_heads)} if args.per_head else None
    per_head_outside = {(l, h): [] for l in layers for h in range(num_heads)} if args.per_head else None
    inside_res, outside_res = [], []
    raw_rows = [] if args.save_npz else None  # (image_idx, label, residual, llmattn_layer0, ...)

    for count, idx in enumerate(indices):
        ex = ds[idx]
        image = ex["image"].convert("RGB")
        expr = ex["answer"][0] if isinstance(ex["answer"], list) and ex["answer"] else str(ex["answer"])
        bx, by, bw, bh = ex["bbox"]
        gt = (bx, by, bx + bw, by + bh)
        prompt = GROUNDING_PROMPT_TMPL.format(expr=expr)
        target_h, target_w = smart_resize(image.height, image.width, factor=factor,
                                           min_pixels=min_pixels, max_pixels=max_pixels)
        sx, sy = target_w / image.width, target_h / image.height
        gt_r = (gt[0] * sx, gt[1] * sy, gt[2] * sx, gt[3] * sy)
        image_np = np.array(image.resize((target_w, target_h), Image.BILINEAR), dtype=np.uint8)

        cfg_d = dict(raw_config)
        cfg_d["image_shape"] = [target_h, target_w, 3]
        cfg = ExperimentConfig(**cfg_d)

        # residual-energy pscore (CPU, unchanged)
        pyr_levels = sorted(cfg.transmission_kwargs.get("pyramid_levels", [2, 0]), reverse=True)
        gaussians = encoder._build_native_gaussians(image_np, pyr_levels[0])
        structure = encoder._collect_residual_metadata_scored(gaussians, cfg, (target_h, target_w))

        # real question + vision, through the executor's normal preprocess/prepare_tokens/approx path
        context = {}
        task = Task(task_id=0, request_id=0,
                    payload=[Patch(image_idx=0, spatial_idx=0, data=b"", text_payload=prompt)], instructions=[])
        executor.preprocess(image_np[None], task, context, cfg)
        executor.prepare_tokens(task, context, cfg)
        total_v = len(executor.vision_tower.blocks)
        with torch.no_grad():
            x_v, cache_v = executor.vision_tower.approx_forward(
                context["vision_current_feature"], 0, total_v, context["vision_ctx"], {}, tag_prefix="vision"
            )
            merged = executor.vision_tower.get_merged_output(x_v, context["vision_ctx"])
            context["llm_input_embeds"] = executor._splice_image_embeds(context, merged)

            received_by_layer = compute_llm_text_to_image_attention(executor, context, layers, per_head=args.per_head)

        image_idx_t = context["image_token_positions"]
        n_image = image_idx_t.shape[0]
        attn_scores_by_layer = {}
        attn_scores_by_layer_head = {} if args.per_head else None
        for lyr, received in received_by_layer.items():
            if args.per_head:
                attn_scores_by_layer_head[lyr] = received.cpu().numpy()  # [H, n_image], native order
                attn_scores_by_layer[lyr] = attn_scores_by_layer_head[lyr].mean(axis=0)
            else:
                attn_scores_by_layer[lyr] = received.cpu().numpy()  # [n_image], native order

        for item in structure:
            gh, gw = item["grid_hw"]
            row, col = item["row"], item["col"]
            cell_h, cell_w = target_h / gh, target_w / gw
            y0, y1 = row * cell_h, (row + 1) * cell_h
            x0, x1 = col * cell_w, (col + 1) * cell_w
            ox0, oy0 = max(x0, gt_r[0]), max(y0, gt_r[1])
            ox1, oy1 = min(x1, gt_r[2]), min(y1, gt_r[3])
            overlap = max(0, ox1 - ox0) * max(0, oy1 - oy0)
            inside = overlap > 0.5 * (x1 - x0) * (y1 - y0)
            sp_idx = item["spatial_idx"]
            r_score = item["pscore"]
            (inside_res if inside else outside_res).append(r_score)

            per_layer_a = []
            for lyr in layers:
                a_score = float(attn_scores_by_layer[lyr][sp_idx]) if sp_idx < n_image else float("nan")
                (per_layer_inside if inside else per_layer_outside)[lyr].append(a_score)
                per_layer_a.append(a_score)
            per_head_a = []
            if args.per_head:
                for lyr in layers:
                    heads_arr = attn_scores_by_layer_head[lyr]  # [H, n_image]
                    for h in range(num_heads):
                        h_score = float(heads_arr[h, sp_idx]) if sp_idx < n_image else float("nan")
                        (per_head_inside if inside else per_head_outside)[(lyr, h)].append(h_score)
                        per_head_a.append(h_score)
            if raw_rows is not None:
                raw_rows.append([idx, float(inside), r_score] + per_layer_a + per_head_a)

        if (count + 1) % 50 == 0:
            print(f"  [{count + 1}/{len(indices)}] processed", flush=True)

    def auc(ins, outs):
        ins, outs = np.array(ins), np.array(outs)
        all_s = np.concatenate([ins, outs])
        labels = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
        order = np.argsort(all_s)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(all_s) + 1)
        n1, n0 = len(ins), len(outs)
        return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    print(f"\n[llmdiag] === RESULTS (n_inside={len(inside_res)}, n_outside={len(outside_res)}) ===")
    print(f"  residual_energy alone:                    AUC={auc(inside_res, outside_res):.4f}")
    for lyr in layers:
        print(f"  LLM text->image attn, layer {lyr:>2} alone:   AUC={auc(per_layer_inside[lyr], per_layer_outside[lyr]):.4f}")

    if args.per_head:
        print(f"\n[llmdiag] === PER-HEAD RESULTS (num_heads={num_heads}) ===")
        best = []
        for lyr in layers:
            for h in range(num_heads):
                a = auc(per_head_inside[(lyr, h)], per_head_outside[(lyr, h)])
                best.append((a, lyr, h))
        best.sort(reverse=True)
        print("  top 10 heads by AUC:")
        for a, lyr, h in best[:10]:
            print(f"    layer {lyr:>2} head {h:>2}: AUC={a:.4f}")
        print("  bottom 5 heads by AUC:")
        for a, lyr, h in best[-5:]:
            print(f"    layer {lyr:>2} head {h:>2}: AUC={a:.4f}")

    if raw_rows is not None:
        arr = np.array(raw_rows, dtype=np.float64)
        cols = ["image_idx", "label", "residual"] + [f"llmattn_{l}" for l in layers]
        if args.per_head:
            cols += [f"llmattn_{l}_h{h}" for l in layers for h in range(num_heads)]
        np.savez(args.save_npz, data=arr, columns=np.array(cols))
        print(f"[llmdiag] saved {arr.shape[0]} rows x {arr.shape[1]} cols to {args.save_npz}")


if __name__ == "__main__":
    main()
