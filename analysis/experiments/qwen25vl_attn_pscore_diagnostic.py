"""
qwen25vl_attn_pscore_diagnostic.py

Fast, model-free-for-residual / one-forward-pass-for-attention diagnostic: does a
"residual_energy x attention" fused pscore separate RefCOCO merge-groups inside the ground-truth
referring-expression bbox from those outside it BETTER than pure residual_energy alone (which
scored AUC=0.5376, barely above chance -- see QWEN25VL_APPCORR_LOG.md section 9)?

This mirrors the AppCorr precedent already validated on DINOv3/OpenVLA/ImageNet
(commits 26384fa/46d1016/7a4eb55/1f33a1f in this repo): mobile_pscore=residual_energy x
server_pscore=attention (there: cls_attn_prob; here, since Qwen2.5-VL's vision tower has NO CLS
token, the analogous CLS-less signal is patch_attn_prob: layer-averaged, head-averaged
attention MASS RECEIVED by each patch from all other patches in its segment -- "how much do other
patches attend to me", which for a full-attention layer is genuinely global, not local-window-only).

Attention weights are captured by manually replicating the LAST full-attention layer's
(index 31, one of `fullatt_block_indexes`) computation off the SIDE of the existing validated
`prepare_full_tokens`/`approx_forward(0, 31)` path (both untouched, still SDPA-based) -- i.e. we
get layer 31's INPUT via the already-forked, already-tested approx forward through layers 0-30,
then manually do norm1 -> qkv -> softmax(QK^T*scale) for layer 31 only, purely to read off the
attention probabilities (which `F.scaled_dot_product_attention` never exposes). No fork files are
modified for this diagnostic.

Run (appcorr env, needs 1 free GPU):
    python analysis/experiments/qwen25vl_attn_pscore_diagnostic.py --device cuda:0 --num-samples 400
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offload.common import Task, ExperimentConfig
from offload.common.protocol import Patch
from offload.policies import get_transmission
from offload.server.model.qwen25vl_executor import Qwen25VLExecutor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="offload/config/realworldqa_qwen25vl_32b_interleaved_g4.json")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-samples", type=int, default=400)
    p.add_argument("--attn-layer", type=int, default=31, help="Which full-attention layer's Q/K to use "
                    "for the manual attention-weight readout (must be in fullatt_block_indexes).")
    p.add_argument("--save-npz", type=str, default=None,
                    help="If set, dump raw per-merge-group [image_idx, label, residual, attn_<layer>...] "
                         "rows to this .npz path for offline weight-fitting (e.g. logistic regression to "
                         "find an optimal per-layer weighted combination instead of naive layermean).")
    p.add_argument("--per-head", action="store_true", help="Also keep the per-head breakdown (not just "
                    "the head-mean) of attention-received, so raw_rows/--save-npz include one "
                    "attn_<layer>_h<head> column per (layer, head) instead of one attn_<layer> column "
                    "per layer. Lets a downstream weight-fit assign an individual weight per head, not "
                    "just per layer -- num_heads=16, so this multiplies column count by 16.")
    return p.parse_args()


def _manual_attention_received(blk, x, vctx, layer_idx, tower, collapse_heads=True):
    """Manually replicates one block's own attention computation (norm1 -> qkv -> RoPE ->
    softmax(QK^T*scale)) purely to read off attention probabilities (F.scaled_dot_product_attention
    never exposes these). With collapse_heads=True (default), returns a [seq_len] tensor
    (permuted-sequence order) of mean attention MASS RECEIVED by each patch, averaged over heads and
    over query positions within its segment -- matches DINOv3's validated `patch_attn_prob`
    definition (attn.py:217-219 in that fork). With collapse_heads=False, returns a
    [num_heads, seq_len] tensor instead (head dim kept, only averaged over query positions), for a
    per-head weight fit."""
    x_normed = blk.norm1(x)
    attn = blk.attn
    seq_length = x_normed.shape[0]
    qkv = attn.qkv(x_normed).reshape(seq_length, 3, attn.num_heads, attn.head_dim).permute(1, 0, 2, 3)
    q, k, v = qkv.unbind(0)
    cos, sin = vctx["position_embeddings"]
    from appcorr.models.qwen25vl.vision.attention import apply_rotary_pos_emb_vision
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

    segs = vctx["cu_seqlens_ranges"] if layer_idx in tower.fullatt_block_indexes else vctx["cu_window_seqlens_ranges"]
    if collapse_heads:
        received = torch.zeros(seq_length, device=x.device, dtype=torch.float32)
    else:
        received = torch.zeros(attn.num_heads, seq_length, device=x.device, dtype=torch.float32)
    for start, length in segs:
        sl = slice(start, start + length)
        q_seg = q[sl].transpose(0, 1)
        k_seg = k[sl].transpose(0, 1)
        scale = attn.head_dim ** -0.5
        attn_prob = (q_seg @ k_seg.transpose(-2, -1) * scale).softmax(dim=-1)
        if collapse_heads:
            received[sl] = attn_prob.mean(dim=0).mean(dim=0).float()
        else:
            received[:, sl] = attn_prob.mean(dim=1).float()  # mean over queries only, keep heads
    return received


def compute_multi_layer_attention_received(executor, vctx, layer_indices, device, per_head=False):
    """Single forward pass through all 32 blocks (via the validated approx_forward, one layer at a
    time), capturing the manual attention-received readout at each requested layer_idx (using that
    layer's real INPUT, before its own approx() runs) -- avoids O(len(layer_indices)) redundant
    recomputation of the shared prefix. Returns {layer_idx: [seq_len] tensor}, or with per_head=True,
    {layer_idx: [num_heads, seq_len] tensor}."""
    tower = executor.vision_tower
    x = vctx["hidden_states"]
    cache = {}
    results = {}
    layer_indices = set(layer_indices)
    max_layer = max(layer_indices)
    for i in range(max_layer + 1):
        if i in layer_indices:
            results[i] = _manual_attention_received(tower.blocks[i], x, vctx, i, tower, collapse_heads=not per_head)
        x, cache = tower.approx_forward(x, i, i + 1, vctx, cache, tag_prefix="diag")
    return results


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    raw_config["batch_size"] = 1
    raw_config["device"] = args.device
    raw_config.setdefault("transmission_kwargs", {})["grouping_strategy"] = "top_energy"
    raw_config["transmission_kwargs"]["num_groups"] = 1
    raw_config["transmission_kwargs"]["keep_rate"] = 1.0  # irrelevant here, just need pscore+grid

    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset
    from PIL import Image

    print("[diag] loading model...")
    config = ExperimentConfig(**raw_config)
    executor = Qwen25VLExecutor(torch.device(args.device))
    executor.load_model(config.model_name, config)
    ip = executor.processor.image_processor
    min_pixels, max_pixels = ip.size["shortest_edge"], ip.size["longest_edge"]
    factor = ip.patch_size * ip.merge_size * 4
    encoder = get_transmission(raw_config["transmission_policy_name"])
    fullatt_layers = sorted(executor.vision_tower.fullatt_block_indexes)
    num_heads = executor.vision_tower.blocks[fullatt_layers[0]].attn.num_heads
    per_layer_inside = {i: [] for i in fullatt_layers}
    per_layer_outside = {i: [] for i in fullatt_layers}
    per_head_inside = {(l, h): [] for l in fullatt_layers for h in range(num_heads)} if args.per_head else None
    per_head_outside = {(l, h): [] for l in fullatt_layers for h in range(num_heads)} if args.per_head else None
    print(f"[diag] fullatt_layers={fullatt_layers} per_head={args.per_head} (num_heads={num_heads})")

    ds = load_dataset("lmms-lab/RefCOCO", split="val")
    n_total = len(ds)
    n_samples = args.num_samples
    stride = max(n_total // n_samples, 1)
    indices = list(range(0, n_total, stride))[:n_samples]
    print(f"[diag] {len(indices)} examples (of {n_total})")

    inside_res, outside_res = [], []
    inside_layermean, outside_layermean = [], []
    inside_fused, outside_fused = [], []
    raw_rows = [] if args.save_npz else None  # (image_idx, label, residual, attn_layer0, attn_layer1, ...)

    for count, idx in enumerate(indices):
        ex = ds[idx]
        image = ex["image"].convert("RGB")
        bx, by, bw, bh = ex["bbox"]
        gt = (bx, by, bx + bw, by + bh)
        target_h, target_w = smart_resize(image.height, image.width, factor=factor,
                                           min_pixels=min_pixels, max_pixels=max_pixels)
        sx, sy = target_w / image.width, target_h / image.height
        gt_r = (gt[0] * sx, gt[1] * sy, gt[2] * sx, gt[3] * sy)
        image_np = np.array(image.resize((target_w, target_h), Image.BILINEAR), dtype=np.uint8)

        cfg_d = dict(raw_config)
        cfg_d["image_shape"] = [target_h, target_w, 3]
        cfg = ExperimentConfig(**cfg_d)

        # residual-energy pscore (CPU, same as before)
        levels = sorted(cfg.transmission_kwargs.get("pyramid_levels", [2, 0]), reverse=True)
        gaussians = encoder._build_native_gaussians(image_np, levels[0])
        structure = encoder._collect_residual_metadata_scored(gaussians, cfg, (target_h, target_w))

        # attention-received pscore (GPU, ALL full-attention layers in one forward pass)
        context = {}
        task = Task(task_id=0, request_id=0, payload=[Patch(image_idx=0, spatial_idx=0, data=b"", text_payload="x")], instructions=[])
        executor.preprocess(image_np[None], task, context, cfg)
        vctx = executor.vision_tower.prepare_full_tokens(context["pixel_values"], context["image_grid_thw"])
        with torch.no_grad():
            received_by_layer = compute_multi_layer_attention_received(
                executor, vctx, fullatt_layers, args.device, per_head=args.per_head)

        unit = executor.vision_tower.spatial_merge_unit
        inv_window_index = vctx["inv_window_index"]  # inv_window_index[native_i] = permuted slot of native group i
        attn_scores_by_layer = {}
        attn_scores_by_layer_head = {} if args.per_head else None
        for lyr, received in received_by_layer.items():
            if args.per_head:
                # received: [num_heads, seq_len] -> [num_heads, n_groups] -> gathered to native order.
                n_groups = received.shape[1] // unit
                received_per_group_permuted = received.view(received.shape[0], n_groups, unit).mean(dim=2)
                per_head_native = received_per_group_permuted[:, inv_window_index].cpu().numpy()  # [num_heads, n_groups]
                attn_scores_by_layer_head[lyr] = per_head_native
                attn_scores_by_layer[lyr] = per_head_native.mean(axis=0)
            else:
                n_groups = received.shape[0] // unit
                received_per_group_permuted = received.view(n_groups, unit).mean(dim=1)
                # Same gather pattern as backbone.py's get_merged_output (`merged[ctx["inv_window_index"]]`).
                attn_scores_by_layer[lyr] = received_per_group_permuted[inv_window_index].cpu().numpy()
        layermean_scores = np.mean(np.stack(list(attn_scores_by_layer.values())), axis=0)

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
            lm_score = float(layermean_scores[sp_idx]) if sp_idx < len(layermean_scores) else float("nan")
            fused = r_score * lm_score
            bucket = (inside_res, inside_layermean, inside_fused) if inside else (outside_res, outside_layermean, outside_fused)
            bucket[0].append(r_score); bucket[1].append(lm_score); bucket[2].append(fused)
            per_layer_a = []
            for lyr in fullatt_layers:
                a_score = float(attn_scores_by_layer[lyr][sp_idx]) if sp_idx < len(attn_scores_by_layer[lyr]) else float("nan")
                (per_layer_inside if inside else per_layer_outside)[lyr].append(a_score)
                per_layer_a.append(a_score)
            per_head_a = []
            if args.per_head:
                for lyr in fullatt_layers:
                    heads_arr = attn_scores_by_layer_head[lyr]  # [num_heads, n_groups]
                    for h in range(num_heads):
                        h_score = float(heads_arr[h, sp_idx]) if sp_idx < heads_arr.shape[1] else float("nan")
                        (per_head_inside if inside else per_head_outside)[(lyr, h)].append(h_score)
                        per_head_a.append(h_score)
            if raw_rows is not None:
                raw_rows.append([idx, float(inside), r_score] + per_layer_a + per_head_a)

        if (count + 1) % 50 == 0:
            print(f"  [{count+1}/{len(indices)}] processed", flush=True)

    def auc(ins, outs):
        ins, outs = np.array(ins), np.array(outs)
        all_s = np.concatenate([ins, outs])
        labels = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
        order = np.argsort(all_s)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(all_s) + 1)
        n1, n0 = len(ins), len(outs)
        return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    print(f"\n[diag] === RESULTS (n_inside={len(inside_res)}, n_outside={len(outside_res)}) ===")
    print(f"  residual_energy alone:                    AUC={auc(inside_res, outside_res):.4f}")
    for lyr in fullatt_layers:
        print(f"  attention layer {lyr:>2} alone:                AUC={auc(per_layer_inside[lyr], per_layer_outside[lyr]):.4f}")
    print(f"  attention LAYERMEAN ({fullatt_layers}) alone: AUC={auc(inside_layermean, outside_layermean):.4f}")
    print(f"  residual x attention LAYERMEAN (fused):   AUC={auc(inside_fused, outside_fused):.4f}")

    if args.per_head:
        print(f"\n[diag] === PER-HEAD RESULTS (num_heads={num_heads}) ===")
        best = []
        for lyr in fullatt_layers:
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
        cols = ["image_idx", "label", "residual"] + [f"attn_{l}" for l in fullatt_layers]
        if args.per_head:
            cols += [f"attn_{l}_h{h}" for l in fullatt_layers for h in range(num_heads)]
        np.savez(args.save_npz, data=arr, columns=np.array(cols))
        print(f"[diag] saved {arr.shape[0]} rows x {arr.shape[1]} cols to {args.save_npz} (columns: {cols if not args.per_head else f'{len(cols)} cols (per-head)'})")


if __name__ == "__main__":
    main()
