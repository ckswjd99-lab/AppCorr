"""
qwen25vl_query_first_diagnostic.py

Tests whether putting the question text BEFORE the image tokens (query-first) vs the standard
Qwen2.5-VL image-first ordering changes things, on nr=400 RefCOCO. Two questions:

  (A) BASE ACCURACY: does reordering hurt the stock model? Qwen2.5-VL is trained image-first, so
      query-first is off-distribution -- this measures how much. Pure stock model.generate on the
      full-resolution image for both orderings (no offload/approx/correction involved), exact-match
      + IoU scored exactly like every other RefCOCO eval this session.

  (B) QUERY-CONDITIONED ATTENTION AUC vs GT-bbox: with image-first, the query-conditioned signal is
      text->image (later text attends back to earlier image -- the current llmattn_36 pscore). With
      query-first, it flips to image->text (each image token, now AFTER the text, attends back to
      the question at its OWN query position -- potentially a cleaner per-patch signal). This
      measures whether image->text separates GT-bbox merge-groups better than text->image, at the
      same candidate layers, so we know if the (accuracy-costing) reorder would even buy a better
      signal.

Run (appcorr env, 1 free GPU):
    python analysis/experiments/qwen25vl_query_first_diagnostic.py --device cuda:0 --num-samples 400
"""

import argparse
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

from analysis.experiments.refcoco_offload_eval import GROUNDING_PROMPT_TMPL, score_answer
from appcorr.models.qwen25vl.llm.decoder_layer import apply_multimodal_rotary_pos_emb, repeat_kv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="offload/config/realworldqa_qwen25vl_32b_interleaved_g4.json")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-samples", type=int, default=400)
    p.add_argument("--layers", type=int, nargs="+", default=[24, 32, 36, 40, 48],
                    help="LLM decoder layers to read the query-conditioned attention from.")
    p.add_argument("--max-new-tokens", type=int, default=63)
    return p.parse_args()


def build_ordered_inputs(processor, image_pil, question, order, device):
    """Stock processor path, image-first or query-first content order."""
    if order == "image_first":
        content = [{"type": "image"}, {"type": "text", "text": question}]
    else:  # query_first
        content = [{"type": "text", "text": question}, {"type": "image"}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image_pil], return_tensors="pt")
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def _manual_query_cond_attention(self_attn, x_normed, position_ids, query_idx, key_idx):
    """For each token in query_idx (a query row), how much total attention it sends to the tokens in
    key_idx, under the real causal mask, normalized over the full causal key set. Returns a score
    indexed by query_idx order. Used both ways:
      image-first (text->image): query_idx = text positions, key_idx = image positions -> but there
        each text row's attention to ALL image keys must be aggregated back to per-image; that's the
        existing compute_llm_text_to_image_attention's job (read image COLUMNS). NOT this function.
      query-first (image->text): query_idx = image positions, key_idx = text positions -> each image
        token is its own query row, so its attention-to-text is a single per-image-token scalar.
        THIS function returns exactly that, one value per image token, in query_idx order."""
    q, k, v = self_attn._project_heads(x_normed)  # [1, H(_kv), N, Dh]
    cos, sin = self_attn.rotary_emb(v, position_ids)
    q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self_attn.mrope_section)
    k_full = repeat_kv(k, self_attn.num_key_value_groups)
    scale = self_attn.head_dim ** -0.5
    N = k_full.shape[2]
    q_sel = q[:, :, query_idx, :]  # [1, H, Q, Dh]
    scores = (q_sel.float() @ k_full.float().transpose(-2, -1)) * scale  # [1, H, Q, N]
    key_pos = torch.arange(N, device=q.device).view(1, 1, 1, N)
    qpos = query_idx.view(1, 1, -1, 1)
    scores = scores.masked_fill(key_pos > qpos, float("-inf"))
    attn_prob = scores.softmax(dim=-1)  # [1, H, Q, N]
    to_key = attn_prob[:, :, :, key_idx].sum(dim=-1)  # [1, H, Q] total attention each query sends to key set
    return to_key.mean(dim=(0, 1))  # [Q], head-mean


def main():
    args = parse_args()
    import json as _json
    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = _json.load(f)
    raw_config["batch_size"] = 1
    raw_config["device"] = args.device
    raw_config.setdefault("transmission_kwargs", {})["grouping_strategy"] = "top_energy"
    raw_config["transmission_kwargs"]["num_groups"] = 1
    raw_config["transmission_kwargs"]["keep_rate"] = 1.0

    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset
    from PIL import Image

    print("[qf] loading model...")
    config = ExperimentConfig(**raw_config)
    executor = Qwen25VLExecutor(torch.device(args.device))
    executor.load_model(config.model_name, config)
    processor = executor.processor
    model = executor.model
    ip = processor.image_processor
    min_pixels, max_pixels = ip.size["shortest_edge"], ip.size["longest_edge"]
    factor = ip.patch_size * ip.merge_size * 4
    encoder = get_transmission(raw_config["transmission_policy_name"])
    layers = sorted(args.layers)
    print(f"[qf] candidate LLM layers={layers}")

    ds = load_dataset("lmms-lab/RefCOCO", split="val")
    n_total = len(ds)
    stride = max(n_total // args.num_samples, 1)
    indices = list(range(0, n_total, stride))[:args.num_samples]
    print(f"[qf] {len(indices)} examples (of {n_total})")

    # (A) accuracy accumulators
    acc = {"image_first": [0, 0.0], "query_first": [0, 0.0]}  # [n_correct, iou_sum]
    n_acc = 0
    # (B) AUC accumulators: image-first text->image vs query-first image->text
    ti_inside = {l: [] for l in layers}; ti_outside = {l: [] for l in layers}  # text->image (image-first)
    it_inside = {l: [] for l in layers}; it_outside = {l: [] for l in layers}  # image->text (query-first)

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
        resized = image.resize((target_w, target_h), Image.BILINEAR)
        image_np = np.array(resized, dtype=np.uint8)

        # ---- (A) stock accuracy, both orderings ----
        for order in ("image_first", "query_first"):
            inputs = build_ordered_inputs(processor, resized, prompt, order, args.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            pred_text = processor.tokenizer.decode(gen, skip_special_tokens=True)
            ok, iou = score_answer(pred_text, gt_r)
            acc[order][0] += int(ok); acc[order][1] += iou
        n_acc += 1

        # ---- (B) attention AUC vs GT-bbox, both orderings ----
        # residual structure for GT-bbox inside/outside labels + spatial_idx
        cfg_d = dict(raw_config); cfg_d["image_shape"] = [target_h, target_w, 3]
        cfg = ExperimentConfig(**cfg_d)
        pyr = sorted(cfg.transmission_kwargs.get("pyramid_levels", [2, 0]), reverse=True)
        gaussians = encoder._build_native_gaussians(image_np, pyr[0])
        structure = encoder._collect_residual_metadata_scored(gaussians, cfg, (target_h, target_w))

        for order, inside_d, outside_d in (("image_first", ti_inside, ti_outside),
                                            ("query_first", it_inside, it_outside)):
            context = {}
            executor._build_prompt_order = order  # consumed by a small patch below
            task = Task(task_id=0, request_id=0,
                        payload=[Patch(image_idx=0, spatial_idx=0, data=b"", text_payload=prompt)], instructions=[])
            executor.preprocess(image_np[None], task, context, cfg)
            _build_prompt_ordered(executor, context, order)
            executor.prepare_tokens(task, context, cfg)  # vision_ctx; input_ids already set
            total_v = len(executor.vision_tower.blocks)
            with torch.no_grad():
                x_v, _ = executor.vision_tower.approx_forward(
                    context["vision_current_feature"], 0, total_v, context["vision_ctx"], {}, tag_prefix="qf")
                merged = executor.vision_tower.get_merged_output(x_v, context["vision_ctx"])
                context["llm_input_embeds"] = executor._splice_image_embeds(context, merged)
                scores_by_layer = _query_cond_by_layer(executor, context, layers, order)

            image_idx_t = context["image_token_positions"]
            n_image = image_idx_t.shape[0]
            for item in structure:
                gh, gw = item["grid_hw"]; row, col = item["row"], item["col"]
                cell_h, cell_w = target_h / gh, target_w / gw
                y0, y1 = row * cell_h, (row + 1) * cell_h
                x0, x1 = col * cell_w, (col + 1) * cell_w
                ox0, oy0 = max(x0, gt_r[0]), max(y0, gt_r[1])
                ox1, oy1 = min(x1, gt_r[2]), min(y1, gt_r[3])
                overlap = max(0, ox1 - ox0) * max(0, oy1 - oy0)
                inside = overlap > 0.5 * (x1 - x0) * (y1 - y0)
                sp = item["spatial_idx"]
                for l in layers:
                    v = float(scores_by_layer[l][sp]) if sp < n_image else float("nan")
                    (inside_d if inside else outside_d)[l].append(v)

        if (count + 1) % 25 == 0:
            a_if = 100.0 * acc["image_first"][0] / n_acc
            a_qf = 100.0 * acc["query_first"][0] / n_acc
            print(f"  [{count+1}/{len(indices)}] running acc: image_first={a_if:.1f}% query_first={a_qf:.1f}%", flush=True)

    def auc(ins, outs):
        ins, outs = np.array(ins), np.array(outs)
        all_s = np.concatenate([ins, outs])
        labels = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
        order = np.argsort(all_s)
        ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(all_s) + 1)
        n1, n0 = len(ins), len(outs)
        return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    print(f"\n[qf] === (A) BASE ACCURACY (stock model.generate, full-res, N={n_acc}) ===")
    for order in ("image_first", "query_first"):
        c, iou = acc[order]
        print(f"    {order:12s}: acc@0.5={100.0*c/n_acc:.2f}% ({c}/{n_acc})  mean_iou={iou/n_acc:.4f}")

    print(f"\n[qf] === (B) QUERY-CONDITIONED ATTENTION AUC vs GT-bbox ===")
    print(f"    {'layer':>6s} {'text->image (image-first)':>26s} {'image->text (query-first)':>26s}")
    for l in layers:
        a_ti = auc(ti_inside[l], ti_outside[l])
        a_it = auc(it_inside[l], it_outside[l])
        print(f"    {l:6d} {a_ti:26.4f} {a_it:26.4f}")


def _build_prompt_ordered(executor, context, order):
    """Rebuild context's input_ids/position_ids/masks for the given content order, mirroring
    executor._build_prompt but with configurable image/text order. (executor._build_prompt is
    image-first only.)"""
    question = context["question"]
    grid_thw = context["image_grid_thw"]
    if order == "image_first":
        content = [{"type": "image"}, {"type": "text", "text": question}]
    else:
        content = [{"type": "text", "text": question}, {"type": "image"}]
    messages = [{"role": "user", "content": content}]
    text = executor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    merge_unit = executor.vision_tower.spatial_merge_unit
    num_image_tokens = int((grid_thw.prod(dim=-1) // merge_unit).sum().item())
    image_pad = "<|image_pad|>" * num_image_tokens
    text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", f"<|vision_start|>{image_pad}<|vision_end|>")
    tok_out = executor.processor.tokenizer(text, return_tensors="pt")
    input_ids = tok_out["input_ids"].to(executor.device)
    attention_mask = tok_out["attention_mask"].to(executor.device)
    image_mask_1d = (input_ids[0] == executor.image_token_id)
    mm_token_type_ids = image_mask_1d.long().unsqueeze(0)
    position_ids, _ = executor.model.model.get_rope_index(
        input_ids, mm_token_type_ids, image_grid_thw=grid_thw, attention_mask=attention_mask)
    context["input_ids"] = input_ids
    context["attention_mask"] = attention_mask
    context["image_mask_1d"] = image_mask_1d
    context["position_ids"] = position_ids
    context["permanent_group_idx"] = (~image_mask_1d).nonzero(as_tuple=True)[0]
    context["image_token_positions"] = image_mask_1d.nonzero(as_tuple=True)[0]
    input_embeds = executor.model.model.language_model.embed_tokens(input_ids)
    context["llm_input_embeds_template"] = input_embeds


def _query_cond_by_layer(executor, context, layers, order):
    """One shared-prefix pass; at each requested layer read the query-conditioned attention in
    NATIVE merge-group order. image-first: text->image (read image columns, aggregate over text
    query rows). query-first: image->text (each image token's own attention-to-text, one value)."""
    from analysis.experiments.qwen25vl_llm_attn_pscore_diagnostic import _manual_llm_attention_received
    layer_set = set(layers); max_layer = max(layer_set)
    x = context["llm_input_embeds"]; cache = {}
    perm = context["permanent_group_idx"]; image_idx = context["image_token_positions"]
    text_idx = perm  # all non-image (text) positions
    results = {}
    for i in range(max_layer + 1):
        layer = executor.llm_layers[i]
        if i in layer_set:
            xn = layer.input_layernorm(x)
            if order == "image_first":
                # text query rows must be causally after the image -> the ones with index > max image idx
                tq = perm[perm > image_idx.max()]
                results[i] = _manual_llm_attention_received(
                    layer.self_attn, xn, context["position_ids"], tq, image_idx, collapse_heads=True).cpu().numpy()
            else:
                # image tokens are the queries; text tokens (before them) are the keys
                results[i] = _manual_query_cond_attention(
                    layer.self_attn, xn, context["position_ids"], image_idx, text_idx).cpu().numpy()
        x, cache = layer.approx(x, context["position_ids"], cache, tag=f"qf_layer{i}")
    return results


if __name__ == "__main__":
    main()
