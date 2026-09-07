"""
unified.py

The Muse Glimmer (29.6B) progressive-arrival axis: vision tower + causal decoder as one in-process
object, following `appcorr/models/qwen35/unified.py` (whose streaming arm this mirrors almost
line-for-line) -- MG is the fourth measured point on the causal<->bidirectional axis.

**Why streaming, not interleaved.** MG's text model builds masks purely from
`create_causal_mask` / `create_sliding_window_causal_mask` -- image tokens are CAUSAL (no
Gemma3-style bidirectional island), 39/52 layers sliding (window 2048) + 13 full. So the LLM half
streams (chunked prefill, exact by causality) and only the vision half runs
approximate-then-correct, same category as OV2/Mistral/Qwen3.5.

Deltas from the qwen35 axis, all simplifications or MG quirks:

  - **1D RoPE.** No get_rope_index / mm_token_type_ids: positions are a plain arange over the
    sequence, sliced per chunk. (Qwen3.5's M-RoPE trap does not exist here.)
  - **Groups are pixel-shuffle blocks.** An LLM image token is `merge_size**2` raw rows gathered by
    `shuffle_index` (see vision/backbone.py) -- bands are contiguous LLM-token ranges, which is
    contiguous in shuffle order, NOT in raw-row order. All row bookkeeping goes through the tower's
    helpers.
  - **Embedding quirk**: stock embeds image-token positions as token id 0 and then masked_scatters
    the vision features in; `emb_all` reproduces that (ids.clone() with image ids zeroed) so the
    identity gate compares like with like.
  - **No thinking flag**: MG's chat template has no think block; the driver applies its own
    reasoning_strength conventions only for generation-time work, which this axis does not do.
"""

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .vision.backbone import ApproxCorrectMuseGlimmerVisionTower

MODEL_ID_MG30 = "meta-models/Muse-Glimmer-30B"


class MuseGlimmerAxis(nn.Module):
    def __init__(self, model: nn.Module, processor: Any, flop_counter: Optional[Any] = None):
        super().__init__()
        self.model = model            # MuseGlimmerForConditionalGeneration
        self.processor = processor
        self.flops = flop_counter
        self.tower = ApproxCorrectMuseGlimmerVisionTower(model.model)
        self.lm = model.model.language_model
        self.cfg = model.config
        self.image_token_id = model.config.image_token_id

    def _arrival(self, index: int):
        return self.flops.arrival(index) if self.flops is not None else nullcontext()

    def _stage(self, name: str):
        return self.flops.stage(name) if self.flops is not None else nullcontext()

    @torch.no_grad()
    def build_inputs(self, image, question: str, reasoning_strength: Optional[str] = "low",
                     force_answer_channel: bool = True) -> Dict[str, Any]:
        """ATEM channel-protocol conventions baked in (same as vlm_bounds_oracle's MG path, which
        measured the existing MMVP/CVB bounds): `reasoning_strength=low` and a forced
        ` to=user<|message|>` tail on the generation prompt -- without both, short greedy decodes
        never leave the to=self deliberation channel (the 49%/30% incidents)."""
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": question}]}]
        kw = {}
        if reasoning_strength:
            kw["reasoning_strength"] = reasoning_strength
        enc = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", **kw,
        )
        if force_answer_channel:
            tail = self.processor.tokenizer(" to=user<|message|>", add_special_tokens=False,
                                            return_tensors="pt").input_ids.to(enc["input_ids"].device)
            enc["input_ids"] = torch.cat([enc["input_ids"], tail], dim=1)
            if "attention_mask" in enc:
                enc["attention_mask"] = torch.cat(
                    [enc["attention_mask"], torch.ones_like(tail)], dim=1)
        return enc

    def _image_token_run(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        pos = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            raise ValueError("no image tokens in input_ids")
        if not bool((pos[1:] - pos[:-1] == 1).all()):
            raise ValueError("image tokens are not one contiguous run; streaming bands need one")
        return int(pos[0]), int(pos.numel())

    def _bands(self, groups: int, n_groups_total: int) -> List[Tuple[int, int]]:
        edges = [round(k * n_groups_total / groups) for k in range(groups + 1)]
        return [(a, b) for a, b in zip(edges[:-1], edges[1:])]

    def _embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Stock's embedding of the token sequence: image-token slots embed id 0 (they are
        overwritten by vision features either way)."""
        llm_ids = ids.clone()
        llm_ids[ids == self.image_token_id] = 0
        return self.model.get_input_embeddings()(llm_ids)

    # --- the reference -------------------------------------------------------------------------- #

    @torch.no_grad()
    def full_forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        with self._arrival(0), self._stage("full"):
            out = self.model(input_ids=inputs["input_ids"],
                             pixel_values=inputs["pixel_values"].to(self.model.dtype),
                             image_grid_thw=inputs["image_grid_thw"], use_cache=False)
        return out.logits[:, -1]

    # --- the floor ------------------------------------------------------------------------------ #

    @torch.no_grad()
    def approx_only_forward(self, inputs: Dict[str, Any], px_base: torch.Tensor) -> torch.Tensor:
        with self._arrival(0), self._stage("floor"):
            out = self.model(input_ids=inputs["input_ids"],
                             pixel_values=px_base.to(self.model.dtype),
                             image_grid_thw=inputs["image_grid_thw"], use_cache=False)
        return out.logits[:, -1]

    # --- the arm -------------------------------------------------------------------------------- #

    @torch.no_grad()
    def streaming_forward(self, inputs: Dict[str, Any], px_base: torch.Tensor,
                          groups: int, keep: float = 1.0) -> Tuple[torch.Tensor, Any, Dict[str, Any]]:
        """Progressive arrival: vision approximates-then-corrects per band, the LLM streams.
        Same contract as Qwen35Axis.streaming_forward: groups=1, keep=1.0 must reproduce
        `full_forward` in exact arithmetic (the identity gate)."""
        from transformers.cache_utils import DynamicCache

        ids = inputs["input_ids"]
        grid = inputs["image_grid_thw"]
        px_full = inputs["pixel_values"].to(self.model.dtype)
        px_base = px_base.to(device=px_full.device, dtype=px_full.dtype)
        if px_base.shape != px_full.shape:
            raise ValueError(f"base/full pixel grids differ: {tuple(px_base.shape)} vs "
                             f"{tuple(px_full.shape)} -- degrade content, never geometry")

        lo, n_tok = self._image_token_run(ids)
        seq = int(ids.shape[1])
        unit = self.tower.spatial_merge_unit

        cache: Dict[str, Any] = {}
        with self._arrival(0):
            with self._stage("prepare"):
                ctx_full = self.tower.prepare_full_tokens(px_full, grid)
                ctx_base = self.tower.prepare_full_tokens(px_base, grid)
            n_rows = ctx_full["seq_len"]
            n_groups_total = n_rows // unit
            if n_tok != n_groups_total:
                raise ValueError(f"{n_tok} image tokens vs {n_groups_total} shuffle groups")
            with self._stage("vision_base"):
                x_base_out, cache = self.tower.approx_forward(
                    ctx_base["hidden_states"], 0, len(self.tower.blocks), ctx_base, cache, "v",
                    collect_attn_mean=(keep < 1.0))
            if keep < 1.0:
                cache = self.tower.finalize_attn_layermean(cache, "v", len(self.tower.blocks))

        emb_all = self._embed_ids(ids)
        bands = self._bands(groups, n_groups_total)
        stats = {"prefill_tokens": 0, "corrected_groups": 0, "decode_start_pos": seq}
        pos_1d = torch.arange(seq, device=ids.device).unsqueeze(0)   # (1, T)

        if keep < 1.0:
            # Per-group score: pixel residual energy x received attention, both pooled through the
            # SHUFFLE gather (a group's rows are not contiguous in raw order).
            shuffle_index = ctx_full["shuffle_index"]
            resid = (px_full.float() - px_base.float()).pow(2).mean(dim=-1)     # [n_rows], natural
            energy = resid[shuffle_index].reshape(n_groups_total, unit).mean(dim=1)
            col_perm = cache["v_attn_layermean"]                                # [n_rows], permuted
            attn_nat = col_perm[ctx_full["inv_window_index"]]
            attn = attn_nat[shuffle_index].reshape(n_groups_total, unit).mean(dim=1).to(energy.device)
            score = ((energy / energy.mean().clamp_min(1e-12))
                     * (attn / attn.mean().clamp_min(1e-12)))
            n_sel = max(1, int(round(keep * n_groups_total)))
            quota = [n_sel // groups + (1 if r_ < n_sel % groups else 0) for r_ in range(groups)]
            selected = torch.zeros(n_groups_total, dtype=torch.bool, device=score.device)

        kv = DynamicCache(config=self.cfg.text_config)
        pos_done = 0
        last_logits = None

        def prefill(end: int):
            nonlocal pos_done, last_logits
            if end <= pos_done:
                return
            out = self.model(inputs_embeds=emb_all[:, pos_done:end], past_key_values=kv,
                             position_ids=pos_1d[:, pos_done:end], use_cache=True)
            last_logits = out.logits[:, -1]
            stats["prefill_tokens"] += end - pos_done
            pos_done = end

        arrived_rows = torch.zeros(n_rows, dtype=torch.bool, device=px_full.device)
        last_arrival = 0
        for r, (g0, g1) in enumerate(bands):
            if g1 <= g0:
                continue
            last_arrival = r + 1
            with self._arrival(last_arrival):
                # Arrival is in LLM-token (shuffle-block) order: mark the band's raw rows.
                band_groups = torch.arange(g0, g1, device=ctx_full["shuffle_index"].device)
                band_rows_nat = (ctx_full["shuffle_index"].reshape(-1, unit)[band_groups]).reshape(-1)
                arrived_rows[band_rows_nat] = True
                # Streams live in PERMUTED order; arrived-mask must be permuted the same way.
                arrived_perm = arrived_rows[ctx_full["window_index"]]
                stream = torch.where(arrived_perm.unsqueeze(-1),
                                     ctx_full["hidden_states"], ctx_base["hidden_states"])
                if keep < 1.0:
                    band_mask = torch.zeros(n_groups_total, dtype=torch.bool, device=score.device)
                    band_mask[g0:g1] = True
                    cand = band_mask & ~selected
                    q = min(quota[r], int(cand.sum()))
                    if q > 0:
                        group_idx = score.masked_fill(~cand, float("-inf")).topk(q).indices.sort().values
                        selected[group_idx] = True
                    else:
                        group_idx = torch.empty(0, dtype=torch.long, device=score.device)
                else:
                    group_idx = band_groups
                if group_idx.numel():
                    with self._stage("vision_correct"):
                        x_v, cache = self.tower.correct_forward(stream, group_idx, 0,
                                                                len(self.tower.blocks), ctx_full,
                                                                cache, "v")
                stats["corrected_groups"] += int(group_idx.numel())
                with self._stage("merge"):
                    if keep < 1.0:
                        # Uncorrected groups take the PURE approx output (gemma3 convention).
                        row_mask_perm = torch.zeros(n_rows, dtype=torch.bool, device=px_full.device)
                        if group_idx.numel():
                            row_mask_perm[self.tower.group_rows_permuted(group_idx, ctx_full)] = True
                        src = (torch.where(row_mask_perm.unsqueeze(-1), x_v, x_base_out)
                               if group_idx.numel() else x_base_out)
                        merged = self.tower.merge_groups(src, ctx_full, g0, g1)
                    else:
                        merged = self.tower.merge_groups(x_v, ctx_full, g0, g1)
                emb_all[:, lo + g0:lo + g1] = merged.unsqueeze(0).to(emb_all.dtype)
                with self._stage("llm_prefill"):
                    prefill(lo + g1)

        with self._arrival(last_arrival), self._stage("llm_prefill"):
            prefill(seq)
        stats["image_embeds"] = emb_all[:, lo:lo + n_groups_total].float()
        return last_logits, kv, stats
