"""
unified.py

The Qwen3.5-MoE (35B-A3B) progressive-arrival axis: vision tower + MoE decoder as one in-process
object, following `appcorr/models/ov2/unified.py` (whose streaming arm this reuses almost verbatim)
and `appcorr/models/gemma3/unified.py` (whose FLOP-scope conventions it follows).

**The arm this model gets, and why it is not OV2's menu.** OV2 carries both `interleaved_forward`
(approx-then-correct on BOTH halves) and `streaming_forward`, and streaming won (ChartQA 85.0 vs
81.0). Qwen3.5 does not get the choice: 30 of its 40 decoder layers are recurrent
(GatedDeltaNet), so an LLM-side approx-then-correct is not merely worse, it is ill-defined --
correcting token i rewrites the state every later token consumes (see `llm/streaming.py`). So the
LLM half streams, full stop, and only the vision half runs approximate-then-correct:

    base image     -> full 27-layer vision approx, caching K/V for every patch row
    band r arrives -> vision-correct band r's merge groups to full depth against the stored K/V
                      (which already carries earlier bands' corrections), re-merge that band,
                      prefill the LLM for exactly that band's token positions
    trailing text  -> prefilled last, against fully-arrived state, same arrival as the final band

What is given up is the same thing OV2's streaming gives up, stated plainly: vision attention is
bidirectional, so band r's features would keep improving as later bands land -- but band r is
prefilled and consumed before that. The LLM sees a stale view of every band but the last. The
degenerate case g=1 has no staleness at all (one band, corrected after everything arrived), which
is what makes `streaming_forward(groups=1)` an exact identity against `full_forward` -- the gate
this file ships with.

**Chunk contiguity is load-bearing.** Bands must be contiguous runs of image-token positions
(sequential grouping), because an LLM chunk is appended to a cache. This is the documented
constraint from the LLM-interleaved work, and it is asserted here, not assumed.
"""

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .vision.backbone import ApproxCorrectQwen35VisionTower

MODEL_ID_35B = "Qwen/Qwen3.5-35B-A3B"
# The FP8 variant is the only 122B that fits one 183GB device (bf16 would need ~244GB). Same
# vision tower as the 35B (verified identical configs), same qwen3_5_moe architecture -- the
# tower's unwindowed assert and the per-checkpoint gate are what stand between "same in the
# config" and "same in the shipped weights".
MODEL_ID_122B_FP8 = "Qwen/Qwen3.5-122B-A10B-FP8"


class Qwen35Axis(nn.Module):
    def __init__(self, model: nn.Module, processor: Any, flop_counter: Optional[Any] = None):
        super().__init__()
        self.model = model            # Qwen3_5MoeForConditionalGeneration
        self.processor = processor
        self.flops = flop_counter
        self.tower = ApproxCorrectQwen35VisionTower(model.model.visual)
        self.lm = model.model.language_model
        self.cfg = model.config
        self.image_token_id = model.config.image_token_id

    # --- flop scopes (same shape as gemma3/ov2) ------------------------------------------------- #

    def _arrival(self, index: int):
        return self.flops.arrival(index) if self.flops is not None else nullcontext()

    def _stage(self, name: str):
        return self.flops.stage(name) if self.flops is not None else nullcontext()

    # --- shared prep ---------------------------------------------------------------------------- #

    @torch.no_grad()
    def build_inputs(self, image, question: str, think: bool = False) -> Dict[str, Any]:
        """Chat-template + pixel preprocessing for one (image, question) request.

        `think` defaults OFF: Qwen3.5's template opens a `<think>` block when thinking is enabled,
        and a short greedy decode then spends its whole budget on reasoning preamble without ever
        reaching the answer -- measured as the 35B scoring 18% on RealWorldQA MCQ, which is a
        truncated-thought artifact, not a model property. Short-answer evals score the ANSWER, so
        thinking stays off; pass think=True only from a driver that decodes past the block.
        """
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": question}]}]
        return self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=think,
        )

    def _image_token_run(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """(start, count) of the single contiguous image-token run. Raises on anything else --
        multiple images or a fragmented run would make a band not be one prefill chunk."""
        pos = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            raise ValueError("no image tokens in input_ids")
        if not bool((pos[1:] - pos[:-1] == 1).all()):
            raise ValueError("image tokens are not one contiguous run; streaming bands need one")
        return int(pos[0]), int(pos.numel())

    def _bands(self, groups: int, n_merge_groups: int) -> List[Tuple[int, int]]:
        """Split [0, n_merge_groups) into `groups` contiguous bands (sequential grouping)."""
        edges = [round(k * n_merge_groups / groups) for k in range(groups + 1)]
        return [(a, b) for a, b in zip(edges[:-1], edges[1:])]

    # --- the reference -------------------------------------------------------------------------- #

    @torch.no_grad()
    def full_forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Stock forward, final-position logits. The ceiling and the gates' reference."""
        with self._arrival(0), self._stage("full"):
            out = self.model(input_ids=inputs["input_ids"],
                             pixel_values=inputs["pixel_values"].to(self.model.dtype),
                             image_grid_thw=inputs["image_grid_thw"],
                             mm_token_type_ids=inputs["mm_token_type_ids"], use_cache=False)
        return out.logits[:, -1]

    # --- the arm -------------------------------------------------------------------------------- #

    @torch.no_grad()
    def streaming_forward(self, inputs: Dict[str, Any], px_base: torch.Tensor,
                          groups: int, keep: float = 1.0) -> Tuple[torch.Tensor, Any, Dict[str, Any]]:
        """Progressive arrival: vision approximates-then-corrects per band, the LLM streams.

        Args:
            inputs: `build_inputs` output built with the FULL-resolution image (its pixel_values
                are the ground truth the bands converge to).
            px_base: pixel_values of the DEGRADED base image, same grid (the transmission's level-2
                base). Same shape as inputs["pixel_values"].
            groups: arrival rounds. groups=1 must reproduce `full_forward` exactly (in exact
                arithmetic): one band corrected after everything arrived = no staleness anywhere.
            keep: fraction of image tokens corrected in total (the standard 0.25/0.50 arms;
                1.0 = the original streaming arm and the identity-gate case). Band r selects its
                quota among arrived-and-uncorrected tokens by residual energy x received
                attention; the attention term rides the base approx pass this arm already runs
                (full depth -- the base approx is not frontier-chunked, so no extra pass exists to
                duplicate). Unselected tokens enter the LLM at their approximate reconstruction
                and, this being streaming, are never revisited -- that permanence is the knob's
                cost, and what the accuracy arms price.

        Returns (final_position_logits, kv_cache, stats).
        """
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

        # Arrival 0: everything that needs only the base image. Both prepares are grid-shape-exact;
        # the base approx pass gives every patch row a value and every layer a K/V.
        cache: Dict[str, Any] = {}
        with self._arrival(0):
            with self._stage("prepare"):
                ctx_full = self.tower.prepare_full_tokens(px_full, grid)
                ctx_base = self.tower.prepare_full_tokens(px_base, grid)
            n_rows = ctx_full["seq_len"]
            n_groups_total = n_rows // unit
            if n_tok != n_groups_total:
                raise ValueError(f"{n_tok} image tokens vs {n_groups_total} merge groups")
            with self._stage("vision_base"):
                x_base_out, cache = self.tower.approx_forward(
                    ctx_base["hidden_states"], 0, len(self.tower.blocks), ctx_base, cache, "v",
                    collect_attn_mean=(keep < 1.0))
            if keep < 1.0:
                cache = self.tower.finalize_attn_layermean(cache, "v", len(self.tower.blocks))

        emb_all = self.lm.embed_tokens(ids)
        bands = self._bands(groups, n_groups_total)
        stats = {"prefill_tokens": 0, "corrected_groups": 0}  # decode_start_pos added below

        # M-RoPE positions, computed ONCE for the whole request and sliced per chunk. This model
        # gives image tokens interleaved 3D (t,h,w) rotary positions via `get_rope_index`; the
        # model's own fallback when `position_ids is None` (and any hand-rolled `arange`) replicates
        # a 1D counter across all four axes, silently destroying the image grid -- the exact
        # M-RoPE trap already hit on Qwen2.5-VL, where it cost days because BOTH arms of an A/B
        # shared the wrong positions and agreed with each other. `mm_token_type_ids` marks the
        # image run, same contract as 2.5.
        # Shape is (3, B, T) -- the t/h/w axes only. That is not a truncation of anything: stock's
        # own forward passes exactly this tensor to the text model, whose `shape[0] == 4` branch
        # (which would split off a text row) therefore never fires on the mrope path. Text tokens
        # carry t == h == w, so the text axis is redundant with any of the three.
        mm_ttids = (ids == self.image_token_id).long()
        pos_3d, rope_deltas = self.model.model.get_rope_index(ids, mm_ttids, image_grid_thw=grid)
        if pos_3d.shape[0] != 3 or pos_3d.shape[-1] != seq:
            raise ValueError(f"get_rope_index returned {tuple(pos_3d.shape)}, expected (3, 1, {seq})")
        # Where a decode loop on top of the returned cache must continue from: stock advances all
        # three axes together for generated (text) tokens.
        stats_decode_pos = int(pos_3d.max().item()) + 1

        if keep < 1.0:
            # Per-merge-group score. Energy is pixel-level (the client hint in deployment): mean
            # squared residual between full and base patch rows, pooled to merge groups.
            resid = (px_full.float() - px_base.float()).pow(2).mean(dim=-1)      # [n_rows]
            unit_ = self.tower.spatial_merge_unit
            energy = resid.reshape(n_groups_total, unit_).mean(dim=1)
            attn = cache["v_attn_layermean"].to(energy.device)
            attn = attn.reshape(n_groups_total, unit_).mean(dim=1)
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
                             position_ids=pos_3d[:, :, pos_done:end], use_cache=True)
            last_logits = out.logits[:, -1]
            stats["prefill_tokens"] += end - pos_done
            pos_done = end

        # Rows arrived so far -- the residual-stream restart mixes full rows (arrived) with base
        # rows (not yet), which is the in-process equivalent of the executor path's "reconstructed
        # canvas": the stream the correction restarts from is exactly what has been received.
        arrived_rows = torch.zeros(n_rows, dtype=torch.bool, device=px_full.device)
        last_arrival = 0
        for r, (g0, g1) in enumerate(bands):
            if g1 <= g0:
                continue
            last_arrival = r + 1
            with self._arrival(last_arrival):
                arrived_rows[g0 * unit:g1 * unit] = True
                stream = torch.where(arrived_rows.unsqueeze(-1),
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
                    group_idx = torch.arange(g0, g1, device=px_full.device)
                if group_idx.numel():
                    with self._stage("vision_correct"):
                        x_v, cache = self.tower.correct_forward(stream, group_idx, 0,
                                                                len(self.tower.blocks), ctx_full,
                                                                cache, "v")
                stats["corrected_groups"] += int(group_idx.numel()) if keep < 1.0 else (g1 - g0)
                # Merge ONLY this band. The merger is per-merge-group (norm -> reshape(unit) ->
                # MLP), so slicing at group granularity is exact. Under keep<1, UNCORRECTED rows
                # take the PURE approx output -- the same convention gemma3's progressive walk
                # uses (mixed = corrected rows from the walk, everything else feats_appr), not the
                # stream+increment reconstruction, which mixes refined layer-0 with degraded
                # increments (the self-inconsistent combination the CLIP memo measured below floor).
                with self._stage("merge"):
                    if keep < 1.0:
                        row_mask = torch.zeros(n_rows, dtype=torch.bool, device=px_full.device)
                        if group_idx.numel():
                            rows_sel = (group_idx.unsqueeze(1) * unit
                                        + torch.arange(unit, device=group_idx.device)).flatten()
                            row_mask[rows_sel] = True
                        src = torch.where(row_mask.unsqueeze(-1), x_v, x_base_out)                             if group_idx.numel() else x_base_out
                        band_rows = src[g0 * unit:g1 * unit]
                    else:
                        band_rows = x_v[g0 * unit:g1 * unit]
                    merged = self.tower.merger(band_rows)
                emb_all[:, lo + g0:lo + g1] = merged.unsqueeze(0).to(emb_all.dtype)
                with self._stage("llm_prefill"):
                    prefill(lo + g1)      # on r=0 this also carries the leading text

        # Trailing text (question + generation prompt) waits on the last band -- same arrival,
        # charging it later would invent an arrival the transmission never had.
        with self._arrival(last_arrival), self._stage("llm_prefill"):
            prefill(seq)
        stats["decode_start_pos"] = stats_decode_pos
        # The image-row embeddings the LLM actually consumed, for feature-space gating. Task
        # metrics are not monotone in fidelity (the interleaved contract says this in as many
        # words), and this model's first generated token is CoT boilerplate that ignores the image
        # entirely -- measured TV(floor, stock) = 0.0005 at that position, i.e. no logit-level gate
        # can see the vision mechanism at all. The embeddings can.
        stats["image_embeds"] = emb_all[:, lo:lo + n_groups_total].float()
        return last_logits, kv, stats

    # --- the floor ------------------------------------------------------------------------------ #

    @torch.no_grad()
    def approx_only_forward(self, inputs: Dict[str, Any], px_base: torch.Tensor) -> torch.Tensor:
        """The floor: the degraded base image through the STOCK path, one-shot. 100% critical."""
        with self._arrival(0), self._stage("floor"):
            out = self.model(input_ids=inputs["input_ids"],
                             pixel_values=px_base.to(self.model.dtype),
                             image_grid_thw=inputs["image_grid_thw"],
                             mm_token_type_ids=inputs["mm_token_type_ids"], use_cache=False)
        return out.logits[:, -1]
