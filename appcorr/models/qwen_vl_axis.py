"""
qwen_vl_axis.py

The progressive-arrival axis shared by the Qwen2-VL family (Qwen2.5-VL, Qwen3.5): vision tower
approximates-then-corrects per band, the LLM streams (chunked prefill per band). The streaming
loop here is `appcorr/models/qwen35/unified.py`'s, lifted verbatim once a second model needed it;
the per-model subclasses (`Qwen35Axis`, `Qwen25VLAxis`) only supply the tower and the four places
where the two vision forks differ:

    _approx_base      the tower's full-depth base approx (attention-mean collection kwarg differs)
    _attn_layermean   where the received-attention mean lives afterwards (key name differs)
    _rows_of_groups   merge-group index -> the tower's ROW indices for that group. Qwen3.5's rows
                      are in natural order (identity); Qwen2.5-VL's tower permutes rows by window
                      (`window_index`) before its block loop, so a group's rows are wherever the
                      permutation put them. Everything the loop touches at row granularity (the
                      arrived-rows mask, the band the merger sees, the keep<1 row mask, the
                      attention pooled to groups) goes through this one mapping.
    build_inputs      chat-template kwargs (Qwen3.5 has a thinking switch)

The LLM side is identical: same `get_rope_index` contract (3, B, T), same `inputs_embeds +
position_ids + DynamicCache` chunked prefill, same `mm_token_type_ids` marking of the image run.

**Two LLM backends, one vision path.** `streaming_forward(..., sink=None)` runs the LLM in
process (HF, the path every table number so far went through). With `sink=` (a
`appcorr.vllm_stream.bridge.StreamSink`), the same loop sends each chunk -- rows of `emb_all`
plus their M-RoPE positions -- to the vLLM streaming server instead of calling the HF model, and
returns no logits: the answer comes back from the server (`sink.result()`), decoded by vLLM's
greedy loop. The vision work, the band schedule and the chunk boundaries are byte-for-byte the
same in both modes; only who consumes the chunks changes. `oneshot_embeds` gives the floor and
ceiling arms the same one-request path (stock tower, one chunk), so all arms of a vLLM campaign
decode through the identical engine -- the decode-mechanism-consistency rule the HF driver
enforces with its shared greedy loop, restated for the two-process form.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# PyTorch's cuDNN SDPA backend returns a WRONG output for some attention heads on the B200 in
# this env (torch 2.12.1+cu130, cuDNN 9.20): found 2026-09-07 through the vLLM bridge gate --
# the HF twin decoded "ribbeded" / "dotteded" where vLLM, eager, the flash and the math
# backends all decode "ribbed" / "dotted". On the captured decode-step call (q [1,28,1,128],
# kv [1,4,382,128], real Qwen2.5-VL-7B activations) one head is off by 1.56 (flash: 4e-3);
# random tensors of the same shape do not trigger it, so it is data-dependent. Eligibility is
# head_dim <= 128 (the backend refuses head_dim 256, so Qwen3.5 / Gemma never hit it); the
# error is confined to q_len=1 decode steps in every case measured (prefill logits within bf16
# noise). Repro tensors: analysis/results/vllm_stream/sdpa_cudnn_repro.pt. Every HF-side
# consumer imports this module, so the switch lives here.
torch.backends.cuda.enable_cudnn_sdp(False)


class QwenVLStreamingAxis(nn.Module):
    def __init__(self, model: nn.Module, processor: Any, flop_counter: Optional[Any] = None):
        super().__init__()
        self.model = model
        self.processor = processor
        self.flops = flop_counter
        self.tower = self._make_tower(model)
        self.lm = model.model.language_model
        self.cfg = model.config
        self.image_token_id = model.config.image_token_id

    # --- per-model hooks (subclasses) ----------------------------------------------------------- #

    def _make_tower(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError

    def _approx_base(self, ctx_base: Dict[str, Any], cache: Dict[str, Any],
                     collect_attn: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Full-depth approx of the base image; with `collect_attn`, also leave the per-row
        received-attention mean where `_attn_layermean` finds it."""
        raise NotImplementedError

    def _attn_layermean(self, cache: Dict[str, Any]) -> torch.Tensor:
        """[n_rows] received-attention mean, in the tower's row order."""
        raise NotImplementedError

    def _rows_of_groups(self, ctx: Dict[str, Any], group_idx: torch.Tensor) -> torch.Tensor:
        """[G * unit] row indices (tower row order) of the given ORIGINAL merge-group indices,
        group-major, so `x[rows].reshape(G, unit, D)` is group g's rows in order."""
        unit = self.tower.spatial_merge_unit
        return (group_idx.unsqueeze(1) * unit
                + torch.arange(unit, device=group_idx.device)).flatten()

    # --- flop scopes (same shape as gemma3/ov2) ------------------------------------------------- #

    def _arrival(self, index: int):
        return self.flops.arrival(index) if self.flops is not None else nullcontext()

    def _stage(self, name: str):
        return self.flops.stage(name) if self.flops is not None else nullcontext()

    # --- shared prep ---------------------------------------------------------------------------- #

    def _chat_template_kwargs(self, **kw) -> Dict[str, Any]:
        return {}

    @torch.no_grad()
    def build_inputs(self, image, question: str, **kw) -> Dict[str, Any]:
        """Chat-template + pixel preprocessing for one (image, question) request. Guarantees
        `mm_token_type_ids` (1 on the image run) is present -- the M-RoPE contract every path here
        relies on -- computing it from `input_ids` if the processor did not."""
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": question}]}]
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", **self._chat_template_kwargs(**kw),
        )
        if "mm_token_type_ids" not in inputs:
            inputs["mm_token_type_ids"] = (inputs["input_ids"] == self.image_token_id).long()
        return inputs

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

    def _positions(self, inputs: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        """M-RoPE positions (3, 1, T) for the whole request + the rope delta (an int: what the
        model adds to a plain counter for every generated token). Computed ONCE per request and
        sliced per chunk. The model's own fallback when `position_ids is None` (and any hand-rolled
        `arange`) replicates a 1D counter across the axes, silently destroying the image grid -- the
        M-RoPE trap hit on Qwen2.5-VL, where BOTH arms of an A/B shared the wrong positions and
        agreed with each other. Shape (3, B, T) is the t/h/w axes only, exactly what stock's own
        forward hands its text model."""
        ids = inputs["input_ids"]
        mm_ttids = inputs.get("mm_token_type_ids")
        if mm_ttids is None:
            mm_ttids = (ids == self.image_token_id).long()
        pos_3d, rope_deltas = self.model.model.get_rope_index(
            ids, mm_ttids, image_grid_thw=inputs["image_grid_thw"])
        seq = int(ids.shape[1])
        if pos_3d.shape[0] != 3 or pos_3d.shape[-1] != seq:
            raise ValueError(f"get_rope_index returned {tuple(pos_3d.shape)}, expected (3, 1, {seq})")
        return pos_3d, int(rope_deltas.flatten()[0].item())

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
                          groups: int, keep: float = 1.0,
                          sink: Optional[Any] = None) -> Tuple[Optional[torch.Tensor], Any, Dict[str, Any]]:
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
            sink: optional `StreamSink`. When given, every chunk goes to the vLLM server instead of
                the in-process HF model and the return is `(None, None, stats)`; read the answer
                from `sink.result()`.

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
                x_base_out, cache = self._approx_base(ctx_base, cache, collect_attn=(keep < 1.0))

        emb_all = self.lm.embed_tokens(ids)
        bands = self._bands(groups, n_groups_total)
        stats = {"prefill_tokens": 0, "corrected_groups": 0, "chunks": []}  # decode_start_pos added below

        pos_3d, rope_delta = self._positions(inputs)
        # Where a decode loop on top of the returned cache must continue from: stock advances all
        # three axes together for generated (text) tokens.
        stats_decode_pos = int(pos_3d.max().item()) + 1
        all_groups = torch.arange(n_groups_total, device=px_full.device)
        rows_all = self._rows_of_groups(ctx_full, all_groups)   # group-major row order

        if keep < 1.0:
            # Per-merge-group score. Energy is pixel-level (the client hint in deployment): mean
            # squared residual between full and base patch rows, pooled to merge groups.
            # pixel_values rows are in patch_embed's native (group) order; the attention mean is
            # in the tower's row order and is gathered into group order through `rows_all`.
            resid = (px_full.float() - px_base.float()).pow(2).mean(dim=-1)      # [n_rows]
            energy = resid.reshape(n_groups_total, unit).mean(dim=1)
            attn = self._attn_layermean(cache).to(energy.device)
            attn = attn[rows_all.to(attn.device)].reshape(n_groups_total, unit).mean(dim=1)
            score = ((energy / energy.mean().clamp_min(1e-12))
                     * (attn / attn.mean().clamp_min(1e-12)))
            n_sel = max(1, int(round(keep * n_groups_total)))
            quota = [n_sel // groups + (1 if r_ < n_sel % groups else 0) for r_ in range(groups)]
            selected = torch.zeros(n_groups_total, dtype=torch.bool, device=score.device)

        kv = None if sink is not None else DynamicCache(config=self.cfg.text_config)
        pos_done = 0
        last_logits = None

        def prefill(end: int):
            nonlocal pos_done, last_logits
            if end <= pos_done:
                return
            if sink is not None:
                # The chunk leaves the process: rows of the (partially corrected) embedding
                # sequence plus their M-RoPE positions; `final` closes the prompt on the server.
                sink.push(emb_all[0, pos_done:end], pos_3d[:, 0, pos_done:end], rope_delta,
                          final=(end == seq))
            else:
                out = self.model(inputs_embeds=emb_all[:, pos_done:end], past_key_values=kv,
                                 position_ids=pos_3d[:, :, pos_done:end], use_cache=True)
                last_logits = out.logits[:, -1]
            stats["chunks"].append((pos_done, end))
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
                band_groups = torch.arange(g0, g1, device=px_full.device)
                band_rows_idx = self._rows_of_groups(ctx_full, band_groups)
                arrived_rows[band_rows_idx] = True
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
                    group_idx = band_groups
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
                            row_mask[self._rows_of_groups(ctx_full, group_idx.to(px_full.device))] = True
                        src = torch.where(row_mask.unsqueeze(-1), x_v, x_base_out) \
                            if group_idx.numel() else x_base_out
                        band_rows = src[band_rows_idx]
                    else:
                        band_rows = x_v[band_rows_idx]
                    merged = self.tower.merger(band_rows)
                emb_all[:, lo + g0:lo + g1] = merged.unsqueeze(0).to(emb_all.dtype)
                with self._stage("llm_prefill"):
                    prefill(lo + g1)      # on r=0 this also carries the leading text

        # Trailing text (question + generation prompt) waits on the last band -- same arrival,
        # charging it later would invent an arrival the transmission never had.
        with self._arrival(last_arrival), self._stage("llm_prefill"):
            prefill(seq)
        stats["decode_start_pos"] = stats_decode_pos
        stats["rope_delta"] = rope_delta
        # The image-row embeddings the LLM actually consumed, for feature-space gating. Task
        # metrics are not monotone in fidelity (the interleaved contract says this in as many
        # words), and Qwen3.5's first generated token is CoT boilerplate that ignores the image
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

    # --- one-shot prompt embeddings (floor / ceiling through the vLLM server) ------------------- #

    @torch.no_grad()
    def oneshot_embeds(self, inputs: Dict[str, Any], pixel_values: torch.Tensor,
                       stage: str = "full") -> Tuple[torch.Tensor, torch.Tensor, int]:
        """The stock path's prompt embeddings for ONE image: text rows from `embed_tokens`, image
        rows from the STOCK vision tower on `pixel_values` (the full image for the ceiling, the
        degraded base for the floor), spliced at the image run -- the same `inputs_embeds` stock's
        forward builds internally before its first decoder layer. Returns (embeds [T, D],
        mrope positions [3, T], rope_delta), i.e. one chunk for `StreamSink.push(final=True)`."""
        ids = inputs["input_ids"]
        lo, n_tok = self._image_token_run(ids)
        with self._arrival(0), self._stage(stage):
            feats = self.model.model.get_image_features(
                pixel_values.to(self.model.dtype), inputs["image_grid_thw"])
            feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
            if isinstance(feats, (list, tuple)):
                feats = torch.cat(list(feats), dim=0)
            if feats.shape[0] != n_tok:
                raise ValueError(f"tower produced {feats.shape[0]} rows for {n_tok} image tokens")
            emb = self.lm.embed_tokens(ids)[0]
            emb[lo:lo + n_tok] = feats.to(emb.dtype)
        pos_3d, rope_delta = self._positions(inputs)
        return emb, pos_3d[:, 0], rope_delta
