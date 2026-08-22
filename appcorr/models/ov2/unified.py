"""LLaVA-OneVision-2 as ONE approx/correct axis: 24 encoder layers then 36 Qwen3 decoder layers.

Why unify. The cost split here is not Gemma 3's, and it is not close. Both halves scale with the
same token count -- the merger folds exactly 4 patches into 1 LLM token at every resolution -- so
the ratio is a constant that falls out of the shapes:

    vision   24L x (4 x N) patches x 1024^2
    LLM      36L x  N      tokens  x 4096^2
    LLM/vision = (36 x 16) / (24 x 4) = 6.0x

Six to one, at 448x448 and at 1988x1988 alike. Approx/correcting the vision tower alone would
overlap 14% of the forward. Gemma 3's equivalent ratio was 1.8x, so the unified axis is *more*
load-bearing here than on any fork so far, not less.

Three things are structurally easier than Gemma 3 and one is harder.

**Easier: the patch->token map is contiguous.** The Qwen2VL-style image processor emits patches
already in 2x2 block order, and the merger is `ln_q(x).view(-1, 4*C)` -- four CONSECUTIVE patches
per token. Gemma 3 needed a 4x4 spatial-block map because its AvgPool2d ran over the patch grid;
here `arange(N) // 4` is exact by construction, and the merger's own reshape is the proof.

**Easier: one rope, one mask.** Qwen3 is causal everywhere with no sliding layers, and OV2 uses
plain 1D position_ids (no mrope). Gemma 3's per-layer-type cos/sin and
`{full_attention, sliding_attention}` mask dict both collapse to a single value.

**Easier: no token_type_ids.** Image tokens are causal like everything else, so there is no
bidirectional block to build a mask for.

**Harder: nothing is a constant.** There is no fixed canvas. `n_patch`, `n_img_tok` and `seq_len`
are all functions of the image (measured: 256 to 1768+ tokens), so every quantity Gemma 3 could read
off the config -- pooling geometry, stage costs, group boundaries -- has to be derived per sample
from `image_grid_thw` / `patch_positions`. Anything cached across samples is a bug waiting for the
second image size.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .llm.decoder_layer import ApproxCorrectQwen3DecoderLayer
from .vision.block import ApproxCorrectOneVisionLayer


class OV2UnifiedAxis(nn.Module):
    """Wraps a stock `LlavaOnevision2Model` as one 60-stage approx/correct axis."""

    def __init__(self, model: nn.Module, flop_counter=None) -> None:
        super().__init__()
        # Optional `appcorr.flops.FlopCounter`. The axis only declares WHEN each piece of work
        # became possible; the counter decides what that makes critical. Left None -- the default,
        # and what every existing driver passes -- the arrival/stage scopes below are `nullcontext`
        # and cost one attribute test per scope, not per operation.
        self.flops = flop_counter
        self.model = model                                   # LlavaOnevision2Model
        vt = model.visual
        self.vt = vt
        self.vision_layers = nn.ModuleList(
            ApproxCorrectOneVisionLayer.from_stock(l) for l in vt.encoder.layers)
        lm = model.language_model
        self.lm = lm
        self.llm_layers = nn.ModuleList(
            ApproxCorrectQwen3DecoderLayer.from_stock(l) for l in lm.layers)
        self.cfg = model.config
        self.merge_size = int(getattr(self.cfg.vision_config, "spatial_merge_size", 2))
        self.merge = self.merge_size ** 2
        # Per-sample geometry, set by `llm_prepare`. Deliberately not cached across samples: the
        # image decides all three and a stale value silently mis-sizes the interleaved schedule.
        self._n_patch: Optional[int] = None
        self._seq_len: Optional[int] = None

    # --- FLOP accounting scopes ------------------------------------------------------------------ #
    # The axis declares WHEN work became possible; `appcorr.flops` turns that into the critical /
    # overlappable split. Arrival 0 is the base image, arrivals 1..g are the detail groups, so the
    # highest index -- and therefore the critical set -- is whatever ran after the last group.
    # `full_forward` opens none, which is what makes the ceiling come out 100% critical.

    def _arrival(self, index: int):
        return self.flops.arrival(index) if self.flops is not None else nullcontext()

    def _stage(self, name: str):
        return self.flops.stage(name) if self.flops is not None else nullcontext()

    # --- geometry ------------------------------------------------------------------------------ #

    @property
    def n_vision(self) -> int:
        return len(self.vision_layers)

    @property
    def n_llm(self) -> int:
        return len(self.llm_layers)

    @property
    def n_stages(self) -> int:
        return self.n_vision + self.n_llm

    def n_tokens(self, n_patch: int) -> int:
        """Patches -> image tokens. Exact, not approximate: the merger reshapes by `4*C`."""
        assert n_patch % self.merge == 0, (
            f"{n_patch} patches is not divisible by the merge factor {self.merge}; the 2x2 block "
            "layout the merger's reshape assumes would be broken")
        return n_patch // self.merge

    def pool_patch_score(self, patch_score: torch.Tensor) -> torch.Tensor:
        """[B, n_patch] -> [B, n_token], by the merger's OWN grouping.

        Four consecutive patches per token, because the image processor hands the tower patches in
        2x2 block order and `LlavaOnevision2VisionPatchMerger.forward` does
        `self.ln_q(x).view(-1, context_dim * 4)`. This is the one place Gemma 3 needed a spatial
        `(r//k)*tps + (c//k)` map; here the reshape IS the map, so a mismatch is impossible rather
        than merely unlikely.
        """
        b, n_patch = patch_score.shape
        return patch_score.reshape(b, self.n_tokens(n_patch), self.merge).mean(dim=-1)

    def token_mask_to_patch_mask(self, token_sel: torch.Tensor) -> torch.Tensor:
        """[B, n_token] -> [B, n_patch]: all 4 patches of every selected token."""
        return token_sel.repeat_interleave(self.merge, dim=1)

    def patch_mask_any_to_token(self, patch_mask: torch.Tensor) -> torch.Tensor:
        """[B, n_patch] -> [B, n_token]: token selected if ANY of its 4 patches was."""
        b, n_patch = patch_mask.shape
        return patch_mask.reshape(b, self.n_tokens(n_patch), self.merge).any(dim=-1)

    # --- vision half --------------------------------------------------------------------------- #

    @torch.no_grad()
    def vision_prepare(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Patch embeddings + `layernorm_pre` -- the tensor the encoder's layer 0 actually sees.

        Skipping `layernorm_pre` leaves every layer on a differently-scaled input and still runs.
        """
        h = self.vt.embeddings(pixel_values)
        if h.dim() == 2:
            h = h.unsqueeze(0)
        return self.vt.layernorm_pre(h)

    @torch.no_grad()
    def rope_freqs(self, patch_positions: torch.Tensor) -> torch.Tensor:
        """Per-patch rotary frequencies, D-wide. [1, n_patch, D].

        `forward_from_positions` returns D/2 and the tower duplicates it to D before use; feeding
        the un-duplicated half rotates only the first half of every head and still produces a
        finite, plausible tensor.
        """
        pp = patch_positions.squeeze(0) if patch_positions.dim() == 3 else patch_positions
        f = self.vt.video_rope.forward_from_positions(pp)
        f = torch.cat([f, f], dim=-1)
        return f.unsqueeze(0) if f.dim() == 2 else f

    @torch.no_grad()
    def _incoming_attention(self, hidden: torch.Tensor, layer_idx: int,
                            freqs: torch.Tensor) -> torch.Tensor:
        """Column mass of one layer's attention, head- and query-averaged. [B, n_patch].

        How much the rest of the image reads FROM each patch -- the term the standard patch score
        multiplies residual energy by. Taken on the approximate pass from the same projections the
        layer is about to use, so it costs one extra QK product and no extra weights.

        The chunk size is derived rather than fixed. Gemma 3 could hardcode 512 because its patch
        count was always 4096; here a 1988x1988 image gives 20164 patches, and a 512-query chunk
        would materialise 512 x 16 x 20164 fp32 = 660 MB. Budgeting elements instead keeps the
        transient bounded at every resolution.

        There is no CLS token on this encoder, so the result already lines up with patch indices.
        """
        layer = self.vision_layers[layer_idx]
        q, k, _ = layer._qkv(layer.layer_norm1(hidden), freqs)
        scale = layer.self_attn.scale
        b, heads, seq, _ = q.shape
        col = torch.zeros(b, seq, device=hidden.device, dtype=torch.float32)
        chunk = int(min(1024, max(64, 3.2e7 // max(1, heads * seq))))
        for s0 in range(0, seq, chunk):
            e0 = min(s0 + chunk, seq)
            w = torch.softmax((q[:, :, s0:e0] @ k.transpose(-1, -2)) * scale, dim=-1)
            col += w.float().sum(dim=2).mean(dim=1)
        return col / seq

    @torch.no_grad()
    def vision_approx(self, hidden, freqs, cache, layers: Optional[Tuple[int, int]] = None,
                      collect_attn: bool = False):
        a, b = (0, self.n_vision) if layers is None else layers
        acc = None
        for i in range(a, b):
            if collect_attn:
                c = self._incoming_attention(hidden, i, freqs)
                acc = c if acc is None else acc + c
            hidden, cache = self.vision_layers[i].approx(hidden, cache, f"v{i}", freqs)
        if collect_attn and acc is not None:
            prev = cache.get("vision_patch_attn_sum")
            cache["vision_patch_attn_sum"] = acc if prev is None else prev + acc
            cache["vision_patch_attn_layers"] = cache.get("vision_patch_attn_layers", 0) + (b - a)
            cache["vision_patch_attn_layermean"] = (cache["vision_patch_attn_sum"]
                                                    / cache["vision_patch_attn_layers"])
        return hidden, cache

    # Ratio band for `_check_entry`, chosen from measurement rather than taste. On three ChartQA
    # samples the LEGITIMATE spread -- the layer-0 stream anywhere between all-approximate and
    # all-full-resolution, which is what a late streaming band hands in -- is 1.03-1.27x. The BUG
    # it exists to catch, a running state fed back into layer 0, is already 2.0-2.4x after a single
    # layer and 70-86x after all 24. 1.6 sits in the empty middle with margin on both sides.
    #
    # The original 1.25x bound was tight enough to reject streaming's last band, where the stream is
    # 100% full-resolution by construction -- a false positive on the arm that needs the guard least.
    _ENTRY_STD_RATIO = 1.6

    @classmethod
    def _check_entry(cls, x, cache, tag, what):
        """The entry tensor of a correction walk must be a layer-0 input, not a running state.

        Checked only at layer 0: from layer 1 on, `correct` legitimately carries a *corrected*
        running state that differs from what approx saw, so a per-layer version of this fires on
        correct behaviour. The mistake worth catching is handing the whole walk a deep hidden state
        -- it runs, returns plausible numbers, and passes every gate that calls the API correctly.
        A Gemma 3 driver did exactly that.
        """
        sig = cache.get(f"{tag}_in_sig")
        if sig is None:
            return
        got = (float(x.float().mean()), float(x.float().std()), tuple(x.shape))
        ratio = abs(got[1]) / max(abs(sig[1]), 1e-9)
        if got[2] != sig[2] or not (1.0 / cls._ENTRY_STD_RATIO <= ratio <= cls._ENTRY_STD_RATIO):
            raise RuntimeError(
                f"{what}_correct got an entry tensor this cache was not built from: approx saw "
                f"mean/std/shape {sig}, got {got} (std ratio {ratio:.2f}, allowed "
                f"{1/cls._ENTRY_STD_RATIO:.2f}-{cls._ENTRY_STD_RATIO:.2f}). Pass the same layer-0 "
                f"input approx received (the mixed stream), not a running state.")

    @torch.no_grad()
    def vision_correct(self, hidden, patch_mask, freqs, cache,
                       layers: Optional[Tuple[int, int]] = None):
        a, b = (0, self.n_vision) if layers is None else layers
        if a == 0:
            self._check_entry(hidden, cache, "v0", "vision")
        for i in range(a, b):
            hidden, cache = self.vision_layers[i].correct(hidden, patch_mask, cache, f"v{i}", freqs)
        return hidden, cache

    @torch.no_grad()
    def project(self, vision_hidden: torch.Tensor,
                patch_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encoder output -> LLM-space image tokens, matching the tower's own tail.

        `layernorm_post` then `merger`. The tower applies post-norm BEFORE the skip_merger branch
        (modeling_llava_onevision2.py:920), so both the fork's unit test and this path norm once.
        """
        h = vision_hidden
        if self.vt.layernorm_post is not None:
            h = self.vt.layernorm_post(h)
        return self.vt.merger(h, patch_positions=patch_positions)

    # --- LLM half ------------------------------------------------------------------------------ #

    @torch.no_grad()
    def llm_prepare(self, input_ids: torch.Tensor, image_features: torch.Tensor):
        """Embed text, splice image features in, and build the rope and causal mask.

        Returns (hidden, ctx). Everything in `ctx` is independent of which tokens get corrected, so
        it is built once and reused by every round.

        Unlike Gemma 3 there is exactly ONE rope and ONE mask: Qwen3 has no sliding layers, and OV2
        drives it with plain 1D `position_ids` (modeling_llava_onevision2.py:1186) rather than
        mrope, despite the Qwen2VL-style vision front end.
        """
        emb = self.lm.get_input_embeddings()(input_ids)
        img_pos = (input_ids[0] == self.cfg.image_token_id).nonzero(as_tuple=True)[0]
        feats = image_features.reshape(-1, image_features.shape[-1])
        assert feats.shape[0] == img_pos.numel(), (
            f"{feats.shape[0]} image features for {img_pos.numel()} image tokens")
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[:, img_pos] = True
        emb = emb.masked_scatter(mask.unsqueeze(-1), feats.to(emb.dtype))

        n = input_ids.shape[1]
        position_ids = torch.arange(n, device=input_ids.device).unsqueeze(0)
        pe = self.lm.rotary_emb(emb, position_ids)
        causal = torch.full((n, n), torch.finfo(emb.dtype).min,
                            device=emb.device, dtype=emb.dtype).triu(1)
        self._seq_len = int(n)
        ctx = {"pe": pe, "mask": causal.view(1, 1, n, n), "position_ids": position_ids,
               "image_positions": img_pos}
        return emb, ctx

    @torch.no_grad()
    def llm_approx(self, hidden, ctx, cache, layers: Optional[Tuple[int, int]] = None):
        a, b = (0, self.n_llm) if layers is None else layers
        for i in range(a, b):
            hidden, cache = self.llm_layers[i].approx(hidden, ctx["pe"], ctx["mask"], cache, f"l{i}")
        return hidden, cache

    @torch.no_grad()
    def llm_correct(self, hidden, token_mask, ctx, cache, layers: Optional[Tuple[int, int]] = None):
        a, b = (0, self.n_llm) if layers is None else layers
        if a == 0:
            self._check_entry(hidden, cache, "l0", "llm")
        for i in range(a, b):
            hidden, cache = self.llm_layers[i].correct(
                hidden, token_mask, ctx["pe"], ctx["mask"], cache, f"l{i}")
        return hidden, cache

    @torch.no_grad()
    def llm_finish(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm.norm(hidden)

    # --- interleaved scheduling ----------------------------------------------------------------- #

    def stage_costs(self, n_patch: int, seq_len: int) -> torch.Tensor:
        """Relative cost of each of the 60 stages, normalised to sum 1.

        Counting stages would be wrong: an encoder layer processes 4N patches at width 1024 and a
        decoder layer ~N tokens at width 4096, so they are not interchangeable units. Cost is taken
        as tokens x width^2 (the projection/MLP term dominates at these shapes) -- the same proxy
        the Gemma 3 axis used, where measured wall clock agreed with it.

        The proxy is weakest at large images, where vision attention is quadratic in 4N and the MLP
        term stops dominating. It is kept because the alternative (a per-sample timing model) would
        make the group boundaries depend on machine load.
        """
        v = self.cfg.vision_config
        cv = float(n_patch) * float(v.hidden_size) ** 2
        t = self.cfg.text_config
        cl = float(seq_len) * float(t.hidden_size) ** 2
        costs = torch.tensor([cv] * self.n_vision + [cl] * self.n_llm, dtype=torch.float64)
        return costs / costs.sum()

    def layer_bounds(self, groups: int, n_patch: int, seq_len: int) -> List[int]:
        """Round boundaries over the 60-stage axis, split by equal COST rather than equal count.

        With the LLM half at 6x the vision half, equal stage counts would put five of every six
        cost units in the last rounds. The last bound is always the full axis.
        """
        if groups <= 1:
            return [self.n_stages]
        cum = torch.cumsum(self.stage_costs(n_patch, seq_len), 0)
        bounds = []
        for r in range(1, groups):
            target = r / groups
            b = int(torch.searchsorted(cum, torch.tensor(target, dtype=torch.float64)).item()) + 1
            bounds.append(max(1, min(b, self.n_stages - 1)))
        bounds = sorted(set(bounds))
        while len(bounds) < groups - 1:                    # keep g distinct rounds
            for cand in range(1, self.n_stages):
                if cand not in bounds:
                    bounds.append(cand)
                    break
            bounds = sorted(set(bounds))
        return bounds + [self.n_stages]

    def spatial_groups(self, selected_patches: torch.Tensor, groups: int,
                       patch_positions: torch.Tensor) -> List[torch.Tensor]:
        """Split the SELECTED patches into `groups` contiguous horizontal bands. [B, n_patch] bool.

        Selection and grouping stay separate: selection is the free top-k by score, and grouping
        only decides the ORDER those already-chosen patches arrive in, which is what interleaving
        schedules.

        Two things about the band index, both specific to this layout.

        It comes from `patch_positions[:, 1]` -- the patch's row in the image grid -- and NOT from
        its index in the sequence. Those differ here: the processor emits patches in 2x2 block
        order (measured: patches 0..3 are rows 0,0,1,1), so `idx // grid_w` is not the row, and
        slicing by it would cut bands that are neither contiguous nor horizontal. Gemma 3's raster
        layout let it use the index.

        And it is the MERGED row, `row // spatial_merge_size`. A band edge on raw patch rows would
        fall inside a 2x2 block and split one token's four patches across two rounds. That token
        would then appear in two rounds' `patch_mask_any_to_token`, be corrected twice, and cost
        more than one-shot for it -- the exact cost inversion contract rule 1 is about, arriving
        through the geometry instead of through the accumulation.
        """
        pp = patch_positions.squeeze(0) if patch_positions.dim() == 3 else patch_positions
        rows = pp[:, 1].to(selected_patches.device) // self.merge_size
        n_rows = int(rows.max().item()) + 1
        edges = torch.linspace(0, n_rows, groups + 1).round().long()
        out = []
        for r in range(groups):
            band = (rows >= edges[r]) & (rows < edges[r + 1])
            out.append(selected_patches & band.unsqueeze(0))
        return out

    # --- interleaved walk ------------------------------------------------------------------------ #

    @torch.no_grad()
    def interleaved_forward(self, px_full, px_approx, patch_positions, input_ids,
                            patch_selection, llm_oneshot, groups):
        """Walk the 60-stage axis in `groups` rounds, correcting one group per round.

        Follows docs/memo/interleaved_correction_contract.md -- read it before touching this. The
        four rules in short, plus the three this axis adds:

        **A round corrects its OWN group** (rule 1), never the accumulated set, which would make the
        last round equal one-shot and invert the cost claim.

        **The input STREAM is cumulative, the RECOMPUTE set is not** (rule 2). `stream` carries
        full-resolution embeddings at every ARRIVED patch, because a corrected patch's layer-0 value
        is its full-resolution embedding; but only this round's group is recomputed.

        **The corrected increment is persisted every round** (rule 3) -- done inside the layer
        forks, both of which write `new_increment` back.

        **Coverage equals the one-shot set** (rule 4): the per-round groups partition
        `patch_selection`, so their union is exactly it.

        And the three specific to a vision+LLM axis:

        **The LLM input is MIXED, never wholly replaced.** Every corrected patch changes its token's
        feature, so handing the LLM the whole projected corrected state would recompute the LLM half
        regardless of its budget. `feats_appr` is the baseline and corrected features enter only at
        tokens that have arrived.

        **Text joins the FINAL round only.** Text is never approximated, but it attends to the image
        block and the answer is read off the last position, which is text; leaving it out gives
        correction no path to the output. One correction against the fully corrected K/V suffices
        because text is causal and sits after the image.

        **`arrived_t` tracks the corrected tokens, not the touched ones.** A token whose patch group
        arrived but which the LLM budget did not select is NOT corrected, so its feature must stay
        approximate -- otherwise the mix mask and the correction mask disagree and `g=1` stops
        reproducing one-shot.
        """
        n_vis = self.n_vision
        n_patch = int(patch_selection.shape[1])
        n_tok = self.n_tokens(n_patch)
        freqs = self.rope_freqs(patch_positions)

        x_appr = self.vision_prepare(px_approx)
        x_full = self.vision_prepare(px_full)
        seq_len = int(input_ids.shape[1])
        bounds = self.layer_bounds(groups, n_patch, seq_len)
        groups_p = self.spatial_groups(patch_selection, groups, patch_positions)

        cache: Dict[str, Any] = {}
        arrived_p = torch.zeros_like(patch_selection)
        arrived_t = None
        feats_appr = None
        ctx = None
        emb = None
        vh = None
        llm_depth = 0                                   # LLM stages already walked approximately
        stats = {"layer_corrections": 0}

        def llm_input():
            """Layer-0 LLM input: approximate everywhere, corrected at arrived tokens."""
            feats = self.project(vh, patch_positions)
            mixed = torch.where(arrived_t.unsqueeze(-1), feats, feats_appr)
            return self.llm_prepare(input_ids, mixed)[0]

        def cross_projector():
            nonlocal feats_appr, ctx, arrived_t, emb
            feats_appr = self.project(vh, patch_positions)
            emb_appr, ctx = self.llm_prepare(input_ids, feats_appr)
            arrived_t = torch.zeros(input_ids.shape[0], n_tok, dtype=torch.bool,
                                    device=input_ids.device)
            emb = emb_appr

        # Opening approximate pass, up to the first bound, on the PURE approximate input --
        # nothing has arrived yet (contract rule 2). Arrival 0: it needs only the base image, so it
        # overlaps the detail transmission entirely.
        v_front = min(bounds[0], n_vis)
        with self._arrival(0), self._stage("approx"):
            vh, cache = self.vision_approx(x_appr, freqs, cache, layers=(0, v_front))
            if bounds[0] > n_vis:
                cross_projector()
                emb, cache = self.llm_approx(emb, ctx, cache, layers=(0, bounds[0] - n_vis))
                llm_depth = bounds[0] - n_vis

        for r in range(groups):
            last = (r == groups - 1)
            arrived_p = arrived_p | groups_p[r]
            stream = torch.where(arrived_p.unsqueeze(-1), x_full, x_appr)

            # Round r cannot start before group r lands, so its whole body -- the correction AND
            # the approximate frontier that follows it -- is charged to arrival r+1. Only the
            # highest index survives as critical, which at g=4 is exactly the r=3 body.
            with self._arrival(r + 1):
                # --- correct this round's group over the vision stages walked so far ---
                if v_front > 0 and bool(groups_p[r].any()):
                    with self._stage("vision_correct"):
                        vh, cache = self.vision_correct(stream, groups_p[r], freqs, cache,
                                                        layers=(0, v_front))
                    stats["layer_corrections"] += v_front

                # --- and over the LLM stages walked so far ---
                if llm_depth > 0:
                    this_round = (self.patch_mask_any_to_token(groups_p[r])
                                  & llm_oneshot[:, ctx["image_positions"]])
                    arrived_t = arrived_t | this_round
                    tm = torch.zeros(input_ids.shape[0], seq_len, dtype=torch.bool,
                                     device=input_ids.device)
                    tm[:, ctx["image_positions"]] = this_round
                    if last:
                        is_text = torch.ones_like(tm)
                        is_text[:, ctx["image_positions"]] = False
                        tm = tm | is_text
                    if bool(tm.any()):
                        with self._stage("llm_correct"):
                            emb, cache = self.llm_correct(llm_input(), tm, ctx, cache,
                                                          layers=(0, llm_depth))
                        stats["layer_corrections"] += llm_depth

                # --- advance the approximate frontier to the next bound ---
                nxt = bounds[r + 1] if r + 1 < len(bounds) else self.n_stages
                with self._stage("approx"):
                    if v_front < min(nxt, n_vis):
                        vh, cache = self.vision_approx(vh, freqs, cache,
                                                       layers=(v_front, min(nxt, n_vis)))
                        v_front = min(nxt, n_vis)
                    if nxt > n_vis:
                        if feats_appr is None:          # first time the axis crosses the projector
                            cross_projector()
                        if nxt - n_vis > llm_depth:
                            emb, cache = self.llm_approx(emb, ctx, cache,
                                                         layers=(llm_depth, nxt - n_vis))
                            llm_depth = nxt - n_vis

        # Return the PRE-finish hidden state: every driver applies `llm_finish` itself before the
        # lm_head. Returning a finished state here made the Gemma 3 interleaved arm norm twice,
        # which its axis gate could not see because the gate's reference was finished too -- rel
        # 0.00e+00 while the driver lost 20pp.
        return emb, cache, stats

    # --- streaming prefill ------------------------------------------------------------------------ #

    def token_bands(self, groups: int, patch_positions: torch.Tensor) -> List[Tuple[int, int]]:
        """Arrival bands as [t_start, t_end) ranges over IMAGE TOKEN indices.

        Chunked prefill needs each band to be a contiguous, increasing range of LLM positions, and
        on this layout it is -- verified on three resolutions: token `t` owns patches `4t..4t+3`,
        those four always share one 2x2 block, and the token index equals the raster index of the
        MERGED grid (`t == row*(W/2) + col`). So a band of merged rows is exactly a token slice.

        Returns ranges rather than masks because that is what a prefill chunk is.
        """
        pp = patch_positions.squeeze(0) if patch_positions.dim() == 3 else patch_positions
        n_tok = pp.shape[0] // self.merge
        tok_row = (pp[:, 1].view(n_tok, self.merge)[:, 0] // self.merge_size)
        n_rows = int(tok_row.max().item()) + 1
        edges = torch.linspace(0, n_rows, groups + 1).round().long()
        out = []
        for r in range(groups):
            idx = ((tok_row >= edges[r]) & (tok_row < edges[r + 1])).nonzero().flatten()
            if idx.numel() == 0:
                out.append((0, 0))
                continue
            # Returning [first, last+1] is only the band if the set between them is unbroken.
            # It is, because `tok_row` is non-decreasing -- but if that ever stopped holding, this
            # range would silently swallow tokens from neighbouring bands, they would be corrected
            # and prefilled twice, and the cost claim would be wrong while every accuracy number
            # still looked plausible. Cheap to check, so check.
            assert int(idx[-1]) - int(idx[0]) + 1 == idx.numel(), (
                f"band {r} is not a contiguous token range ({idx.numel()} tokens spanning "
                f"{int(idx[0])}..{int(idx[-1])}); chunked prefill assumes bands are slices")
            out.append((int(idx[0]), int(idx[-1]) + 1))
        return out

    @torch.no_grad()
    def streaming_forward(self, px_base, px_full, patch_positions, input_ids, groups):
        """Progressive arrival with EXACT chunked prefill -- the vision half approximates, the LLM
        half does not approximate at all.

        This trades the opposite way from `interleaved_forward`. There, both halves run
        approximate-then-correct and the LLM's untouched positions are reconstructed from a cached
        increment. Here vision correctness is deliberately given up, and in exchange the LLM half is
        *exact given its input*: every token is prefilled exactly once, in causal order, against
        real K/V. Since the LLM is 86.5% of the axis and its image block is causal (measured: a
        perturbation at image token 330 moves positions after it and leaves positions before it at
        exactly 0.0), that removes approximation from where nearly all the compute is.

        The schedule, for `groups` arrival rounds:

            base image     -> full 24-layer vision encoder, caching K/V for every patch
            band r arrives -> recompute band r's patches through all 24 layers against the STORED
                              K/V, then prefill the LLM for band r's token positions
            ...
            trailing text  -> prefilled last, against fully arrived image K/V

        What is given up, stated plainly: vision attention is global and bidirectional, so band r's
        features change when band r+1 arrives -- and band r has already been prefilled by then. The
        LLM therefore consumes a STALE view of every band but the last. That is the deliberate
        trade, not an oversight; `interleaved_forward` is the arm that instead keeps correcting and
        pays for it in the LLM half.

        Cost: vision runs twice (one base pass, plus the bands which together are one full pass) and
        the LLM runs exactly once -- 2 x 0.135 + 0.865 = 1.135x a stock forward. The critical path
        after the LAST byte arrives is only the final band: one band's vision recompute plus one
        prefill chunk.

        Returns (last_hidden_normed, kv_cache, stats). The hidden state is already `lm.norm`ed,
        because the stock `Qwen3Model` applies it -- do not apply `llm_finish` again.
        """
        from transformers import DynamicCache

        freqs = self.rope_freqs(patch_positions)
        n_patch = int(px_full.shape[0])
        n_tok = self.n_tokens(n_patch)
        bands = self.token_bands(groups, patch_positions)

        x_base = self.vision_prepare(px_base)
        x_full = self.vision_prepare(px_full)

        # --- base image: the whole encoder, so every token has a value and every layer a K/V ---
        # Arrival 0 -- it needs only the base image, so it overlaps the detail transmission.
        cache: Dict[str, Any] = {}
        with self._arrival(0), self._stage("vision_base"):
            vh, cache = self.vision_approx(x_base, freqs, cache)
        stats = {"vision_layer_passes": self.n_vision, "prefill_tokens": 0}

        emb_all = self.lm.get_input_embeddings()(input_ids)
        img_pos = (input_ids[0] == self.cfg.image_token_id).nonzero(as_tuple=True)[0]
        assert img_pos.numel() == n_tok, f"{img_pos.numel()} image tokens for {n_tok} expected"
        lo, seq = int(img_pos[0]), int(input_ids.shape[1])
        assert bool((img_pos[1:] - img_pos[:-1] == 1).all()), (
            "image tokens are not contiguous; a band would not be one prefill chunk")

        kv = DynamicCache(config=self.cfg.text_config)
        pos_done = 0
        last_hidden = None

        def prefill(end: int):
            """Prefill positions [pos_done, end) against the growing cache. Exact and causal."""
            nonlocal pos_done, last_hidden
            if end <= pos_done:
                return
            cp = torch.arange(pos_done, end, device=input_ids.device)
            out = self.lm(inputs_embeds=emb_all[:, pos_done:end],
                          past_key_values=kv, position_ids=cp.unsqueeze(0),
                          cache_position=cp, use_cache=True)
            last_hidden = out.last_hidden_state
            stats["prefill_tokens"] += end - pos_done
            pos_done = end

        arrived = torch.zeros(1, n_patch, dtype=torch.bool, device=x_full.device)
        last_arrival = 0
        for r, (t0, t1) in enumerate(bands):
            if t1 <= t0:
                continue
            # Band r cannot start before band r lands, so its whole body is charged to arrival r+1.
            # Only the highest index survives as critical, which is the 1/g claim this schedule
            # makes: one band's vision recompute plus one prefill chunk.
            last_arrival = r + 1
            with self._arrival(last_arrival):
                pm = torch.zeros(1, n_patch, dtype=torch.bool, device=x_full.device)
                pm[0, t0 * self.merge:t1 * self.merge] = True
                arrived |= pm
                stream = torch.where(arrived.unsqueeze(-1), x_full, x_base)

                # Recompute this band against the stored K/V -- which already carries earlier
                # bands' corrected keys and values, so a later band sees a better context.
                with self._stage("vision_correct"):
                    vh, cache = self.vision_correct(stream, pm, freqs, cache)
                stats["vision_layer_passes"] += self.n_vision * (t1 - t0) / n_tok

                # Project only this band. The merger is per-token (LayerNorm over features, a
                # reshape of 4 consecutive patches, then an MLP), so slicing it is exact.
                sl = slice(t0 * self.merge, t1 * self.merge)
                pp_sl = (patch_positions.squeeze(0) if patch_positions.dim() == 3
                         else patch_positions)[sl]
                with self._stage("project"):
                    emb_all[:, lo + t0:lo + t1] = self.project(vh[:, sl], pp_sl).reshape(
                        1, t1 - t0, -1).to(emb_all.dtype)

                # Everything up to the end of this band is now final as far as this schedule is
                # concerned; prefill it. On r=0 this also carries the leading text.
                with self._stage("llm_prefill"):
                    prefill(lo + t1)

        # The trailing text waits on the last band, so it belongs to that same arrival -- charging
        # it to a later index of its own would invent an arrival the transmission never had.
        with self._arrival(last_arrival), self._stage("llm_prefill"):
            prefill(seq)                  # the question and the generation prompt
        return last_hidden, kv, stats

    # --- whole axis ----------------------------------------------------------------------------- #

    @torch.no_grad()
    def full_forward(self, pixel_values: torch.Tensor, patch_positions: torch.Tensor,
                     input_ids: torch.Tensor) -> torch.Tensor:
        """Stock-equivalent walk of all 60 stages; the reference the gates compare against."""
        freqs = self.rope_freqs(patch_positions)
        h = self.vision_prepare(pixel_values)
        for l in self.vt.encoder.layers:
            h = l(h, rotary_pos_emb=freqs, attention_mask=None)
            if isinstance(h, tuple):
                h = h[0]
        emb, ctx = self.llm_prepare(input_ids, self.project(h, patch_positions))
        for l in self.lm.layers:
            out = l(emb, position_embeddings=ctx["pe"], attention_mask=ctx["mask"],
                    position_ids=ctx["position_ids"])
            emb = out[0] if isinstance(out, tuple) else out
        return self.llm_finish(emb)
