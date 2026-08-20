"""Gemma 3 as ONE approx/correct axis: 27 SigLIP layers then 34 decoder layers.

Why unify instead of forking the vision tower alone. Measured on Gemma3-4B, one 896x896 image, bf16:

    vision tower (4096 patches x 27L, h1152)     9.39 ms   27% of the forward
    LLM prefill  (277 tokens  x 34L, h2560)     16.96 ms   49%
    full forward                                34.38 ms

Prefill is **1.8x** the vision tower, against the intuition that 277 tokens must be negligible: the
token count is small but h2560 x 34 layers is not, while the vision tower is wide in patches and
thin in width. Approx/correcting the vision tower alone overlaps 27% of the forward; walking both as
one axis reaches ~76%. This is the VGGT lesson -- there, treating the 24 patch-embed blocks as
"preprocessing" repeated them in full on every interleaved round, 28.7% of the forward wasted.

Three things make the axis more than two loops glued together:

**The token identity changes at stage 27, and the two halves get SEPARATE budgets.** Vision stages
act on 4096 patches; LLM stages act on ~272 positions of which 256 are the pooled image. Translating
a patch selection into a token selection does not work: each token pools 16 patches, so "corrected
if any of mine was" saturates -- measured, a 55% patch keep marks 86% of tokens and 10% already marks
43%. Instead the patch *score* is pooled the same 16:1 way and the LLM half runs its own top-k.

**Text tokens are always exact.** They arrive as text, not pixels, so there is nothing to
approximate and nothing to correct -- the role VGGT's camera and register tokens play. They are
excluded from selection and left out of the recompute budget.

**Sliding and full layers need different rope AND different masks.** `Gemma3RotaryEmbedding` takes a
`layer_type` and returns per-type cos/sin; the mask comes as a
`{full_attention, sliding_attention}` dict. Reusing one layer type's values across all 34 layers
silently corrupts the 5 full-attention layers -- an error the LLM fork's unit test measures at rel
9e-2, invisible in an argmax.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .llm.decoder_layer import ApproxCorrectGemma3DecoderLayer
from .vision.block import ApproxCorrectSiglipLayer


class Gemma3UnifiedAxis(nn.Module):
    """Wraps a stock `Gemma3ForConditionalGeneration`'s inner model as one approx/correct axis."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model                                   # Gemma3Model
        vt = model.vision_tower
        self.vt = vt
        self.vision_layers = nn.ModuleList(
            ApproxCorrectSiglipLayer.from_stock(l) for l in vt.encoder.layers)
        lm = model.language_model
        self.lm = lm
        self.llm_layers = nn.ModuleList(
            ApproxCorrectGemma3DecoderLayer.from_stock(l) for l in lm.layers)
        self.cfg = model.config
        self._last_seq_len = None

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

    def patch_grid(self) -> Tuple[int, int, int]:
        """(patches per side, tokens per side, pooling kernel) -- 64, 16, 4 on Gemma 3 4B."""
        v = self.cfg.vision_config
        pps = int(v.image_size // v.patch_size)
        tps = int(self.cfg.mm_tokens_per_image ** 0.5)
        return pps, tps, pps // tps

    def patch_to_token(self) -> torch.Tensor:
        """[n_patch] -> the image-token index each patch belongs to.

        Gemma 3 pools with `AvgPool2d(kernel_size=4, stride=4)` over the 64x64 patch GRID, so an
        image token owns a 4x4 spatial BLOCK of patches -- not 16 consecutive patches in raster
        order. Grouping by `reshape(256, 16)` picks a 16x1 strip instead and lands only 6.2% of
        patches in the right token, which is close enough to random to be worse than useless while
        still producing plausible numbers.
        """
        pps, tps, k = self.patch_grid()
        idx = torch.arange(pps * pps)
        r, c = idx // pps, idx % pps
        return (r // k) * tps + (c // k)

    def pool_patch_score(self, patch_score: torch.Tensor) -> torch.Tensor:
        """[B, n_patch] patch scores -> [B, n_token] token scores, by Gemma 3's own 4x4 pooling."""
        b = patch_score.shape[0]
        n_tok = int(self.cfg.mm_tokens_per_image)
        p2t = self.patch_to_token().to(patch_score.device)
        out = torch.zeros(b, n_tok, device=patch_score.device, dtype=patch_score.dtype)
        out.index_add_(1, p2t, patch_score)
        return out / (patch_score.shape[1] / n_tok)

    def token_mask_to_patch_mask(self, token_sel: torch.Tensor, n_patch: int) -> torch.Tensor:
        """[B, n_token] -> [B, n_patch]: every patch of a selected token, by the same 4x4 blocks."""
        p2t = self.patch_to_token().to(token_sel.device)
        return token_sel[:, p2t]

    def patch_mask_any_to_token(self, patch_mask: torch.Tensor) -> torch.Tensor:
        """[B, n_patch] -> [B, n_token]: token selected if ANY patch of its 4x4 block was."""
        b = patch_mask.shape[0]
        n_tok = int(self.cfg.mm_tokens_per_image)
        p2t = self.patch_to_token().to(patch_mask.device)
        counts = torch.zeros(b, n_tok, device=patch_mask.device, dtype=torch.int32)
        counts.index_add_(1, p2t, patch_mask.to(torch.int32))
        return counts > 0

    def llm_mask_from_score(self, patch_score: torch.Tensor, keep: float, seq_len: int,
                            image_positions: torch.Tensor) -> torch.Tensor:
        """Select image tokens on their OWN budget, from pooled scores. [B, seq_len].

        Do NOT translate the patch selection across the boundary. Each image token pools 16 patches,
        so "corrected if any of my patches was" saturates: measured on 20 RealWorldQA images with the
        real residual-energy score, a 55% patch keep marks **219.8 of 256** tokens (86%), and 10%
        already marks 109.6. Real scores do cluster -- 0.52x the tokens a uniformly random selection
        of the same size would touch, at 10% -- but nowhere near the 26 tokens an ideally clustered
        selection needs. Under `any()` nothing saved in the vision half reaches the LLM half.

        The two halves simply have different units (4096 patches vs 256 tokens), so they get
        different budgets, and `keep` here means what it says. It also becomes a knob worth having:
        the vision tower and the prefill need not be equally sensitive to approximation.

        Text positions are never selected -- they arrive as text, were never approximated, and
        spending budget on them would recompute tokens that are already exact.
        """
        pooled = self.pool_patch_score(patch_score)                    # [B, 256]
        b, n_tok = pooled.shape
        k = max(1, int(round(keep * n_tok)))
        idx = pooled.topk(k, dim=-1).indices
        sel = torch.zeros_like(pooled, dtype=torch.bool).scatter_(1, idx, True)
        out = torch.zeros(b, seq_len, dtype=torch.bool, device=patch_score.device)
        out[:, image_positions] = sel
        return out

    # --- vision half --------------------------------------------------------------------------- #

    @torch.no_grad()
    def vision_prepare(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vt.embeddings(pixel_values)

    @torch.no_grad()
    def _incoming_attention(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Column mass of one layer's attention, head- and query-averaged. [B, patches].

        How much the rest of the image reads FROM each patch -- the term the standard patch score
        multiplies residual energy by. Taken on the approximate pass, from the same projections the
        layer is about to use, so it costs one extra QK product and no extra weights.

        SigLIP has no CLS token, so the result already lines up with patch indices; InternVL's
        equivalent has to drop position 0.
        """
        layer = self.vision_layers[layer_idx]
        q, k, _ = layer._qkv(layer.layer_norm1(hidden))
        scale = layer.self_attn.scale
        b, heads, seq, _ = q.shape
        # [B, heads, 4096, 4096] in fp32 would be ~1 GB per head-batch; accumulate in query chunks.
        col = torch.zeros(b, seq, device=hidden.device, dtype=torch.float32)
        chunk = 512
        for s0 in range(0, seq, chunk):
            e0 = min(s0 + chunk, seq)
            w = torch.softmax((q[:, :, s0:e0] @ k.transpose(-1, -2)) * scale, dim=-1)
            col += w.float().sum(dim=2).mean(dim=1)
        return col / seq

    @torch.no_grad()
    def vision_approx(self, hidden, cache, layers: Optional[Tuple[int, int]] = None,
                      collect_attn: bool = False):
        a, b = (0, self.n_vision) if layers is None else layers
        acc = None
        for i in range(a, b):
            if collect_attn:
                c = self._incoming_attention(hidden, i)
                acc = c if acc is None else acc + c
            hidden, cache = self.vision_layers[i].approx(hidden, cache, f"v{i}")
        if collect_attn and acc is not None:
            cache["vision_patch_attn_layermean"] = acc / max(1, b - a)
        return hidden, cache

    @staticmethod
    def _check_entry(x, cache, tag, what):
        """The entry tensor of a correction walk must be the layer-0 input the approx walk saw.

        Only at layer 0: from layer 1 on, `correct` legitimately carries a *corrected* running state
        that differs from what approx saw, so checking per layer is wrong (it fired on legitimate
        corrections). The mistake worth catching is handing the whole walk the approximate pass's
        OUTPUT -- a 34-layer-deep hidden state fed back into layer 0. That runs, returns plausible
        numbers, and passed every gate that called the API correctly; a driver did exactly it.
        """
        sig = cache.get(f"{tag}_in_sig")
        if sig is None:
            return
        got = (float(x.float().mean()), float(x.float().std()), tuple(x.shape))
        if got[2] != sig[2] or abs(got[1] - sig[1]) > 0.25 * max(abs(sig[1]), 1e-6):
            raise RuntimeError(
                f"{what}_correct got an entry tensor this cache was not built from: approx saw "
                f"mean/std/shape {sig}, got {got}. Pass the same layer-0 input approx received "
                f"(the mixed stream), not the approximate pass's output.")

    @torch.no_grad()
    def vision_correct(self, hidden, patch_mask, cache, layers: Optional[Tuple[int, int]] = None):
        a, b = (0, self.n_vision) if layers is None else layers
        if a == 0:
            self._check_entry(hidden, cache, "v0", "vision")
        for i in range(a, b):
            hidden, cache = self.vision_layers[i].correct(hidden, patch_mask, cache, f"v{i}")
        return hidden, cache

    @torch.no_grad()
    def project(self, vision_hidden: torch.Tensor) -> torch.Tensor:
        """Tower output -> LLM-space image tokens, matching `get_image_features`."""
        return self.model.multi_modal_projector(self.vt.post_layernorm(vision_hidden))

    # --- LLM half ------------------------------------------------------------------------------ #

    @torch.no_grad()
    def llm_prepare(self, input_ids: torch.Tensor, image_features: torch.Tensor,
                    token_type_ids: Optional[torch.Tensor] = None):
        """Embed text, splice image features in, and build the per-layer rope and masks.

        Returns (hidden, ctx) where ctx carries everything the decoder layers need. Built once and
        reused by every round, because none of it depends on which tokens are corrected.
        """
        emb = self.lm.get_input_embeddings()(input_ids)
        img_pos = (input_ids[0] == self.cfg.image_token_id).nonzero(as_tuple=True)[0]
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[:, img_pos] = True
        emb = emb.masked_scatter(mask.unsqueeze(-1),
                                 image_features.reshape(-1, image_features.shape[-1]))

        n = input_ids.shape[1]
        position_ids = torch.arange(n, device=input_ids.device).unsqueeze(0)
        # Per layer_type rope AND per layer_type masks -- see the module docstring.
        pe = {lt: self.lm.rotary_emb(emb, position_ids, lt)
              for lt in set(self.cfg.text_config.layer_types)}
        if token_type_ids is None:
            # The processor emits this; deriving it from image_token_id is a fallback, and the
            # mask it produces is what makes image tokens bidirectional. Passing it is not optional
            # -- without it the model builds a plain causal mask and the whole prefix behaves
            # differently (rel 0.886 against the intended forward).
            token_type_ids = torch.zeros_like(input_ids)
            token_type_ids[:, img_pos] = 1
        from transformers.models.gemma3.modeling_gemma3 import (
            create_masks_for_vision_model, get_block_sequence_ids_for_mask)
        masks = create_masks_for_vision_model(
            config=self.cfg.text_config, inputs_embeds=emb,
            attention_mask=torch.ones_like(input_ids), past_key_values=None,
            position_ids=position_ids,
            block_sequence_ids=get_block_sequence_ids_for_mask(token_type_ids, device=emb.device))
        self._last_seq_len = int(n)
        ctx = {"pe": pe, "masks": masks, "position_ids": position_ids, "image_positions": img_pos}
        return emb, ctx

    def _layer_ctx(self, i: int, ctx):
        lt = self.cfg.text_config.layer_types[i]
        m = ctx["masks"][lt] if isinstance(ctx["masks"], dict) else ctx["masks"]
        return ctx["pe"][lt], m

    @torch.no_grad()
    def llm_approx(self, hidden, ctx, cache, layers: Optional[Tuple[int, int]] = None):
        a, b = (0, self.n_llm) if layers is None else layers
        for i in range(a, b):
            pe, m = self._layer_ctx(i, ctx)
            hidden, cache = self.llm_layers[i].approx(hidden, pe, m, cache, f"l{i}")
        return hidden, cache

    @torch.no_grad()
    def llm_correct(self, hidden, token_mask, ctx, cache, layers: Optional[Tuple[int, int]] = None):
        a, b = (0, self.n_llm) if layers is None else layers
        if a == 0:
            self._check_entry(hidden, cache, "l0", "llm")
        for i in range(a, b):
            pe, m = self._layer_ctx(i, ctx)
            hidden, cache = self.llm_layers[i].correct(hidden, token_mask, pe, m, cache, f"l{i}")
        return hidden, cache

    @torch.no_grad()
    def llm_finish(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm.norm(hidden)

    # --- interleaved scheduling ------------------------------------------------------------------ #

    def stage_costs(self) -> torch.Tensor:
        """Relative cost of each of the 61 stages, for the interleaved cost model.

        Counting stages would be wrong: a vision layer processes 4096 tokens at width 1152 and an
        LLM layer 272 at width 2560, so they are not interchangeable units. Cost is taken as
        tokens x width^2 (the projection/MLP term dominates at these shapes), normalised so the
        whole axis sums to 1. Measured wall clock agrees with the ratio this gives -- vision 9.39 ms
        against LLM 16.96 ms -- which is the check that matters, since the formula is a proxy.
        """
        v = self.cfg.vision_config
        n_patch = int(v.image_size // v.patch_size) ** 2
        cv = n_patch * v.hidden_size ** 2
        t = self.cfg.text_config
        n_tok = self._last_seq_len or int(self.cfg.mm_tokens_per_image)
        cl = n_tok * t.hidden_size ** 2
        costs = torch.tensor([cv] * self.n_vision + [cl] * self.n_llm, dtype=torch.float64)
        return costs / costs.sum()

    def layer_bounds(self, groups: int) -> list:
        """Round boundaries over the 61-stage axis, split by equal COST rather than equal count.

        Equal stage counts would give the early rounds far too little work: the 27 vision stages are
        cheaper per stage than the 34 LLM ones here. The last bound is always the full axis.
        """
        if groups <= 1:
            return [self.n_stages]
        cum = torch.cumsum(self.stage_costs(), 0)
        bounds = []
        for r in range(1, groups):
            target = r / groups
            b = int(torch.searchsorted(cum, torch.tensor(target, dtype=torch.float64)).item()) + 1
            bounds.append(max(1, min(b, self.n_stages - 1)))
        bounds = sorted(set(bounds))
        while len(bounds) < groups - 1:                    # keep g distinct rounds
            for cand in range(1, self.n_stages):
                if cand not in bounds:
                    bounds.append(cand); break
            bounds = sorted(set(bounds))
        return bounds + [self.n_stages]

    def spatial_groups(self, selected_patches: torch.Tensor, groups: int) -> list:
        """Split the SELECTED patches into `groups` contiguous spatial blocks. [B, n_patch] bool.

        Selection and grouping are deliberately separate. Selection stays free -- the top-k patches
        by score, wherever they are -- because forcing patches to follow token blocks costs more
        than it buys (measured: the token-driven arm trails the free one by 13pp). Grouping then
        only decides the ORDER those already-chosen patches arrive in, which is what interleaving
        schedules.

        Blocks are rows of the patch grid, so a group is a contiguous horizontal band -- the
        block_grid shape that beat dispersed grids on ADE20K.
        """
        pps, _, _ = self.patch_grid()
        b, n_patch = selected_patches.shape
        rows = torch.arange(n_patch, device=selected_patches.device) // pps
        edges = torch.linspace(0, pps, groups + 1).round().long()
        out = []
        for r in range(groups):
            band = (rows >= edges[r]) & (rows < edges[r + 1])
            out.append(selected_patches & band.unsqueeze(0))
        return out

    # --- interleaved walk ------------------------------------------------------------------------ #

    @torch.no_grad()
    def interleaved_forward(self, px_full, px_approx, input_ids, token_type_ids,
                            patch_selection, llm_oneshot, groups):
        """Walk the 61-stage axis in `groups` rounds, correcting one group per round.

        Follows docs/memo/interleaved_correction_contract.md. Three things this axis adds to the
        rules, all of them learned the hard way on the one-shot driver:

        **The LLM input is MIXED, never wholly replaced.** Vision correction changes essentially
        every image token's feature (4x4 pooling saturates), so projecting the corrected vision
        state and handing all of it to the LLM means the LLM half was fully recomputed no matter
        what its budget said. `feats_appr` -- the projection of the purely approximate vision pass --
        is the baseline, and corrected features enter only at tokens that have arrived.

        **Text joins the FINAL round only.** Text is never approximated, but it attends to the image
        block and the answer is read off the last position, which is text; leaving it out gives
        correction no path to the output at all. One correction against the fully corrected K/V is
        enough because text is causal and sits after the image.

        **A round corrects its OWN group** -- never the accumulated set, which would make the last
        round equal one-shot.
        """
        n_vis = self.n_vision
        bounds = self.layer_bounds(groups)
        groups_p = self.spatial_groups(patch_selection, groups)

        x_appr = self.vision_prepare(px_approx)
        x_full = self.vision_prepare(px_full)

        cache: Dict[str, Any] = {}
        arrived_p = torch.zeros_like(patch_selection)
        arrived_t = None
        feats_appr = None
        ctx = None
        emb = None
        llm_depth = 0                                   # LLM stages already walked approximately

        def llm_input():
            """Layer-0 LLM input: approximate everywhere, corrected at arrived tokens."""
            feats = self.project(vh)
            mixed = torch.where(arrived_t.unsqueeze(-1), feats, feats_appr)
            return self.llm_prepare(input_ids, mixed, token_type_ids)[0]

        # Opening approximate pass, up to the first bound. Nothing has arrived yet.
        vh, cache = self.vision_approx(x_appr, cache, layers=(0, min(bounds[0], n_vis)))
        v_front = min(bounds[0], n_vis)
        if bounds[0] > n_vis:
            feats_appr = self.project(vh)
            emb_appr, ctx = self.llm_prepare(input_ids, feats_appr, token_type_ids)
            arrived_t = torch.zeros(input_ids.shape[0], int(self.cfg.mm_tokens_per_image),
                                    dtype=torch.bool, device=input_ids.device)
            emb, cache = self.llm_approx(emb_appr, ctx, cache, layers=(0, bounds[0] - n_vis))
            llm_depth = bounds[0] - n_vis

        for r in range(groups):
            last = (r == groups - 1)
            arrived_p = arrived_p | groups_p[r]
            stream = torch.where(arrived_p.unsqueeze(-1), x_full, x_appr)

            # --- correct this round's group over the vision stages walked so far ---
            if v_front > 0 and bool(groups_p[r].any()):
                vh, cache = self.vision_correct(stream, groups_p[r], cache, layers=(0, v_front))

            # --- and over the LLM stages walked so far ---
            if llm_depth > 0:
                arrived_t = arrived_t | self.patch_mask_any_to_token(groups_p[r])
                tm = torch.zeros(input_ids.shape[0], input_ids.shape[1], dtype=torch.bool,
                                 device=input_ids.device)
                tm[:, ctx["image_positions"]] = (
                    self.patch_mask_any_to_token(groups_p[r])
                    & llm_oneshot[:, ctx["image_positions"]])
                if last:
                    is_text = torch.ones_like(tm)
                    is_text[:, ctx["image_positions"]] = False
                    tm = tm | is_text
                if bool(tm.any()):
                    emb, cache = self.llm_correct(llm_input(), tm, ctx, cache,
                                                  layers=(0, llm_depth))

            # --- advance the approximate frontier to the next bound ---
            nxt = bounds[r + 1] if r + 1 < len(bounds) else self.n_stages
            if v_front < min(nxt, n_vis):
                vh, cache = self.vision_approx(vh, cache, layers=(v_front, min(nxt, n_vis)))
                v_front = min(nxt, n_vis)
            if nxt > n_vis:
                if feats_appr is None:                  # first time the axis crosses the projector
                    feats_appr = self.project(vh)
                    emb_appr, ctx = self.llm_prepare(input_ids, feats_appr, token_type_ids)
                    arrived_t = torch.zeros(input_ids.shape[0],
                                            int(self.cfg.mm_tokens_per_image),
                                            dtype=torch.bool, device=input_ids.device)
                    emb = emb_appr
                if nxt - n_vis > llm_depth:
                    emb, cache = self.llm_approx(emb, ctx, cache,
                                                 layers=(llm_depth, nxt - n_vis))
                    llm_depth = nxt - n_vis

        return self.llm_finish(emb), cache

    def _llm_group_mask(self, patch_group, ctx, input_ids, llm_oneshot):
        """This round's LLM-side selection.

        The LLM budget is the SAME set one-shot would correct (`llm_oneshot`, chosen by its own
        top-k on the pooled score); interleaving only decides WHEN each of those tokens is done.
        A token belongs to the round whose patch group touches it, so the union over rounds is
        exactly the one-shot set -- which is coverage rule 4, and what makes `g=1` an identity
        rather than an approximation.

        Recomputing the budget per round instead would let the rounds correct more tokens in total
        than one-shot and quietly make interleaving look better by doing more work.
        """
        touched = self.patch_mask_any_to_token(patch_group)
        out = torch.zeros(input_ids.shape[0], input_ids.shape[1], dtype=torch.bool,
                          device=input_ids.device)
        out[:, ctx["image_positions"]] = touched
        return out & llm_oneshot

    # --- whole axis ----------------------------------------------------------------------------- #

    @torch.no_grad()
    def full_forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor,
                     token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Stock-equivalent walk of all 61 stages; the ceiling arm and the reference for the gates."""
        h = self.vision_prepare(pixel_values)
        for l in self.vt.encoder.layers:
            h = l(h, attention_mask=None)
        emb, ctx = self.llm_prepare(input_ids, self.project(h), token_type_ids)
        for i, l in enumerate(self.lm.layers):
            pe, m = self._layer_ctx(i, ctx)
            emb = l(emb, position_embeddings=pe, attention_mask=m,
                    position_ids=ctx["position_ids"])
        return self.llm_finish(emb)
