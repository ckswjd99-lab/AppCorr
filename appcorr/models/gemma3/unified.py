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

    @torch.no_grad()
    def vision_correct(self, hidden, patch_mask, cache, layers: Optional[Tuple[int, int]] = None):
        a, b = (0, self.n_vision) if layers is None else layers
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
        for i in range(a, b):
            pe, m = self._layer_ctx(i, ctx)
            hidden, cache = self.llm_layers[i].correct(hidden, token_mask, pe, m, cache, f"l{i}")
        return hidden, cache

    @torch.no_grad()
    def llm_finish(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm.norm(hidden)

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
