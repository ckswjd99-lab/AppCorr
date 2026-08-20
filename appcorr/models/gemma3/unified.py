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

**The token identity changes at stage 27.** Vision stages act on 4096 patches; LLM stages act on 277
sequence positions of which 256 are the pooled image and ~21 are text. A patch selection therefore
has to be *mapped* across the boundary, not carried. `patch_mask_to_llm_mask` does that mapping in
one place: Gemma 3 pools 4096 patches to 256 tokens by a 4x4 average
(`mm_tokens_per_image=256`), so LLM image token `t` covers 16 patches and is marked corrected if any
of them was.

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

    def patch_mask_to_llm_mask(self, patch_mask: torch.Tensor, seq_len: int,
                               image_positions: torch.Tensor) -> torch.Tensor:
        """[B, 4096] over patches -> [B, seq_len] over LLM positions.

        Gemma 3 pools 4096 patches to `mm_tokens_per_image` (256) image tokens, so each LLM image
        token covers 4096/256 = 16 patches; it counts as corrected if ANY of its patches was. Text
        positions are never selected -- they were never approximated.
        """
        b, n_patch = patch_mask.shape
        n_img_tok = int(self.cfg.mm_tokens_per_image)
        per = n_patch // n_img_tok
        pooled = patch_mask.reshape(b, n_img_tok, per).any(dim=-1)      # [B, 256]
        out = torch.zeros(b, seq_len, dtype=torch.bool, device=patch_mask.device)
        out[:, image_positions] = pooled
        return out

    # --- vision half --------------------------------------------------------------------------- #

    @torch.no_grad()
    def vision_prepare(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vt.embeddings(pixel_values)

    @torch.no_grad()
    def vision_approx(self, hidden, cache, layers: Optional[Tuple[int, int]] = None):
        a, b = (0, self.n_vision) if layers is None else layers
        for i in range(a, b):
            hidden, cache = self.vision_layers[i].approx(hidden, cache, f"v{i}")
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
