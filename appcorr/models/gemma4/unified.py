"""Gemma 4 31B unified axis — level 1 (2026-08-28, pre-maintenance).

Scope of this first cut (see docs/memo/gemma4_port_scoping.md for the full plan):
`build_inputs`, a manual `full_forward` that must match stock bit-for-bit, the
level-2 degradation that preserves the native-resolution patch grid, and the
floor arm. The vision approx/correct fork and the interleaved walk are steps 3-4
of the plan and intentionally NOT here yet.

Architecture notes that shaped this file:
- Native-resolution ViT: processor resizes aspect-preserving to <= 2520 patches
  (280 soft tokens x 3x3 pool) and emits `pixel_values` [n_patches, 768] plus
  `image_position_ids` [(x,y)] with (-1,-1) padding. Degradation therefore
  operates on the PIL image at its original size and must re-encode to the SAME
  grid (asserted), exactly the gemma3 `l2_from_native` pattern.
- LLM masks: use_bidirectional_attention="vision" -> full-attention layers are
  strictly causal, sliding layers get OR(causal, same-image-block) inside the
  1024 window. We REUSE transformers' own mask builders rather than replicate.
- 31B is dense, no per-layer inputs (hidden_size_per_layer_input=0), no shared
  KV layers (num_kv_shared_layers=0), final_logit_softcapping=30.0 (lm_head
  path only -- irrelevant until logits are compared, where we compare the
  PRE-softcap hidden or apply the same cap).
"""
from typing import Any, Dict, Tuple

import torch

from transformers.models.gemma4.modeling_gemma4 import (
    create_masks_for_vision_model,
    get_block_sequence_ids_for_mask,
)

MODEL_ID_31B = "google/gemma-4-31B-it"


class Gemma4Axis:
    """Thin staged wrapper over Gemma4ForConditionalGeneration (level 1)."""

    def __init__(self, model, proc):
        self.cg = model                       # Gemma4ForConditionalGeneration
        self.model = model.model              # Gemma4Model
        self.vision = model.model.vision_tower
        self.embed_vision = model.model.embed_vision
        self.llm = model.model.language_model
        self.proc = proc
        self.n_llm = model.config.get_text_config().num_hidden_layers

    # ------------------------------------------------------------------ inputs
    def build_inputs(self, img, prompt: str) -> Dict[str, Any]:
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompt}]}]
        return self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")

    def assert_same_grid(self, enc_full: Dict[str, Any], enc_base: Dict[str, Any]):
        """Degradation must not change the native-resolution patch grid."""
        a, b = enc_full["pixel_values"], enc_base["pixel_values"]
        assert a.shape == b.shape, f"patch grid changed: {a.shape} vs {b.shape}"
        pa = enc_full.get("image_position_ids")
        pb = enc_base.get("image_position_ids")
        if pa is not None:
            assert torch.equal(pa, pb), "patch positions changed under degradation"

    # ----------------------------------------------------------------- vision
    def vision_features(self, pixel_values, image_position_ids):
        """Full vision tower -> embedded soft tokens [n_soft, text_hidden].

        Level 1 runs the stock tower end to end. The approx/correct fork (plan
        step 3) will split this into prepare/approx/correct with the pooler and
        float32 standardize kept AFTER the per-round merge.
        """
        out = self.vision(pixel_values=pixel_values,
                          pixel_position_ids=image_position_ids)
        return self.embed_vision(inputs_embeds=out.last_hidden_state)

    # ------------------------------------------------------------------- full
    def full_forward(self, inputs: Dict[str, Any]) -> Tuple[torch.Tensor, Any]:
        """Manual replication of Gemma4Model.forward for the single-image case.

        Returns (last_hidden_state, past_key_values). Must match
        `self.model(**inputs)` exactly -- gated in gemma4_axis_gate.py.
        """
        ids = inputs["input_ids"]
        mm_tti = inputs.get("mm_token_type_ids")
        assert mm_tti is not None, "processor did not emit mm_token_type_ids"

        image_mask = ids == self.cg.config.image_token_id
        llm_ids = torch.where(image_mask, self.cg.config.text_config.pad_token_id, ids)
        embeds = self.model.get_input_embeddings()(llm_ids)

        feats = self.vision_features(inputs["pixel_values"],
                                     inputs.get("image_position_ids"))
        feats = feats.to(embeds.device, embeds.dtype)
        assert int(image_mask.sum()) * embeds.shape[-1] == feats.numel(), (
            f"{int(image_mask.sum())} image slots vs {feats.shape} soft tokens")
        embeds = embeds.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(embeds), feats)

        position_ids = torch.arange(embeds.shape[1], device=embeds.device).unsqueeze(0)
        block_ids = get_block_sequence_ids_for_mask(mm_tti, embeds.device)
        masks = create_masks_for_vision_model(
            config=self.cg.config.get_text_config(),
            inputs_embeds=embeds,
            attention_mask=inputs.get("attention_mask"),
            past_key_values=None,
            position_ids=position_ids,
            block_sequence_ids=block_ids,
        )
        out = self.llm(attention_mask=masks, position_ids=position_ids,
                       inputs_embeds=embeds, use_cache=True, return_dict=True)
        return out.last_hidden_state, out.past_key_values



    # --- LLM approx/correct walk (port-plan step 4, 2026-08-30) ------------------------------- #

    def build_llm_ctx(self, inputs, embeds):
        """Masks (per layer type, via transformers' own builders -- the same calls full_forward
        makes) + per-layer-type rotary embeddings, computed ONCE per request."""
        ids = inputs["input_ids"]
        mm_tti = inputs["mm_token_type_ids"]
        position_ids = torch.arange(embeds.shape[1], device=embeds.device).unsqueeze(0)
        block_ids = get_block_sequence_ids_for_mask(mm_tti, embeds.device)
        masks = create_masks_for_vision_model(
            config=self.cg.config.get_text_config(),
            inputs_embeds=embeds,
            attention_mask=inputs.get("attention_mask"),
            past_key_values=None,
            position_ids=position_ids,
            block_sequence_ids=block_ids,
        )
        tm = self.llm
        pe = {lt: tm.rotary_emb(embeds, position_ids, lt) for lt in tm.unique_layer_types}
        return {"masks": masks, "pe": pe}

    def _ensure_llm_fork(self):
        if not hasattr(self, "llm_fork_layers"):
            from appcorr.models.gemma4.llm_fork import ApproxCorrectGemma4TextLayer
            self.llm_fork_layers = torch.nn.ModuleList(
                ApproxCorrectGemma4TextLayer.from_stock(l)
                for l in self.llm.layers[: self.cg.config.get_text_config().num_hidden_layers])

    def _layer_ctx(self, i, ctx):
        lt = self.cg.config.get_text_config().layer_types[i]
        m = ctx["masks"][lt] if isinstance(ctx["masks"], dict) else ctx["masks"]
        return ctx["pe"][lt], m

    @torch.no_grad()
    def llm_approx(self, hidden, ctx, cache):
        self._ensure_llm_fork()
        for i, layer in enumerate(self.llm_fork_layers):
            pe, m = self._layer_ctx(i, ctx)
            hidden, cache = layer.approx(hidden, pe, m, cache, f"l{i}")
        return hidden, cache

    @torch.no_grad()
    def llm_correct(self, hidden, token_idx, ctx, cache):
        self._ensure_llm_fork()
        for i, layer in enumerate(self.llm_fork_layers):
            pe, m = self._layer_ctx(i, ctx)
            hidden, cache = layer.correct(hidden, token_idx, pe, m, cache, f"l{i}")
        return hidden, cache

    @torch.no_grad()
    def llm_finish(self, hidden):
        return self.llm.norm(hidden)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """lm_head + the 30.0 softcap, matching Gemma4ForConditionalGeneration."""
        lg = self.cg.lm_head(hidden)
        cap = self.cg.config.get_text_config().final_logit_softcapping
        if cap is not None:
            lg = torch.tanh(lg / cap) * cap
        return lg
