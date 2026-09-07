"""Mistral Small 3.1 (mistral3/Pixtral) unified axis + vision fork -- level 1-2.

The cheapest port of the family (survey 2026-08-28): Pixtral's tower is a plain
24-layer bidirectional ViT with meshgrid 2D rope (position-INDEXED cos/sin, so
row slicing commutes), single-image mask is None, and the LLM is pure causal
llava-style scatter -- no dual masks, no channels, no windows.

Level-2 scope (same bar as gemma4's day one): build_inputs, manual full_forward
(harness == stock gate), vision fork approx/partial-correct with per-layer K/V
cache, one-shot corrected features. Streaming is the next step (band = merged
patch rows; [IMG_BREAK] tokens sit between LLM rows in the id stream).
"""
from typing import Any, Dict, Tuple

import torch

from transformers.models.pixtral.modeling_pixtral import (
    apply_rotary_pos_emb,
    generate_block_attention_mask,
    position_ids_in_meshgrid,
)

MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


class Mistral3Axis:
    def __init__(self, model, proc):
        self.cg = model                    # Mistral3ForConditionalGeneration
        self.model = model.model           # Mistral3Model
        self.vision = model.model.vision_tower
        self.projector = model.model.multi_modal_projector
        self.llm = model.model.language_model
        self.proc = proc
        self.image_token_id = model.config.image_token_id \
            if getattr(model.config, "image_token_id", None) is not None \
            else model.config.image_token_index

    def build_inputs(self, img, prompt: str) -> Dict[str, Any]:
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompt}]}]
        return self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")

    def assert_same_grid(self, enc_full, enc_base):
        a, b = enc_full["pixel_values"], enc_base["pixel_values"]
        assert a.shape == b.shape, f"grid changed: {a.shape} vs {b.shape}"
        assert torch.equal(enc_full["image_sizes"], enc_base["image_sizes"])

    # ------------------------------------------------------------------ vision
    def vision_features(self, pixel_values, image_sizes):
        """Stock tower + projector -> per-image feature list (B=1: one tensor)."""
        out = self.model.get_image_features(
            pixel_values=pixel_values, image_sizes=image_sizes,
            vision_feature_layer=self.cg.config.vision_feature_layer)
        return out.pooler_output[0]

    def scatter_and_prefill_embeds(self, enc, feats):
        """LLM input embeds with image placeholders replaced by `feats`."""
        ids = enc["input_ids"]
        mask = ids == self.image_token_id
        embeds = self.model.get_input_embeddings()(ids)
        feats = feats.to(embeds.device, embeds.dtype)
        assert int(mask.sum()) * embeds.shape[-1] == feats.numel(), (
            f"{int(mask.sum())} slots vs {feats.shape}")
        return embeds.masked_scatter(mask.unsqueeze(-1).expand_as(embeds), feats)

    def full_forward(self, enc) -> Tuple[torch.Tensor, Any]:
        feats = self.vision_features(enc["pixel_values"], enc["image_sizes"])
        embeds = self.scatter_and_prefill_embeds(enc, feats)
        out = self.llm(inputs_embeds=embeds,
                       attention_mask=enc.get("attention_mask"),
                       use_cache=True, return_dict=True)
        return out.last_hidden_state, out.past_key_values

    def logits(self, hidden):
        return self.cg.lm_head(hidden)


class PixtralVisionFork:
    """approx / partial-correct over the 24-layer bidirectional Pixtral tower.

    Same contract as every AppCorr vision fork (gemma4's docstring has the full
    statement): approx caches post-rope K / V per layer plus the final hidden;
    correct recomputes ONLY the selected rows against the mixed K/V and updates
    the cache in place; correct(ALL rows) must reproduce stock bitwise.
    Single image, B=1, no attention mask (block mask exists only multi-image).
    """

    def __init__(self, vision_model):
        self.tower = vision_model
        self.layers = vision_model.transformer.layers
        self.cfg = vision_model.config

    def prepare(self, pixel_values, image_sizes):
        """conv patch embed + ln_pre -> ([1,P,H], position_ids [P])."""
        pe = self.tower.patch_conv(pixel_values.to(self.tower.patch_conv.weight.dtype))
        lst = [e[..., : (s[0] // self.tower.patch_size), : (s[1] // self.tower.patch_size)]
               for e, s in zip(pe, image_sizes)]
        embeds = torch.cat([p.flatten(1).T for p in lst], dim=0).unsqueeze(0)
        embeds = self.tower.ln_pre(embeds)
        pos = position_ids_in_meshgrid(
            lst, max_width=self.cfg.image_size // self.cfg.patch_size)
        return embeds, pos.to(embeds.device)

    def _rope(self, embeds, pos):
        return self.tower.patch_positional_embedding(embeds, pos)

    def _attn_qkv(self, m, h_norm, cos, sin):
        b, p, _ = h_norm.shape
        shape = (b, p, m.num_heads, m.head_dim)
        q = m.q_proj(h_norm).view(shape).transpose(1, 2)
        k = m.k_proj(h_norm).view(shape).transpose(1, 2)
        v = m.v_proj(h_norm).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)
        return q, k, v

    @staticmethod
    def _sdpa(q, k, v, mask):
        # Stock passes the block mask (zeros for a single image) rather than None;
        # the mask selects the sdpa kernel path, so parity requires passing it too.
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def approx(self, embeds, pos, cache):
        cos, sin = self._rope(embeds, pos)
        mask = generate_block_attention_mask([embeds.shape[1]], embeds)
        cache["p_mask"] = mask
        h = embeds
        for li, layer in enumerate(self.layers):
            m = layer.attention
            h_norm = layer.attention_norm(h)
            q, k, v = self._attn_qkv(m, h_norm, cos, sin)
            cache[f"p{li}_k"], cache[f"p{li}_v"] = k, v
            out = self._sdpa(q, k, v, mask).transpose(1, 2).reshape(h.shape)
            h = h + m.o_proj(out)
            h = h + layer.feed_forward(layer.ffn_norm(h))
        cache["p_last"] = h
        return h

    def correct(self, mixed_embeds, row_mask, pos, cache):
        rows = row_mask.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return cache["p_last"]
        cos, sin = self._rope(mixed_embeds, pos)
        cos_r, sin_r = cos[rows], sin[rows]
        mask_r = cache["p_mask"][:, :, rows]
        h_r = mixed_embeds[:, rows]
        for li, layer in enumerate(self.layers):
            m = layer.attention
            h_norm = layer.attention_norm(h_r)
            q, k_r, v_r = self._attn_qkv(m, h_norm, cos_r, sin_r)
            k = cache[f"p{li}_k"].index_copy(2, rows, k_r)
            v = cache[f"p{li}_v"].index_copy(2, rows, v_r)
            cache[f"p{li}_k"], cache[f"p{li}_v"] = k, v
            out = self._sdpa(q, k, v, mask_r).transpose(1, 2).reshape(h_r.shape)
            h_r = h_r + m.o_proj(out)
            h_r = h_r + layer.feed_forward(layer.ffn_norm(h_r))
        last = cache["p_last"].index_copy(1, rows, h_r)
        cache["p_last"] = last
        return last
