"""Gemma 4 vision-tower fork: approx / partial-correct over the 27-layer ViT.

Port of the gemma3 backbone-fork pattern to Gemma 4's native-resolution encoder
(2D rotary, per-patch (x,y) positions, optional padding rows). Design contract,
same as every AppCorr vision fork:

  - `approx` walks the full tower on the APPROXIMATE patches, caching each
    layer's post-norm post-rope K and post-norm V for every row, plus the final
    hidden state. The cache is the baseline partial correction reconstructs
    untouched rows from.
  - `correct` recomputes ONLY the selected rows through all layers: at layer l
    the selected rows' fresh K/V replace their cached rows, queries exist only
    for selected rows, and attention reads the mixed K/V of the whole image.
    Cached rows' K/V for NON-selected rows are stale w.r.t. this correction --
    the standard AppCorr approximation, identical to gemma3/ov2/sam3.
  - `correct` with ALL rows selected reproduces the stock tower on the mixed
    input exactly (identity gate V2), because every K/V is then fresh.

Numerics: attention goes through the SAME `ALL_ATTENTION_FUNCTIONS` interface
the stock module uses, so approx-vs-stock is bitwise (gate V1). The pscore
attention term is computed as a separate fp32 column-mean side pass (chunked)
that never touches the main path -- FLOPs for it belong to the PSCORE stage.
"""
from typing import Dict, Optional, Tuple

import torch

from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma4.modeling_gemma4 import (
    apply_multidimensional_rope,
    eager_attention_forward,
    repeat_kv,
)


class Gemma4VisionFork:
    def __init__(self, vision_model):
        self.tower = vision_model                    # Gemma4VisionModel
        self.encoder = vision_model.encoder
        self.layers = vision_model.encoder.layers
        self.cfg = vision_model.config

    # ---------------------------------------------------------------- helpers
    def prepare(self, pixel_values: torch.Tensor, positions: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """patch_embedder -> (embeds [B,P,H], padding_positions [B,P])."""
        padding = (positions == -1).all(dim=-1)
        embeds = self.tower.patch_embedder(pixel_values, positions, padding)
        return embeds, padding

    def rope(self, embeds: torch.Tensor, positions: torch.Tensor):
        return self.encoder.rotary_emb(embeds, positions)

    def _mask(self, embeds: torch.Tensor, padding: torch.Tensor):
        return create_bidirectional_mask(
            config=self.cfg, inputs_embeds=embeds, attention_mask=~padding)

    def _attn_fn(self, module):
        return ALL_ATTENTION_FUNCTIONS.get_interface(
            self.cfg._attn_implementation, eager_attention_forward)

    def _qkv(self, module, h_norm, cos, sin, positions):
        """q/k/v exactly as Gemma4VisionAttention.forward computes them.

        h_norm/cos/sin/positions may be row-sliced (partial correction) -- every
        op here is per-row, so slicing commutes with the computation.
        """
        shape = (*h_norm.shape[:-1], -1, module.head_dim)
        q = module.q_norm(module.q_proj(h_norm).view(shape))
        q = apply_multidimensional_rope(q, cos, sin, positions).transpose(1, 2)
        k = module.k_norm(module.k_proj(h_norm).view(shape))
        k = apply_multidimensional_rope(k, cos, sin, positions).transpose(1, 2)
        v = module.v_norm(module.v_proj(h_norm).view(shape)).transpose(1, 2)
        return q, k, v

    # ----------------------------------------------------------------- approx
    def approx(self, embeds: torch.Tensor, positions: torch.Tensor,
               padding: torch.Tensor, cache: Dict[str, torch.Tensor],
               collect_attn: bool = False) -> torch.Tensor:
        """Full walk on the approximate input; fills the K/V cache per layer.

        Returns the tower's last hidden state (pre-pooler). With collect_attn,
        also accumulates `vision_attn_colmean` [B,P]: the mean over layers/heads
        /query-rows of each column's attention mass -- the pscore attention term.
        """
        mask = self._mask(embeds, padding)
        cos, sin = self.rope(embeds, positions)
        h = embeds
        colsum, n_terms = None, 0
        for li, layer in enumerate(self.layers):
            residual = h
            h_norm = layer.input_layernorm(h)
            m = layer.self_attn
            q, k, v = self._qkv(m, h_norm, cos, sin, positions)
            cache[f"v{li}_k"], cache[f"v{li}_v"] = k, v
            attn_fn = self._attn_fn(m)
            out, _ = attn_fn(m, q, k, v, mask, dropout=0.0, scaling=m.scaling)
            if collect_attn:
                # PSCORE side pass: fp32 column mass of softmax(QK^T), chunked
                # over query rows. Never feeds the main path.
                kf = repeat_kv(k, m.num_key_value_groups).float()
                acc = torch.zeros(h.shape[0], h.shape[1], device=h.device)
                for s in range(0, q.shape[2], 1024):
                    qc = q[:, :, s:s + 1024].float()
                    w = torch.softmax(qc @ kf.transpose(2, 3) * m.scaling, dim=-1)
                    if mask is not None:
                        w = torch.softmax(qc @ kf.transpose(2, 3) * m.scaling
                                          + mask[:, :, s:s + 1024].float(), dim=-1)
                    acc += w.sum(dim=(1, 2))
                colsum = acc if colsum is None else colsum + acc
                n_terms += q.shape[1] * q.shape[2]
            out = out.reshape(*h.shape[:-1], -1).contiguous()
            h = residual + layer.post_attention_layernorm(m.o_proj(out))
            residual = h
            h = residual + layer.post_feedforward_layernorm(
                layer.mlp(layer.pre_feedforward_layernorm(h)))
        cache["v_last_hidden"] = h
        if collect_attn and colsum is not None:
            cache["vision_attn_colmean"] = colsum / max(n_terms, 1)
        return h

    # ---------------------------------------------------------------- correct
    def correct(self, mixed_embeds: torch.Tensor, row_mask: torch.Tensor,
                positions: torch.Tensor, padding: torch.Tensor,
                cache: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Recompute the selected rows through all layers against the cache.

        mixed_embeds: layer-0 input where selected rows carry FULL-resolution
        patches and the rest whatever `approx` saw (contract rule: pass the
        layer-0 input, not a running state).
        row_mask: [B,P] bool, True = recompute this row. Updates the cache in
        place (fresh K/V + last hidden for selected rows) so later rounds
        compose, and returns the full last hidden (untouched rows = approx).
        """
        b, p, _ = mixed_embeds.shape
        assert b == 1, "fork is single-image; batch handled by the driver"
        rows = row_mask[0].nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return cache["v_last_hidden"]
        mask = self._mask(mixed_embeds, padding)
        row_mask_q = mask[:, :, rows] if mask is not None else None
        cos, sin = self.rope(mixed_embeds, positions)
        cos_r, sin_r = cos[:, rows], sin[:, rows]
        pos_r = positions[:, rows]
        h_r = mixed_embeds[:, rows]
        for li, layer in enumerate(self.layers):
            residual = h_r
            h_norm = layer.input_layernorm(h_r)
            m = layer.self_attn
            q, k_r, v_r = self._qkv(m, h_norm, cos_r, sin_r, pos_r)
            k_full = cache[f"v{li}_k"].index_copy(2, rows, k_r)
            v_full = cache[f"v{li}_v"].index_copy(2, rows, v_r)
            cache[f"v{li}_k"], cache[f"v{li}_v"] = k_full, v_full
            attn_fn = self._attn_fn(m)
            out, _ = attn_fn(m, q, k_full, v_full, row_mask_q,
                             dropout=0.0, scaling=m.scaling)
            out = out.reshape(*h_r.shape[:-1], -1).contiguous()
            h_r = residual + layer.post_attention_layernorm(m.o_proj(out))
            residual = h_r
            h_r = residual + layer.post_feedforward_layernorm(
                layer.mlp(layer.pre_feedforward_layernorm(h_r)))
        last = cache["v_last_hidden"].index_copy(1, rows, h_r)
        cache["v_last_hidden"] = last
        return last

    # ----------------------------------------------------------------- finish
    def finish(self, last_hidden: torch.Tensor, positions: torch.Tensor,
               padding: torch.Tensor, n_patches: int,
               work_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Pooler + float32 standardize, matching Gemma4VisionModel.forward."""
        pk = self.cfg.pooling_kernel_size
        output_length = n_patches // (pk * pk)
        hidden, pooler_mask = self.tower.pooler(
            hidden_states=last_hidden, pixel_position_ids=positions,
            padding_positions=padding, output_length=output_length)
        hidden = hidden[pooler_mask]
        if self.cfg.standardize:
            hidden = (hidden - self.tower.std_bias.float()) * self.tower.std_scale.float()
        return hidden.to(work_dtype or last_hidden.dtype)
