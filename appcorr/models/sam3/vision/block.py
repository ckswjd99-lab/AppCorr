"""SAM 3 ViT layer with an approx/correct split, mirroring the DINOv3 and CLIP forks.

`approx` runs the whole layer over every token and caches two things: the layer's total increment
(`{tag}_blocks_out_sum` = attention contribution + MLP contribution) so a later `correct` can
reconstruct untouched positions exactly as `x_in + increment`, and the raw K/V so `correct` can
splice fresh keys and values in for the positions it recomputes.

Three things make this harder than the CLIP fork, and each is a place where a plausible-looking
implementation is silently wrong:

**The residual stream is spatial.** SAM 3's ViT carries `[B, H, W, C]` between layers, not
`[B, N, C]`. Token indices are therefore flattened row-major over (H, W), and every gather has to
round-trip through that layout.

**28 of the 32 layers are windowed.** With `window_size=24` on a 72x72 grid there are 9 windows and
attention never crosses them, so a corrected query only ever attends to its own window. Correction
groups the selected tokens by window and runs one attention per window that owns at least one --
recomputing whole windows instead would touch nearly all of them whenever the selection is spread
out, which is exactly the case the method is for. The four global layers (`global_attn_indexes`
[7, 15, 23, 31]) are the single-window case of the same code.

**RoPE positions are window-local on windowed layers.** `Sam3ViTLayer.__init__` builds its rotary
embedding with `end_x = end_y = window_size` whenever `window_size > 0`, so a token's rotary phase
comes from its coordinates *inside* its window, not from its position in the image. Feeding global
coordinates would produce a result that looks reasonable and is wrong everywhere; the unit test in
`analysis/experiments/sam3_vision_fork_unittest.py` exists mainly to catch that.

Padding: `window_partition` pads H and W up to a multiple of the window. At SAM 3's own geometry
(72 = 3 x 24) there is none, and this fork asserts that rather than pretending to handle it -- the
padded tokens would need to be excluded from selection and from the K/V cache, and getting that
wrong is not worth carrying untested.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from transformers.models.sam3.modeling_sam3 import (
    apply_rotary_pos_emb_2d,
    window_partition,
    window_unpartition,
)


class ApproxCorrectSam3ViTLayer(nn.Module):
    """Wraps a stock `Sam3ViTLayer`, adding `.approx()` and `.correct()`."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer_norm1 = layer.layer_norm1
        self.attention = layer.attention
        self.layer_norm2 = layer.layer_norm2
        self.mlp = layer.mlp
        self.dropout = layer.dropout
        self.rotary_emb = layer.rotary_emb
        self.window_size = layer.window_size

    @classmethod
    def from_stock(cls, layer: nn.Module) -> "ApproxCorrectSam3ViTLayer":
        return cls(layer)

    # ---------------------------------------------------------------- helpers

    def _window_grid(self, height: int, width: int) -> tuple[int, int, int]:
        """(windows down, windows across, tokens per window) for this layer's window size."""
        if self.window_size <= 0:
            return 1, 1, height * width
        if height % self.window_size or width % self.window_size:
            raise NotImplementedError(
                f"window_partition would pad a {height}x{width} grid to a multiple of "
                f"{self.window_size}; the approx/correct split does not handle padded tokens. "
                "SAM 3's own 72x72 grid divides exactly."
            )
        return height // self.window_size, width // self.window_size, self.window_size ** 2

    def _locate(self, token_idx: torch.Tensor, height: int, width: int):
        """Map flat token indices to (window id, index within that window).

        Mirrors `window_partition`, which reshapes to
        (B, H/w, w, W/w, w, C) and permutes to (0, 1, 3, 2, 4, 5) -- so windows run row-major over
        the window grid, and tokens run row-major inside each window.
        """
        rows = torch.div(token_idx, width, rounding_mode="floor")
        cols = token_idx % width
        if self.window_size <= 0:
            return torch.zeros_like(token_idx), rows * width + cols
        _, windows_across, _ = self._window_grid(height, width)
        win_id = (rows // self.window_size) * windows_across + (cols // self.window_size)
        in_win = (rows % self.window_size) * self.window_size + (cols % self.window_size)
        return win_id, in_win

    def _project_qkv(self, hidden: torch.Tensor, seq_len: int):
        """q/k/v as [rows, heads, seq_len, head_dim] from a [rows, seq_len, C] input."""
        attn = self.attention
        shape = (hidden.shape[0], seq_len, attn.num_attention_heads, attn.head_dim)
        q = attn.q_proj(hidden).view(*shape).transpose(1, 2)
        k = attn.k_proj(hidden).view(*shape).transpose(1, 2)
        v = attn.v_proj(hidden).view(*shape).transpose(1, 2)
        return q, k, v

    # ------------------------------------------------------------------ stock

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Byte-for-byte the stock `Sam3ViTLayer.forward`, kept as the reference path."""
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, pad_hw = window_partition(hidden_states, self.window_size)
        position_embeddings = self.rotary_emb()
        hidden_states, _ = self.attention(hidden_states, position_embeddings, **kwargs)
        if self.window_size > 0:
            hidden_states = window_unpartition(hidden_states, self.window_size, pad_hw, (height, width))
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + self.dropout(hidden_states)

    # ----------------------------------------------------------------- approx

    def approx(self, x: torch.Tensor, cache_feature: Dict[str, Any], tag: str):
        """Full layer over all tokens; caches the layer increment and the K/V.

        `x` is [B, H, W, C]. The cached K/V are stored in *window* layout,
        [rows, heads, tokens_per_window, head_dim] with rows = B * num_windows, because that is the
        shape `correct` needs to attend against and reshaping it later would only invite an error.
        """
        batch, height, width, channels = x.shape
        _, _, per_window = self._window_grid(height, width)

        normed = self.layer_norm1(x)
        if self.window_size > 0:
            windows, pad_hw = window_partition(normed, self.window_size)
        else:
            windows, pad_hw = normed, (height, width)

        q, k, v = self._project_qkv(windows.reshape(windows.shape[0], per_window, channels), per_window)
        cos, sin = self.rotary_emb()
        q, k = apply_rotary_pos_emb_2d(q, k, cos=cos, sin=sin)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, scale=self.attention.scaling
        )
        attn_out = attn_out.transpose(1, 2).reshape(windows.shape[0], *windows.shape[1:-1], channels)
        attn_out = self.attention.o_proj(attn_out)

        if self.window_size > 0:
            attn_out = window_unpartition(attn_out, self.window_size, pad_hw, (height, width))

        # Cache the post-RoPE K/V: `correct` splices fresh rows into these and attends directly, so
        # storing the pre-RoPE values would mean re-deriving each row's phase on every round.
        cache_feature[f"{tag}_k"] = k.detach()
        cache_feature[f"{tag}_v"] = v.detach()

        x_mid = x + attn_out
        mlp_out = self.dropout(self.mlp(self.layer_norm2(x_mid)))
        cache_feature[f"{tag}_blocks_out_sum"] = (attn_out + mlp_out).detach()
        return x_mid + mlp_out, cache_feature

    # ---------------------------------------------------------------- correct

    def correct(self, x: torch.Tensor, token_idx: torch.Tensor, cache_feature: Dict[str, Any], tag: str):
        """Recompute `token_idx` exactly; reconstruct every other position from the cached increment.

        `token_idx` indexes the flattened (H, W) grid. Positions not in it come out identical to an
        approx-only forward, which is what makes partial correction meaningful rather than a
        different model.
        """
        batch, height, width, channels = x.shape
        _, _, per_window = self._window_grid(height, width)
        token_idx = token_idx.to(x.device)

        k_cache = cache_feature[f"{tag}_k"].clone()
        v_cache = cache_feature[f"{tag}_v"].clone()
        win_id, in_win = self._locate(token_idx, height, width)

        flat = x.reshape(batch, height * width, channels)
        x_active = flat[:, token_idx]                      # [B, Q, C]
        normed = self.layer_norm1(x_active)

        q_sel, k_sel, v_sel = self._project_qkv(normed, normed.shape[1])
        cos, sin = self.rotary_emb()
        # RoPE phase comes from the position *inside the window* on windowed layers -- see the module
        # docstring. `cos`/`sin` are indexed by that, never by the global token index.
        q_sel, k_sel = apply_rotary_pos_emb_2d(q_sel, k_sel, cos=cos[in_win], sin=sin[in_win])

        attn_pieces = torch.zeros(
            batch, self.attention.num_attention_heads, token_idx.numel(), self.attention.head_dim,
            device=x.device, dtype=q_sel.dtype,
        )
        for w in torch.unique(win_id).tolist():
            sel = (win_id == w).nonzero(as_tuple=True)[0]
            rows = torch.arange(batch, device=x.device) * (k_cache.shape[0] // batch) + w
            # Fresh K/V for the corrected positions, stale cached K/V for the rest of the window.
            k_cache[rows[:, None], :, in_win[sel][None, :], :] = k_sel[:, :, sel, :].transpose(1, 2)
            v_cache[rows[:, None], :, in_win[sel][None, :], :] = v_sel[:, :, sel, :].transpose(1, 2)
            attn_pieces[:, :, sel, :] = torch.nn.functional.scaled_dot_product_attention(
                q_sel[:, :, sel, :], k_cache[rows], v_cache[rows],
                attn_mask=None, dropout_p=0.0, scale=self.attention.scaling,
            )

        attn_sel = self.attention.o_proj(
            attn_pieces.transpose(1, 2).reshape(batch, token_idx.numel(), channels)
        )
        x_attn_active = x_active + attn_sel
        mlp_out_new = self.dropout(self.mlp(self.layer_norm2(x_attn_active)))

        increment = cache_feature[f"{tag}_blocks_out_sum"].reshape(batch, height * width, channels)
        out = (flat + increment.to(flat.dtype)).clone()
        out[:, token_idx] = (x_attn_active + mlp_out_new).to(out.dtype)

        cache_feature[f"{tag}_k"] = k_cache
        cache_feature[f"{tag}_v"] = v_cache
        return out.reshape(batch, height, width, channels), cache_feature
