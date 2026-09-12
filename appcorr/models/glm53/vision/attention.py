"""
attention.py

`Glm5NextVisionAttention`'s approx/correct fork. It is the Qwen3.5 / GLM-4.6V vision attention
plus ONE module-level addition, and that addition lands in exactly one method.

Checked term by term against vLLM main 658c813 `vllm/models/glm5next/nvidia/multimodal.py:112-229`
and transformers 5.16.1 `models/glm5_next/modeling_glm5_next.py:1582-1668`:

  * **Fused QKV with bias.** `qkv: [1024 -> 3072]`, reshaped `(T, 3, 16, 64)` and permuted to
    `(3, T, 16, 64)` -- the Qwen2-VL layout `_qkv_heads` already implements. `attention_bias` is
    `true` here (GLM-4.6V's is false), so `qkv` and `proj` both carry bias vectors; the fork reads
    the stock Linears, so that rides along with no branch. (vLLM's `hf_to_vllm_mapper` remaps
    `.attn.q/.k/.v` onto a stacked `qkv`; this checkpoint already ships the fused form, so the
    remap is a no-op -- see `stock.py`.)
  * **NEW: per-head q/k RMSNorm before the RoPE.** `q_norm(q)` / `k_norm(k)` over the head_dim
    axis of `[T, H, 64]`, i.e. per (token, head). GLM-4.6V and Qwen3.5 have no such norm. Its
    epsilon is 1e-5 -- hard-coded in vLLM (`multimodal.py:142-143`), and NOT the 1e-6 vLLM forces
    on this tower's other RMSNorms; see `stock.py`'s docstring for why the two differ.
    It is applied inside `_qkv_heads`, which is the single point every path (`forward`, `approx`,
    `correct`) takes its q/k from and is upstream of the RoPE in all three -- so the approx cache,
    the corrected rows and the received-attention score all see the normalised q/k with no other
    edit. Getting this into `forward` only, or after the RoPE, would be wrong and shape-invisible.
  * **Identical vision RoPE.** `apply_rotary_pos_emb_vision` is character-identical to Qwen3.5's
    (fp32 upcast, `rotate_half` at head_dim/2, cast back); the table is `[T, 64]` built by the
    backbone from a 32-dim (16-frequency) inv_freq over the block-major (h, w) ids.
  * **Full bidirectional attention, varlen by `cu_seqlens`, no windows, no causal mask** --
    `MMEncoderAttention` with `cu_seqlens = cumsum(h*w)` (`multimodal.py:164, 583-587`). So one
    `segment_ranges` for all 24 layers, every layer's received attention comparable, and the
    layer mean well defined (same situation as GLM-4.6V and Qwen3.5).
  * **`scaling = head_dim ** -0.5`** = 64**-0.5, taken from the stock module rather than recomputed.

Everything else -- the deferred received-attention stash, the Triton column-sum kernel, the
sync-free `plan=` correction path -- is inherited as measured.
"""

from typing import Tuple

import torch
import torch.nn as nn

from ...qwen35.vision.attention import (  # noqa: F401  (re-exported for symmetry)
    ApproxCorrectQwen35VisionAttention,
    apply_rotary_pos_emb_vision,
    rotate_half,
)


class ApproxCorrectGlm5NextVisionAttention(ApproxCorrectQwen35VisionAttention):
    """Qwen3.5 vision attention + per-head q/k RMSNorm (see the module docstring)."""

    def __init__(self, qkv: nn.Module, proj: nn.Module, num_heads: int, head_dim: int,
                 scaling: float, q_norm: nn.Module, k_norm: nn.Module):
        super().__init__(qkv=qkv, proj=proj, num_heads=num_heads, head_dim=head_dim,
                         scaling=scaling)
        self.q_norm = q_norm
        self.k_norm = k_norm

    @classmethod
    def from_stock(cls, attn: nn.Module) -> "ApproxCorrectGlm5NextVisionAttention":
        # Asserted rather than assumed: a stock module missing `q_norm` is a DIFFERENT tower
        # (GLM-4.6V's, say) whose output this fork would silently change by not normalising.
        for name in ("qkv", "proj", "num_heads", "head_dim", "scaling", "q_norm", "k_norm"):
            if not hasattr(attn, name):
                raise AttributeError(
                    f"{type(attn).__name__} has no `{name}`: the GLM-5.3-Flash vision attention "
                    "fork assumes the fused-qkv Qwen2-VL layout WITH per-head q/k RMSNorm")
        return cls(qkv=attn.qkv, proj=attn.proj, num_heads=attn.num_heads,
                   head_dim=attn.head_dim, scaling=attn.scaling,
                   q_norm=attn.q_norm, k_norm=attn.k_norm)

    def _qkv_heads(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [T, dim] -> q, k, v each [T, num_heads, head_dim], with q/k RMSNormed.

        The ONE override. Every caller (`forward`, `approx`, `correct`) gets its q/k from here and
        applies the RoPE afterwards, which is the stock order (`multimodal.py:196-219`:
        `split_qkv -> fused_q_kv_rmsnorm -> apply_rotary_emb`; HF :1601-1608 the same)."""
        q, k, v = super()._qkv_heads(x)
        return self.q_norm(q), self.k_norm(k), v
