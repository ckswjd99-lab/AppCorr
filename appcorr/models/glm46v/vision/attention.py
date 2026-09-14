"""
attention.py

`Glm4vMoeVisionAttention`'s approx/correct fork. It is `ApproxCorrectQwen35VisionAttention`
with nothing changed, and this file exists to say WHY that is legitimate rather than to
re-type 255 lines that would then drift.

Checked term by term against `transformers/models/glm4v_moe/modeling_glm4v_moe.py`
(`Glm4vMoeVisionAttention.forward`, :636-700) and against vLLM 0.28's `Glm4vVisionAttention`
(`vllm/model_executor/models/glm4_1v.py:272-380`):

  * **Fused QKV, one Linear.** `self.qkv: [dim -> 3*dim]`, reshaped `(T, 3, heads, head_dim)`
    and permuted to `(3, T, heads, head_dim)` -- byte for byte the Qwen2-VL family's layout,
    which is what `_qkv_heads` implements. `attention_bias` is false in this checkpoint, so the
    Linear carries no bias; the fork reads the module, so a bias would ride along anyway.
  * **Identical vision RoPE.** glm4v_moe's `apply_rotary_pos_emb_vision` (:607-619) is
    character-identical to Qwen3.5's (fp32 upcast of q/k/cos/sin, `rotate_half` split at
    head_dim/2, cast back). The tables differ only in how the backbone builds them
    (`Glm4vMoeVisionRotaryEmbedding(head_dim // 2)` over the block-major (h, w) position ids,
    `cat(rot, rot)` -> cos/sin), and that is the backbone's job, not this module's.
  * **Full bidirectional attention, varlen by `cu_seqlens`, no windows and no causal mask** --
    one `segment_ranges` for the whole 24-layer tower, exactly the Qwen3.5 situation. GLM's
    tower has no `fullatt_block_indexes` / `window_size` knob at all (its vision config has
    neither field), so there is no windowed subset to exclude from the received-attention
    layer mean either: all 24 layers are comparable.
  * **`scaling = head_dim ** -0.5`**, taken from the stock module rather than recomputed.
  * `proj` is a plain `[dim -> dim]` Linear with no bias, applied to the concatenated heads.

What follows from the sameness: the deferred received-attention stash, the Triton column-sum
kernel (`qwen35/vision/recv_attn_triton.py`) and the sync-free `plan=` correction path are all
inherited as measured, not re-derived.
"""

from ...qwen35.vision.attention import (  # noqa: F401  (re-exported for symmetry with qwen35)
    ApproxCorrectQwen35VisionAttention,
    apply_rotary_pos_emb_vision,
    rotate_half,
)


class ApproxCorrectGlm4vVisionAttention(ApproxCorrectQwen35VisionAttention):
    """See the module docstring: structurally identical to the Qwen3.5 vision attention."""

    @classmethod
    def from_stock(cls, attn) -> "ApproxCorrectGlm4vVisionAttention":
        # `Glm4vMoeVisionAttention` exposes qkv / proj / num_heads / head_dim / scaling under
        # exactly these names (modeling_glm4v_moe.py:621-634); asserted so a renamed attribute
        # is a crash rather than a silently different tower.
        for name in ("qkv", "proj", "num_heads", "head_dim", "scaling"):
            if not hasattr(attn, name):
                raise AttributeError(
                    f"{type(attn).__name__} has no `{name}`: the GLM-4.6V vision attention fork "
                    "assumes the Qwen2-VL-family module layout (fused qkv, per-head scaling)")
        return cls(qkv=attn.qkv, proj=attn.proj, num_heads=attn.num_heads,
                   head_dim=attn.head_dim, scaling=attn.scaling)
