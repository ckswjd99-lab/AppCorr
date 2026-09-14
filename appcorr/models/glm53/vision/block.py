"""
block.py

`Glm5NextVisionBlock`'s approx/correct fork. The block-level arithmetic is the plain pre-norm
residual pair

    x = x + attn(norm1(x))
    x = x + mlp(norm2(x))

(transformers 5.16.1 `modeling_glm5_next.py:1678-1698`; vLLM writes the same thing with a fused
add-RMSNorm, `multimodal.py:262-279`: `norm2(x, residual=x_attn)` -> `(rms(x + x_attn), x +
x_attn)` then `residual + mlp(...)`, which is the same value), i.e. exactly what
`ApproxCorrectQwen35VisionBlock` implements. The staleness bookkeeping
(`{tag}_blocks_out_sum`, the rule-3 write-back of `docs/memo/interleaved_correction_contract.md`)
is block-shape-only and carries over untouched.

Module-level differences from the GLM-4.6V block, none of which this file branches on -- every
submodule is taken from the stock block rather than reconstructed -- but which a reader porting
changes between the two files must not assume away:

  * **norm1 / norm2 are RMSNorm at eps 1e-6, not 1e-5.** vLLM overrides the checkpoint's
    `vision_config.rms_norm_eps` (`configs/glm5_next.py:315-319`); `vision/stock.py` has the
    detail and `load_stock_vision_tower` applies it.
  * **mlp is a CLAMPED gated SwiGLU**: `down(silu(clamp(gate, max=10)) * clamp(up, -10, 10))` at
    intermediate 4096 over hidden 1024 (`swiglu_limit: 10.0` in the checkpoint's vision_config).
    GLM-4.6V's is an unclamped SwiGLU. Three GEMMs either way, so the FLOP closed form in
    `appcorr/models/glm53/axis.py::_vision_stage_cost` is the GLM-4.6V one with GLM-5.3's widths;
    the clamps are elementwise and excluded under the same convention that excludes the norms.
  * **attn carries per-head q/k RMSNorm** -- see `attention.py`.
"""

from ...qwen35.vision.block import ApproxCorrectQwen35VisionBlock
from .attention import ApproxCorrectGlm5NextVisionAttention


class ApproxCorrectGlm5NextVisionBlock(ApproxCorrectQwen35VisionBlock):
    """See the module docstring: same residual shape as the Qwen3.5 / GLM-4.6V vision block."""

    @classmethod
    def from_stock(cls, blk) -> "ApproxCorrectGlm5NextVisionBlock":
        return cls(
            norm1=blk.norm1,
            attn=ApproxCorrectGlm5NextVisionAttention.from_stock(blk.attn),
            norm2=blk.norm2,
            mlp=blk.mlp,
        )

    # `forward` / `approx` / `correct` / `correct_rows` are inherited unchanged.
