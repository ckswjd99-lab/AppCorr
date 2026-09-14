"""
block.py

`Glm4vMoeVisionBlock`'s approx/correct fork. Like `attention.py` this is the Qwen3.5 block with
nothing changed; the block-level arithmetic is the plain pre-norm residual pair

    x = x + attn(norm1(x))
    x = x + mlp(norm2(x))

(`transformers/models/glm4v_moe/modeling_glm4v_moe.py:714-735`), which is precisely what
`ApproxCorrectQwen35VisionBlock` implements, and the staleness bookkeeping
(`{tag}_blocks_out_sum`, the rule-3 write-back of `docs/memo/interleaved_correction_contract.md`)
is block-shape-only: it never looks inside `norm*` or `mlp`.

Two module-level differences that the fork does NOT have to branch on, because every submodule is
taken from the stock block rather than reconstructed -- but which a reader porting further changes
between the two files must not assume away:

  * **norm1 / norm2 are RMSNorm** (`Glm4vMoeRMSNorm`, eps 1e-5), where Qwen3.5's vision block uses
    LayerNorm and Qwen2.5-VL's used RMSNorm. Fused-residual in vLLM (`glm4_1v.py:414-419`,
    `norm2(x, residual=x_attn)` -> `(x + x_attn, rms(x + x_attn))`), which is the same value as
    HF's separate add + norm; the fork follows HF.
  * **mlp is a gated SwiGLU** `down(silu(gate(x)) * up(x))` with intermediate = out_hidden_size =
    4096 (`Glm4vMoeisionMlp`, :482-493), where Qwen3.5's vision MLP is ungated two-layer. This
    matters for the FLOP closed form (3 GEMMs, not 2) -- `appcorr/models/glm46v/axis.py`
    `_vision_stage_cost` prices it -- and for nothing else here.
"""

from ...qwen35.vision.block import ApproxCorrectQwen35VisionBlock
from .attention import ApproxCorrectGlm4vVisionAttention


class ApproxCorrectGlm4vVisionBlock(ApproxCorrectQwen35VisionBlock):
    """See the module docstring: same residual shape as the Qwen3.5 vision block."""

    @classmethod
    def from_stock(cls, blk) -> "ApproxCorrectGlm4vVisionBlock":
        return cls(
            norm1=blk.norm1,
            attn=ApproxCorrectGlm4vVisionAttention.from_stock(blk.attn),
            norm2=blk.norm2,
            mlp=blk.mlp,
        )

    # `forward` / `approx` / `correct` / `correct_rows` are inherited unchanged.
