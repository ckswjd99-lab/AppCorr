# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .attention import CausalSelfAttention, LinearKMaskedBias, SelfAttention
from .block import CausalSelfAttentionBlock, SelfAttentionBlock
from .ffn_layers import Mlp, SwiGLUFFN
from .fp8_linear import convert_linears_to_fp8
from .jacobian_support import (
    AttentionDelta,
    SupportStatistics,
    attention_delta,
    attention_edge_energy,
    exact_attention_delta,
    exact_swiglu_delta,
    gap_recovery,
    relative_l2_error,
    select_attention_block_support,
    select_ffn_block_support,
    silu_derivative,
    softmax_jvp,
    swiglu_channel_energy,
    swiglu_jvp,
    swiglu_training_free_score,
)
from .layer_scale import LayerScale
from .patch_embed import PatchEmbed
from .rms_norm import RMSNorm
from .rope_position_encoding import RopePositionEmbedding
