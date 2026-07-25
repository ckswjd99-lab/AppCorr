"""Deterministic masked-softmax checks for pi0-FAST Gemma vision/language score collection."""

import torch
import torch.nn as nn

from appcorr.models.pi0fast.gemma_prefill_layer import ApproxCorrectGemmaAttention
from appcorr.models.pi0fast.progressive_model import Pi0FastProgressiveModel


def main():
    torch.manual_seed(7)
    hidden_size = 4
    head_dim = 2
    num_heads = 2
    num_kv_heads = 1
    attention = ApproxCorrectGemmaAttention(
        nn.Linear(hidden_size, num_heads * head_dim, bias=False),
        nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False),
        nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False),
        nn.Linear(num_heads * head_dim, hidden_size, bias=False),
        num_heads,
        num_kv_heads,
        head_dim,
        head_dim ** -0.5,
    )
    x = torch.randn(1, 4, hidden_size)
    cos = torch.ones(1, 4, head_dim)
    sin = torch.zeros(1, 4, head_dim)
    allowed = torch.tensor(
        [
            [
                [True, True, True, False],
                [True, True, True, False],
                [True, True, True, False],
                [False, False, False, False],
            ]
        ]
    )
    mask = torch.where(allowed[:, None], 0.0, -2.3819763e38)
    query_indices = torch.tensor([0, 1])
    language_query_indices = torch.tensor([2])
    key_indices = torch.tensor([0, 1])
    cache = {}

    attention.approx(
        x,
        cache,
        "test",
        cos,
        sin,
        causal=False,
        attn_mask=mask,
        score_query_groups={
            "vision": query_indices,
            "language": language_query_indices,
        },
        score_key_indices=key_indices,
    )

    q, k, _ = attention._project_heads(x)
    k = k.repeat_interleave(attention.num_key_value_groups, dim=1)
    logits = (
        q[:, :, query_indices].float() @ k.float().transpose(-2, -1)
    ) * attention.scaling
    logits = logits + mask[:, :, query_indices].float()
    expected = logits.softmax(dim=-1)[..., key_indices].mean(dim=(1, 2))
    actual = cache["test_received_attn_vision"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    language_logits = (
        q[:, :, language_query_indices].float() @ k.float().transpose(-2, -1)
    ) * attention.scaling
    language_logits = language_logits + mask[:, :, language_query_indices].float()
    expected_language = (
        language_logits.softmax(dim=-1)[..., key_indices].mean(dim=(1, 2))
    )
    torch.testing.assert_close(
        cache["test_received_attn_language"],
        expected_language,
        rtol=0,
        atol=0,
    )
    assert actual.shape == (1, 2)
    assert torch.isfinite(actual).all()
    assert cache["test_received_attn_language"].shape == (1, 2)
    assert torch.isfinite(cache["test_received_attn_language"]).all()
    vit_pscore = torch.tensor([0.2, 0.1, 0.4, 0.3])
    llm_pscore = torch.tensor([0.1, 0.4, 0.2, 0.3])
    weight_zero = Pi0FastProgressiveModel._fuse_vision_pscore(
        vit_pscore,
        llm_pscore,
        0,
    )
    torch.testing.assert_close(weight_zero, vit_pscore, rtol=0, atol=0)
    print("PASS: Gemma received-attention collector matches explicit masked softmax")


if __name__ == "__main__":
    main()
