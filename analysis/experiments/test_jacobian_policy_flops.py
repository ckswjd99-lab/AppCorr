import json
from pathlib import Path

import pytest

from analysis.experiments.jacobian_policy_flops import (
    estimate,
    structured_keep,
)


def test_structured_keep_rounds_to_whole_blocks() -> None:
    assert structured_keep(0, 17) == 0
    assert structured_keep(1, 17) == 1
    assert structured_keep(0.5, 17) == 9 / 17
    assert structured_keep(0.2, 64) == 13 / 64


def test_checked_in_policy_flop_ratios() -> None:
    path = (
        Path(__file__).parent
        / "results"
        / "jacobian_pruning_policy_token50_attnffn.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = estimate(
        payload,
        tokens=261,
        special_tokens=5,
        hidden_size=4096,
        ffn_hidden_size=8192,
        layers=40,
        attention_key_block=16,
        ffn_channel_block=128,
    )

    assert result["approx"]["flops_total"] == 3_547_726_479_360
    policy25 = result["policies"]["attn_ffn_budget_25%"]
    policy50 = result["policies"]["attn_ffn_budget_50%"]
    assert policy25["structured_correction_over_approx"] == pytest.approx(
        0.8004707112
    )
    assert policy50["structured_correction_over_approx"] == pytest.approx(
        0.6551923906
    )
    assert (
        policy25["structured_component_share_of_correction"][
            "ffn_projections"
        ]
        > 0.74
    )
    dense = result["dense_accuracy_oracle"]
    assert dense["policy_branches"]["attn_ffn_budget_25%"][
        "over_approx"
    ] == pytest.approx(1.1047598959)
    assert dense["policy_branches"]["attn_ffn_budget_50%"][
        "over_approx"
    ] == pytest.approx(1.0593823827)
    assert dense["all_policy_eval_over_approx"] == pytest.approx(
        4.8553336387
    )
