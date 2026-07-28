from pathlib import Path

import torch

from analysis.experiments.jacobian_policy_imagenet_eval import (
    SUM_KEYS,
    empty_method_stats,
    finish_state,
    load_policies,
    update_metrics,
)
from analysis.experiments.jacobian_support_oracle import (
    select_residual_token_support,
)


def test_checked_in_policy_targets_are_complete() -> None:
    policy_path = (
        Path(__file__).parent
        / "results"
        / "jacobian_pruning_policy_equal_work.json"
    )
    policies, metadata = load_policies(policy_path, [0.25, 0.5, 0.75])
    assert list(policies) == ["prune_25", "prune_50", "prune_75"]
    assert list(metadata) == list(policies)
    assert all(len(schedule) == 40 for schedule in policies.values())


def test_metric_accumulation_and_finalization() -> None:
    state = {
        "methods": {
            "l0_full": empty_method_stats(False),
            "l2_approx": empty_method_stats(True),
        }
    }
    raw_sums = {
        "error_sq": 1.0,
        "reference_sq": 4.0,
        "actual_sq": 1.0,
        "dot": 2.0,
    }
    outputs = {
        "l0_full": {
            "logits": torch.tensor([[9.0, 1.0, 0.0], [0.0, 9.0, 1.0]])
        },
        "l2_approx": {
            "logits": torch.tensor([[8.0, 2.0, 0.0], [9.0, 0.0, 1.0]]),
            **{key: raw_sums for key in SUM_KEYS},
        },
    }
    update_metrics(state, outputs, torch.tensor([0, 1]))
    finished = finish_state(state)

    assert finished["methods"]["l0_full"]["top1_accuracy"] == 1.0
    assert finished["methods"]["l2_approx"]["top1_accuracy"] == 0.5
    assert finished["methods"]["l2_approx"]["l0_top1_match_rate"] == 0.5
    assert (
        finished["methods"]["l2_approx"]["token_feature"][
            "relative_l2_error"
        ]
        == 0.5
    )


def test_residual_token_support_always_keeps_special_prefix() -> None:
    delta = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
    mask = select_residual_token_support(
        delta,
        keep_ratio=0.5,
        always_keep_prefix=2,
    )
    assert mask.tolist() == [[
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]]
    assert int(mask.sum()) == 2 + 4

    prefix_only = select_residual_token_support(
        delta,
        keep_ratio=0,
        always_keep_prefix=2,
    )
    assert prefix_only.tolist() == [[
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]]
