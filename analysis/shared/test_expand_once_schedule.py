import torch

from analysis.experiments.dinov3_expand_once_probe import (
    build_group_dindices,
    build_parent_token_index,
    validate_group_partition,
    validate_parent_group_alignment,
)


def test_g4_checkerboard_splits_every_parent_across_groups():
    groups = build_group_dindices(
        high_hw=(4, 4),
        num_groups=4,
        num_prefix=2,
        device=torch.device("cpu"),
    )
    validate_parent_group_alignment(
        groups,
        low_hw=(2, 2),
        high_hw=(4, 4),
        num_prefix=2,
    )
    assert all(group.numel() == 2 + 4 for group in groups)


def test_parent_lookup_repeats_each_low_patch_four_times():
    parent = build_parent_token_index(
        low_hw=(2, 2),
        high_hw=(4, 4),
        num_prefix=2,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(parent[:2], torch.tensor([0, 1]))
    counts = torch.bincount(parent[2:] - 2, minlength=4)
    torch.testing.assert_close(counts, torch.full((4,), 4, dtype=torch.long))


def test_g4_block_grid_forms_equal_quadrants():
    groups = build_group_dindices(
        high_hw=(4, 4),
        num_groups=4,
        num_prefix=2,
        device=torch.device("cpu"),
        group_strategy="block_grid",
    )
    validate_group_partition(groups, high_hw=(4, 4), num_prefix=2)
    patch_groups = [group[2:] - 2 for group in groups]
    assert [group.tolist() for group in patch_groups] == [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15],
    ]
