"""Gates for the quarter-wavefront grouping (user scenario 2026-09-03).

The schedule engine (partial-depth interleaved correct + approx advance) is contract-gated
elsewhere; this change only adds (1) a quadrant group assignment and (2) per-level keep-RATIO
resolution. Both are pure functions -- gate them exactly.
"""
import numpy as np

from offload.policies.transmission.ade20k_window_progressive import (
    ADE20KWindowL2L1L0ProgressiveLaplacianPolicy as Policy,
)
from offload.server.model.dinov3_segmentor_m2f import DINOv3SegmentorM2FExecutor


def _structure(grid_h, grid_w):
    return [
        {"row": r, "col": c, "res_level": 0}
        for r in range(grid_h)
        for c in range(grid_w)
    ]


def test_quarter_assignment_covers_and_balances():
    pol = Policy.__new__(Policy)  # no __init__: _precompute_group_assignments is self-contained
    for gh, gw in ((7, 9), (8, 8), (5, 4)):
        st = _structure(gh, gw)
        a = pol._precompute_group_assignments("quarter_cover", st, 4)
        assert len(a) == gh * gw
        assert set(np.unique(a)) == {1, 2, 3, 4}, (gh, gw, np.unique(a))
        # quadrant identity: TL cell -> 1, BR cell -> 4
        assert a[0] == 1
        assert a[-1] == 4
        # balance: no quadrant may deviate from N/4 by more than one row+col band
        counts = np.bincount(a)[1:]
        assert counts.max() - counts.min() <= max(gh, gw) + 2, counts


def test_quarter_assignment_is_spatial_not_ordinal():
    pol = Policy.__new__(Policy)
    gh, gw = 6, 6
    st = _structure(gh, gw)
    a = pol._precompute_group_assignments("quarter_cover", st, 4).reshape(gh, gw)
    assert (a[:3, :3] == 1).all() and (a[:3, 3:] == 2).all()
    assert (a[3:, :3] == 3).all() and (a[3:, 3:] == 4).all()


def test_keep_ratio_resolution_per_level():
    opts = {"token_keep_ratio": 0.2, "l1_token_keep_ratio": 0.5, "l0_token_keep_ratio": 0.3}
    f = DINOv3SegmentorM2FExecutor._token_keep_ratio_for_group
    assert f(opts, group_id=1, l2l1l0_mode=True) == 0.5
    assert f(opts, group_id=3, l2l1l0_mode=True) == 0.3
    assert f(opts, group_id=1, l2l1l0_mode=False) == 0.2
    opts_null = {"token_keep_ratio": 0.2, "l1_token_keep_ratio": None, "l0_token_keep_ratio": None}
    assert f(opts_null, group_id=1, l2l1l0_mode=True) == 0.2


def test_layer_boundaries_are_8_per_group_at_n5():
    from offload.policies.scheduling.ade20k_window_trigger import (
        ADE20KWindowInterleavedPolicy,
    )

    class _Cfg:
        scheduler_kwargs = {"total_layers": 40, "final_full_layers": 0}
        transmission_kwargs = {}

    b = ADE20KWindowInterleavedPolicy._layer_boundaries(_Cfg(), 5)
    assert b == [0, 8, 16, 24, 32, 40], b
