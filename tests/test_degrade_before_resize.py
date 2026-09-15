"""The approx level is built in NATIVE coordinates; only the selected level is scaled.

AGENTS.md "Approx/Correct Contracts", restated with its failure history in
docs/memo/pyramid_degradation_native_vs_canvas.md. Degrading a resized canvas instead of the
original once made COCO's approx-only floor equal its full-transmission ceiling to 1e-4, and on
2026-09-16 it did the same thing to the resolution ladder: `--target-tokens` resized first, so
every floor and adaptive rung -- and every threshold fitted by pscore_dump against that base --
measured the resize rather than the resolution.

The behavioural test below is the reason the rule exists: above native the resize is an upscale
that carries no information, so degrading it strips the upscale instead of real content and the
approximate image comes out far too strong. The source-order tests are the cheap guard, because
the bug is invisible in any single arm's output -- it only shows up as a floor that drifts toward
the ceiling, which reads like a result.
"""
import ast
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.experiments.qwen35_accuracy import degrade  # noqa: E402
from analysis.experiments.qwen_vllm_accuracy import resize_to_tokens  # noqa: E402

FACTOR = 32          # Qwen3.5: patch 16 x merge 2
LEVEL = 2            # the campaign's transmission level: 4x down, back up


def _detail(im):
    """Laplacian variance -- how much high-frequency content survived."""
    a = np.asarray(im.convert("L"), dtype=np.float32)
    k = (a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:])
    return float(k.var())


def _textured(w, h, seed=0):
    """An image with real high-frequency content at every scale (white noise + edges)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    a[:, ::16] = 255          # hard vertical edges the pyramid has to actually remove
    a[::16, :] = 0
    return Image.fromarray(a)


@pytest.mark.parametrize("filt", ["pyr", "box", "bicubic"])
def test_upscaling_target_degrading_the_canvas_keeps_detail_it_should_have_lost(filt):
    """T above native: the wrong order leaks detail into the approximate image."""
    img = _textured(640, 640)                              # 400 tokens at factor 32
    target = 4 * (img.width // FACTOR) * (img.height // FACTOR)   # 4x native -> a 2x upscale

    right = resize_to_tokens(degrade(img, LEVEL, filt), target, FACTOR)
    wrong = degrade(resize_to_tokens(img, target, FACTOR), LEVEL, filt)

    assert right.size == wrong.size, "both orders must land on the same token grid"
    # The upscale added no information, so degrading it removes the upscale rather than content.
    assert _detail(wrong) > 1.2 * _detail(right), (
        f"{filt}: canvas-relative degradation kept {_detail(wrong) / _detail(right):.2f}x the "
        "detail of the contract-compliant order; the floor would be measuring the resize")


def test_degrade_preserves_size_so_both_paths_share_one_token_grid():
    """Why the fix can scale full and degraded independently: degrade() is geometry-neutral."""
    for w, h in ((640, 480), (1023, 769), (2048, 2048)):
        img = _textured(w, h, seed=w)
        assert degrade(img, LEVEL, "pyr").size == img.size
        target = 2048
        assert (resize_to_tokens(img, target, FACTOR).size
                == resize_to_tokens(degrade(img, LEVEL, "pyr"), target, FACTOR).size)


def _order_in(path, func_name):
    """(first degrade line, first resize_to_tokens line) inside `func_name`."""
    tree = ast.parse(open(path).read())
    hits = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == func_name]
    assert hits, f"{func_name} not found in {path}"
    deg = res = None
    for node in ast.walk(hits[0]):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "degrade" and deg is None:
            deg = node.lineno
        if node.func.id == "resize_to_tokens" and res is None:
            res = node.lineno
    return deg, res


def test_driver_degrades_before_it_resizes():
    """`prep` must call degrade() on the native image, then scale onto the target grid."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "analysis", "experiments", "qwen_vllm_accuracy.py")
    deg, res = _order_in(p, "prep")
    assert deg is not None, "prep no longer degrades -- has the pipeline moved?"
    assert res is not None, "prep no longer resizes -- has --target-tokens moved back into build()?"
    assert deg < res, (
        f"qwen_vllm_accuracy.prep resizes at line {res} before degrading at line {deg}: that is "
        "canvas-relative degradation, which invalidates every floor and adaptive arm of the run")


def test_driver_build_hands_prep_a_native_image():
    """The precise thing that regressed on 2026-09-16: `build` resized, so `prep` degraded a canvas.

    Checked separately from the ordering test because it is the version that actually shipped --
    with the resize upstream of prep there is no out-of-order call inside prep to notice.
    """
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "analysis", "experiments", "qwen_vllm_accuracy.py")
    _, res = _order_in(p, "build")
    assert res is None, (
        f"qwen_vllm_accuracy.build resizes at line {res}. build() feeds prep(), which degrades, so "
        "a resize here makes the approximate image canvas-relative -- the exact defect that voided "
        "the resolution ladder. Scale inside prep, after degrade()")


def test_pscore_dump_degrades_before_it_resizes():
    """The same contract on the calibration path: a mis-fitted theta is silent."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "analysis", "experiments", "pscore_dump.py")
    src = open(p).read()
    deg = src.index("base = degrade(")
    res = src.index("img = resize_to_tokens(", src.index("for j, i in enumerate(idxs):"))
    assert deg < res, (
        "pscore_dump resizes before degrading: every theta it fits would be calibrated against "
        "an approximate image the driver never produces")
