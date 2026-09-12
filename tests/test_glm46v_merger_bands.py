"""CPU gates for the GLM-4.6V vision fork (`appcorr/models/glm46v/`).

Everything here runs on the CPU against the REAL `zai-org/GLM-4.6V-FP8` visual weights (bf16 in
the checkpoint, upcast to fp32 here so a "bitwise" claim is about the code path and not about
bf16 rounding hiding a difference). Nothing here needs a GPU; the checks that do -- served
stock-vs-HF first token, and the same identities on CUDA kernels -- live in
`analysis/experiments/glm46v_tower_gate.py` and are NOT run by pytest.

What is gated:

  1. `test_fork_reference_matches_stock_tower` -- the fork's own one-shot forward
     (`ApproxCorrectGlm4vVisionTower.reference_forward`: Conv3d patch embed ->
     post_conv_layernorm -> bicubic-interpolated absolute posemb -> 24 blocks -> merge head)
     against the STOCK `Glm4vMoeVisionModel.forward` on the same module objects. This is the
     check that the pre-stage order, the rotary tables and the merge head were ported right;
     everything below it is relative to a reference that this pins to stock.
  2. `test_staged_tower_matches_unstaged_reference` -- the 24 block stages walked in four
     `approx_forward` layer ranges (the unified axis's chunked walk) against that reference.
  3. `test_merger_band_slicing_is_bitwise` -- `merger(all rows)[g0:g1] == merger(rows of groups
     [g0, g1))`, the invariant the whole per-band merge rests on (`qwen_vl_axis.py` ~567, ~768).
  4. `test_correct_rows_matches_correct_forward` -- the rows-only correction shortcut against the
     full-stream one, the contract `qwen35/vision/block.py::correct_rows` documents.
  5. `test_groups1_correction_reproduces_full_resolution` -- approx on the degraded base, then
     correct EVERY group in one band: g=1 has no staleness anywhere, so the merged result must be
     the full-resolution forward. The in-process form of the GPU G2 identity gate.
  6. `test_prompt_layout_*` / `test_positions_fast_matches_get_rope_index` -- processor-only
     checks of the GLM prompt (`[gMASK]<sop>`, the `<|begin_of_image|>` / `<|end_of_image|>`
     sentinels flanking the `<|image|>` run, `/nothink`) and of the inherited M-RoPE closed form
     against `Glm4vMoeModel.get_rope_index`.

**Thread count and "bitwise" (measured on this box, 2026-09-12).** Identities between calls of
the SAME shape are bitwise at any thread count. The band-slicing identity compares GEMMs of
DIFFERENT M (a band of `n` merge groups is an `n`-row GEMM), and oneDNN picks its kernel and its
partitioning per M, so:

  * single-threaded, `merger(band)` is bitwise `merger(all)[band]` for every band of **4 or more
    merge groups**; M in {1, 2, 3} takes a small-M / GEMV path and differs;
  * multi-threaded (72 here) most M differ, because the reduction is split over threads by M;
  * either way the worst gap over all 24-group sub-bands is **3.6e-6 absolute in fp32**
    (~7e-7 relative), i.e. GEMM-kernel noise and not a structural dependency on the other rows.

The structural half of the claim -- that nothing MIXES rows across merge groups -- is the part
that has to be exact, and it is asserted on its own
(`test_merge_head_prefix_stages_are_bitwise_at_any_thread_count`: `post_layernorm` and the 2x2
stride-2 downsample, the only two stages that could, are bitwise at any thread count and any M).
The tests that want the bitwise form set `torch.set_num_threads(1)` around the comparison; every
campaign band is far larger than 3 groups.
"""
import os

import numpy as np
import pytest
import torch
from PIL import Image

MODEL_ID = "zai-org/GLM-4.6V-FP8"
DTYPE = torch.float32


def _snapshot_available() -> bool:
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(MODEL_ID, "config.json")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _snapshot_available(),
    reason=f"{MODEL_ID} is not in the local HF cache (set HF_HOME / HF_HUB_OFFLINE)")


def _image(seed: int = 1, h: int = 120, w: int = 160) -> Image.Image:
    return Image.fromarray((np.random.RandomState(seed).rand(h, w, 3) * 255).astype("uint8"))


@pytest.fixture(scope="module")
def tower_and_stock():
    """One copy of the weights for the whole module: the fork holds references to the stock
    submodules, so `stock` and `tower` are the same ~860M parameters seen two ways."""
    from appcorr.models.glm46v.vision.backbone import (
        ApproxCorrectGlm4vVisionTower, load_stock_vision_tower)
    stock = load_stock_vision_tower(MODEL_ID, device="cpu", dtype=DTYPE)
    return ApproxCorrectGlm4vVisionTower(stock), stock


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def pixels(processor):
    """(pixel_values, grid_thw) of a 120x160 image -> an 8 x 12 patch grid = 96 rows = 24 merge
    groups, which splits four ways into the campaign's g=4 bands."""
    enc = processor(text=["x"], images=[_image()], return_tensors="pt")
    px, grid = enc["pixel_values"].to(DTYPE), enc["image_grid_thw"]
    assert tuple(grid[0].tolist()) == (1, 8, 12), grid
    assert px.shape[0] == 96
    return px, grid


# --- 1 / 2: the tower ------------------------------------------------------------------------ #

def test_fork_reference_matches_stock_tower(tower_and_stock, pixels):
    tower, stock = tower_and_stock
    px, grid = pixels
    with torch.no_grad():
        want = stock(px, grid_thw=grid).pooler_output
        got = tower.reference_forward(px, grid)
    assert got.shape == want.shape == (24, 4096)
    assert torch.equal(got, want), (got - want).abs().max().item()


def test_staged_tower_matches_unstaged_reference(tower_and_stock, pixels):
    tower, _ = tower_and_stock
    px, grid = pixels
    with torch.no_grad():
        want = tower.reference_forward(px, grid)
        ctx = tower.prepare_full_tokens(px, grid)
        x, cache = ctx["hidden_states"], {}
        for a, b in ((0, 6), (6, 12), (12, 18), (18, 24)):
            x, cache = tower.approx_forward(x, a, b, ctx, cache, "v")
        got = tower.merger(x)
    assert torch.equal(got, want), (got - want).abs().max().item()
    # the approximate walk must have left a K/V cache for every layer -- the correction path
    # reads `{tag}_kv` and would otherwise silently rebuild from nothing
    assert all(f"v_layer{i}_kv" in cache for i in range(24))


# --- 3: the merge head ----------------------------------------------------------------------- #

def _bands(n_groups: int, groups: int):
    edges = [round(k * n_groups / groups) for k in range(groups + 1)]
    return list(zip(edges[:-1], edges[1:]))


def test_merge_head_prefix_stages_are_bitwise_at_any_thread_count(tower_and_stock, pixels):
    """`post_layernorm` (per row) and the 2x2 stride-2 downsample (per merge group) are the only
    stages that could mix rows across groups. They do not, at any thread count."""
    tower, _ = tower_and_stock
    head = tower.merger
    torch.manual_seed(0)
    x = torch.randn(96, 1536, dtype=DTYPE) * 0.5

    def conv(v):
        v = v.view(-1, 2, 2, v.shape[-1]).permute(0, 3, 1, 2)
        return head.downsample(v).view(-1, head.out_hidden_size)

    with torch.no_grad():
        ln_all = head.post_layernorm(x)
        conv_all = conv(ln_all)
        for g0, g1 in _bands(24, 4) + [(3, 7), (0, 24)]:
            rows = slice(g0 * 4, g1 * 4)
            assert torch.equal(head.post_layernorm(x[rows]), ln_all[rows]), (g0, g1)
            assert torch.equal(conv(ln_all[rows]), conv_all[g0:g1]), (g0, g1)


def test_merger_band_slicing_is_bitwise(tower_and_stock, pixels):
    """The invariant the per-band merge rests on, on the REAL last-layer activations rather than
    on random rows: merging a band alone gives the same rows as merging everything and slicing."""
    tower, _ = tower_and_stock
    px, grid = pixels
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)          # see the module docstring: GEMM partitioning by M
        with torch.no_grad():
            ctx = tower.prepare_full_tokens(px, grid)
            x, cache = tower.approx_forward(ctx["hidden_states"], 0, 24, ctx, {}, "v")
            n_groups = ctx["seq_len"] // tower.spatial_merge_unit
            merged_all = tower.merger(x)
            assert merged_all.shape == (n_groups, 4096)
            # Every band of >= 4 merge groups, on and off the `_bands` boundaries: bitwise.
            spans = [b for g in (2, 3, 4, 6) for b in _bands(n_groups, g)]
            spans += [(3, 7), (0, n_groups), (5, 9), (n_groups - 4, n_groups)]
            for g0, g1 in spans:
                assert g1 - g0 >= 4
                assert torch.equal(tower.merger(x[g0 * 4:g1 * 4]), merged_all[g0:g1]), (g0, g1)
            # Bands of 1-3 groups take a small-M GEMM path (module docstring): exact in
            # arithmetic, not bitwise. Pinned as a tolerance so a real row-mixing bug -- which
            # would move a whole feature, not its last bits -- still fails here.
            worst = 0.0
            for g0, g1 in ((0, 1), (11, 13), (7, 10), (n_groups - 1, n_groups)):
                band = tower.merger(x[g0 * 4:g1 * 4])
                worst = max(worst, (band - merged_all[g0:g1]).abs().max().item())
            assert worst < 1e-5, worst
    finally:
        torch.set_num_threads(threads)


def test_merger_rejects_partial_merge_groups(tower_and_stock):
    tower, _ = tower_and_stock
    with pytest.raises(ValueError, match="merge groups"):
        tower.merger(torch.zeros(6, 1536, dtype=DTYPE))


# --- 4 / 5: the approx / correct plumbing ---------------------------------------------------- #

def _degrade(img: Image.Image, level: int = 2) -> Image.Image:
    w, h = img.size
    f = 2 ** level
    return img.resize((max(1, w // f), max(1, h // f)), Image.BOX).resize((w, h), Image.BICUBIC)


@pytest.fixture(scope="module")
def pixels_base(processor):
    enc = processor(text=["x"], images=[_degrade(_image())], return_tensors="pt")
    return enc["pixel_values"].to(DTYPE), enc["image_grid_thw"]


def test_correct_rows_matches_correct_forward(tower_and_stock, pixels, pixels_base):
    """`correct_rows` (carry only the corrected rows) against `correct_forward` (carry the whole
    [T, D] stream and scatter): same rows, same values. Shapes differ between the two GEMM
    batches, so single-threaded for the bitwise claim."""
    tower, _ = tower_and_stock
    px, grid = pixels
    px_base, _ = pixels_base
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.no_grad():
            gctx = tower.prepare_grid(grid, px.device)
            ctx_full = tower.prepare_full_tokens(px, grid, gctx)
            ctx_base = tower.prepare_full_tokens(px_base, grid, gctx)
            n_groups = gctx["seq_len"] // tower.spatial_merge_unit
            g0, g1 = _bands(n_groups, 4)[1]
            gidx = torch.arange(g0, g1)

            _, cache_a = tower.approx_forward(ctx_base["hidden_states"], 0, 24, ctx_base, {}, "v")
            rows, _ = tower.correct_rows(ctx_full["hidden_states"], gidx, ctx_full, cache_a, "v",
                                         span=(g0 * 4, g1 * 4))

            _, cache_b = tower.approx_forward(ctx_base["hidden_states"], 0, 24, ctx_base, {}, "v")
            stream = torch.where(
                torch.zeros(gctx["seq_len"], dtype=torch.bool).index_fill_(
                    0, torch.arange(g0 * 4, g1 * 4), True).unsqueeze(-1),
                ctx_full["hidden_states"], ctx_base["hidden_states"])
            full, _ = tower.correct_forward(stream, gidx, 0, 24, ctx_full, cache_b, "v")
        assert torch.equal(rows, full[g0 * 4:g1 * 4]), \
            (rows - full[g0 * 4:g1 * 4]).abs().max().item()
    finally:
        torch.set_num_threads(threads)


def test_groups1_correction_reproduces_full_resolution(tower_and_stock, pixels, pixels_base):
    """g=1: one band, corrected after everything arrived -- no staleness anywhere, so the merged
    output must be the full-resolution tower forward. Exact in exact arithmetic; the correction
    runs a different sequence of kernels (cached K/V, gathered rows) from the one-shot forward,
    so this asserts a relative tolerance and prints the gap rather than claiming bitwise."""
    tower, _ = tower_and_stock
    px, grid = pixels
    px_base, _ = pixels_base
    with torch.no_grad():
        want = tower.reference_forward(px, grid)
        gctx = tower.prepare_grid(grid, px.device)
        ctx_full = tower.prepare_full_tokens(px, grid, gctx)
        ctx_base = tower.prepare_full_tokens(px_base, grid, gctx)
        n_groups = gctx["seq_len"] // tower.spatial_merge_unit
        _, cache = tower.approx_forward(ctx_base["hidden_states"], 0, 24, ctx_base, {}, "v")
        rows, _ = tower.correct_rows(ctx_full["hidden_states"], torch.arange(n_groups), ctx_full,
                                     cache, "v", span=(0, gctx["seq_len"]))
        got = tower.merger(rows)
    rel = (got - want).norm() / want.norm()
    assert rel < 1e-5, (rel.item(), (got - want).abs().max().item())


# --- 6: the prompt and the positions (processor only) ---------------------------------------- #

class _RopeStub:
    """`Glm4vMoeModel.get_rope_index` reads `self.config` and calls `self.get_vision_position_ids`
    and nothing else, so the reference can be evaluated without instantiating a 106B model."""

    def __init__(self, config):
        from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeModel
        self.config = config
        self._cls = Glm4vMoeModel

    def get_vision_position_ids(self, *a, **kw):
        return self._cls.get_vision_position_ids(self, *a, **kw)

    def get_rope_index(self, *a, **kw):
        return self._cls.get_rope_index(self, *a, **kw)


@pytest.fixture(scope="module")
def prompt(processor):
    msgs = [{"role": "user", "content": [{"type": "image", "image": _image()},
                                         {"type": "text", "text": "What is shown?"}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                         return_dict=True, return_tensors="pt",
                                         enable_thinking=False)


def test_prompt_layout_sentinels_and_prefix(prompt):
    ids = prompt["input_ids"][0]
    assert ids[:2].tolist() == [151331, 151333], "mandatory [gMASK]<sop> prefix"
    pos = (ids == 151363).nonzero(as_tuple=True)[0]
    lo, n = int(pos[0]), int(pos.numel())
    assert int(pos[-1]) - lo == n - 1, "one contiguous <|image|> run"
    assert int(ids[lo - 1]) == 151339 and int(ids[lo + n]) == 151340, "sentinels flank the run"
    assert 151360 in ids.tolist(), "enable_thinking=False appends /nothink"
    assert int(ids.shape[0]) - (lo + n) >= 2, "a non-empty text suffix plus the held-back row"
    t, h, w = (int(v) for v in prompt["image_grid_thw"][0].tolist())
    assert n == t * h * w // 4, "one placeholder per merge group"
    assert "mm_token_type_ids" in prompt


def test_positions_fast_matches_get_rope_index(processor, prompt):
    """The inherited `_positions_fast` closed form against transformers' own `get_rope_index`.
    Exercised through the axis class with only the pieces of it these two methods touch."""
    from transformers import AutoConfig

    from appcorr.models.glm46v.axis import Glm46VAxis
    config = AutoConfig.from_pretrained(MODEL_ID)
    axis = Glm46VAxis.__new__(Glm46VAxis)       # no tower: neither method touches one
    axis.cfg = config
    axis.image_token_id = int(config.image_token_id)
    axis.processor = processor
    axis.positions_mode = "fast"

    ids = prompt["input_ids"]
    lo, n = axis._image_token_run(ids)          # the GLM override: sentinels + suffix checked
    pos_fast, delta_fast = axis._positions_fast(prompt, image_run=(lo, n))

    stub = _RopeStub(config)
    pos_ref, delta_ref = stub.get_rope_index(ids, prompt["mm_token_type_ids"],
                                             image_grid_thw=prompt["image_grid_thw"])
    assert torch.equal(pos_fast, pos_ref), (pos_fast - pos_ref).abs().max().item()
    assert delta_fast == int(delta_ref.flatten()[0])


def test_image_token_run_rejects_a_missing_end_sentinel(processor, prompt):
    from transformers import AutoConfig

    from appcorr.models.glm46v.axis import Glm46VAxis
    config = AutoConfig.from_pretrained(MODEL_ID)
    axis = Glm46VAxis.__new__(Glm46VAxis)
    axis.cfg = config
    axis.image_token_id = int(config.image_token_id)

    ids = prompt["input_ids"].clone()
    pos = (ids[0] == 151363).nonzero(as_tuple=True)[0]
    ids[0, int(pos[-1]) + 1] = 9999            # clobber <|end_of_image|>
    with pytest.raises(ValueError, match="end_of_image"):
        axis._image_token_run(ids)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v", "-s"]))
