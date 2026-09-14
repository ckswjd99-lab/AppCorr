"""CPU gates for the GLM-5.3-Flash vision fork (`appcorr/models/glm53/`).

Everything here runs on the CPU against the REAL `zai-org/GLM-5.3-Flash` visual weights (bf16 in
the checkpoint -- all 347 `model.visual.*` tensors live in shard 62 of 62 and none of them is
FP8-quantised -- upcast to fp32 here so a "bitwise" claim is about the code path and not about
bf16 rounding hiding a difference; the one test that must run in bf16 says so). Nothing here
needs a GPU.

There is no processor fixture. The checkpoint's `processor_config.json` declares a custom
`processor_class` (`Glm5NextProcessor`) with its image-processor config inlined and no
`preprocessor_config.json`, which is exactly why vLLM bypasses `AutoProcessor` and ships its own
port (`vllm/transformers_utils/processors/glm5next.py`; `multimodal.py:610-625` says so). The
tower takes `pixel_values [n_patches, C*T*P*P]` and `grid_thw`, so the gates synthesise those
directly -- the numbers under test are identities between code paths, not image content.

What is gated:

  1. `test_stock_port_matches_hf_vision_model` -- `vision/stock.py`'s dependency-free port of the
     tower against transformers 5.16.1's `Glm5NextVisionModel` on the SAME weights. Skipped on a
     box whose transformers predates `models/glm5_next` (the `appcorr` env here, 5.13.0); run it
     with the served env's python. This is the check that pins everything below to stock.
  2. `test_fork_reference_matches_stock_tower` -- the fork's own one-shot forward
     (`ApproxCorrectGlm5NextVisionTower.reference_forward`: Conv3d patch embed -- no post-conv
     norm, no absolute posemb -- 24 blocks with per-head q/k RMSNorm, merge head) against the
     stock tower's `forward` on the same module objects.
  3. `test_staged_tower_matches_unstaged_reference` -- the 24 block stages walked in four
     `approx_forward` layer ranges (the unified axis's chunked walk) against that reference.
  4. `test_merger_band_slicing_is_bf16_noise` -- `merger(all rows)[g0:g1]` vs `merger(rows of
     groups [g0, g1))` in bf16, judged by the fp32-referenced criterion of
     `analysis/experiments/glm46v_tower_gate.py`'s g1c (does band slicing add error BEYOND the
     bf16 GEMM noise the two paths already differ by?), not by an absolute tolerance.
  5. `test_correct_rows_matches_correct_forward` -- the rows-only correction shortcut against the
     full-stream one, the contract `qwen35/vision/block.py::correct_rows` documents.
  6. `test_groups1_correction_reproduces_full_resolution` -- g=1 has no staleness anywhere, so
     the corrected merge must be the full-resolution forward.
  7. `test_qk_norm_is_load_bearing` -- perturbing `q_norm.weight` must move the fork's output.
     Without it, a fork that silently dropped the per-head norm (the ONE module-level addition
     over GLM-4.6V) would pass 2-6 against a stock tower that dropped it too.
  8. `test_tower_rejects_a_glm46v_tower` -- the constructor's guard.

Epsilons. The served tower runs `norm1/norm2/post_layernorm` at 1e-6 and `q_norm/k_norm` at 1e-5
(vLLM overrides the checkpoint's `vision_config.rms_norm_eps` for the former and hard-codes the
latter); `load_stock_vision_tower(..., vllm_eps=True)` is the default and these gates use it.
`vision/stock.py`'s docstring carries the file:line.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_ID = "zai-org/GLM-5.3-Flash"
DTYPE = torch.float32
GRID = (1, 8, 12)          # 96 patch rows = 24 merge groups -> four g=4 bands of 6
DEPTH = 24


def _snapshot_available() -> bool:
    try:
        from appcorr.models.glm53.vision.backbone import resolve_snapshot
        return os.path.exists(os.path.join(resolve_snapshot(MODEL_ID), "config.json"))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _snapshot_available(),
    reason=f"{MODEL_ID} is not in the local HF cache (set HF_HOME / HF_HUB_OFFLINE)")


def _pixels(seed: int, grid=GRID, dtype=DTYPE):
    """`(pixel_values, grid_thw)`: `[t*h*w, C*T*P*P]` in the layout the patch embed views."""
    from appcorr.models.glm53.vision.backbone import vision_config
    v = vision_config(MODEL_ID)
    t, h, w = grid
    dim = int(v.in_channels) * int(v.temporal_patch_size) * int(v.patch_size) ** 2
    g = torch.Generator().manual_seed(seed)
    return torch.randn(t * h * w, dim, generator=g).to(dtype), torch.tensor([list(grid)])


@pytest.fixture(scope="module")
def stock():
    from appcorr.models.glm53.vision.backbone import load_stock_vision_tower
    return load_stock_vision_tower(MODEL_ID, device="cpu", dtype=DTYPE, prefer_hf=False)


@pytest.fixture(scope="module")
def tower(stock):
    """The fork holds references to the stock submodules -- one copy of the weights, two views."""
    from appcorr.models.glm53.vision.backbone import ApproxCorrectGlm5NextVisionTower
    return ApproxCorrectGlm5NextVisionTower(stock)


@pytest.fixture(scope="module")
def pixels():
    return _pixels(1)


@pytest.fixture(scope="module")
def pixels_base():
    """The "degraded base": the same grid, different content (what `px_base` is on the axis)."""
    px, grid = _pixels(1)
    g = torch.Generator().manual_seed(77)
    return px + 0.3 * torch.randn(px.shape, generator=g).to(px.dtype), grid


def _hf_available() -> bool:
    try:
        import transformers.models.glm5_next.modeling_glm5_next  # noqa: F401
        return True
    except Exception:
        return False


# --- 1: the port against the real HF class ---------------------------------------------------- #

@pytest.mark.skipif(not _hf_available(),
                    reason="this transformers has no models/glm5_next (needs >= 5.16)")
def test_stock_port_matches_hf_vision_model(stock, pixels):
    """`vision/stock.py` vs transformers' own `Glm5NextVisionModel`, same weights, same epsilons.

    Not asserted bitwise: HF's attention goes through `ALL_ATTENTION_FUNCTIONS` (sdpa on a
    `[1, H, T, D]` layout split by `torch.split`) while the port calls
    `scaled_dot_product_attention` per segment on the same rows, and HF's RMSNorm may be a fused
    kernel from the hub. Same arithmetic, possibly a different reduction order -- so this asserts
    fp32 GEMM noise (rel-L2 < 1e-6) and prints the gap."""
    from appcorr.models.glm53.vision.backbone import load_stock_vision_tower
    hf = load_stock_vision_tower(MODEL_ID, device="cpu", dtype=DTYPE, prefer_hf=True)
    assert type(hf).__name__ == "Glm5NextVisionModel", type(hf).__name__
    px, grid = pixels
    with torch.no_grad():
        want = hf(px, grid_thw=grid).pooler_output
        got = stock(px, grid)
    del hf
    assert got.shape == want.shape == (96 // 4, 4096), (got.shape, want.shape)
    rel = float((got - want).norm() / want.norm())
    print(f"\n[1] port vs HF Glm5NextVisionModel: rel-L2 {rel:.3e} "
          f"max_abs {(got - want).abs().max().item():.3e}")
    assert rel < 1e-6, rel


# --- 2 / 3: the tower ------------------------------------------------------------------------- #

def test_fork_reference_matches_stock_tower(tower, stock, pixels):
    px, grid = pixels
    with torch.no_grad():
        want = stock(px, grid)
        got = tower.reference_forward(px, grid)
    assert got.shape == want.shape == (24, 4096)
    print(f"\n[2] fork reference vs stock: bitwise={torch.equal(got, want)} "
          f"max_abs {(got - want).abs().max().item():.3e}  eps={tower.eps()}")
    assert torch.equal(got, want), (got - want).abs().max().item()


def test_staged_tower_matches_unstaged_reference(tower, pixels):
    px, grid = pixels
    with torch.no_grad():
        want = tower.reference_forward(px, grid)
        ctx = tower.prepare_full_tokens(px, grid)
        x, cache = ctx["hidden_states"], {}
        for a, b in ((0, 6), (6, 12), (12, 18), (18, DEPTH)):
            x, cache = tower.approx_forward(x, a, b, ctx, cache, "v")
        got = tower.merger(x)
    print(f"\n[3] staged vs reference: bitwise={torch.equal(got, want)} "
          f"max_abs {(got - want).abs().max().item():.3e}")
    assert torch.equal(got, want), (got - want).abs().max().item()
    # the approximate walk must have left a K/V cache for every layer -- the correction path
    # reads `{tag}_kv` and would otherwise silently rebuild from nothing
    assert all(f"v_layer{i}_kv" in cache for i in range(DEPTH))
    assert "pos_embeds" not in ctx, "this tower has no absolute position embedding"


# --- 4: the merge head ------------------------------------------------------------------------ #

def _bands(n_groups: int, groups: int):
    edges = [round(k * n_groups / groups) for k in range(groups + 1)]
    return list(zip(edges[:-1], edges[1:]))


def test_merger_band_slicing_is_bf16_noise(tower, pixels):
    """`merger(all)[g0:g1]` vs `merger(rows of [g0, g1))` in bf16 -- the dtype the tower serves.

    The merger is four GEMMs + a LayerNorm + a GELU with bf16 re-rounding between stages, so two
    bf16 paths differ by compounded rounding whenever the BLAS picks a different kernel for the
    band's M. The question is whether band slicing adds error BEYOND that, so both bf16 paths are
    compared to an fp32 truth computed from the same rows and the band path must be no worse
    (max and mean within 2x) and both must sit at bf16 noise level (<= 2 ULP of the tensor's max
    magnitude). Row mixing would blow every bound. This is `glm46v_tower_gate.py` g1c's rule."""
    import copy
    import math
    px, grid = pixels
    with torch.no_grad():
        ctx = tower.prepare_full_tokens(px, grid)
        x32, cache = ctx["hidden_states"], {}
        for i in range(DEPTH):
            x32, cache = tower.approx_forward(x32, i, i + 1, ctx, cache, "v")
        x = x32.to(torch.bfloat16)
        merger16 = copy.deepcopy(tower.merger).to(torch.bfloat16)
        merger32 = tower.merger
        ref32 = merger32(x.float())
        merged_all = merger16(x)
    n_groups = x.shape[0] // tower.spatial_merge_unit
    scale = float(ref32.abs().max())
    ulp_top = 2.0 ** (math.floor(math.log2(max(scale, 2.0 ** -126))) - 7)
    e_all_max = float((merged_all.float() - ref32).abs().max())
    e_all_mean = float((merged_all.float() - ref32).abs().mean())
    e_band_max = e_band_mean = worst = 0.0
    nb = 0
    with torch.no_grad():
        for groups in (2, 3, 4, 8):
            for g0, g1 in _bands(n_groups, groups):
                out = merger16(x[g0 * 4:g1 * 4])
                ref = merged_all[g0:g1]
                nb += int(not bool(torch.equal(out.view(torch.int16), ref.view(torch.int16))))
                worst = max(worst, float((out.float() - ref.float()).abs().max()))
                d = (out.float() - ref32[g0:g1]).abs()
                e_band_max = max(e_band_max, float(d.max()))
                e_band_mean = max(e_band_mean, float(d.mean()))
    print(f"\n[4] merger bands (bf16): non_bitwise_bands={nb} max_abs_bf16_vs_bf16={worst:.4g} "
          f"ref32_scale={scale:.4g} ulp@scale={ulp_top:.4g} "
          f"err_all(max/mean)={e_all_max:.4g}/{e_all_mean:.4g} "
          f"err_band(max/mean)={e_band_max:.4g}/{e_band_mean:.4g}")
    if nb:
        assert (e_band_max <= 2.0 * max(e_all_max, ulp_top)
                and e_band_mean <= 2.0 * max(e_all_mean, 1e-9)
                and e_band_max <= 2.0 * ulp_top), \
            (f"band err max {e_band_max:.4g} mean {e_band_mean:.4g} vs all-rows max "
             f"{e_all_max:.4g} mean {e_all_mean:.4g}, ulp@scale {ulp_top:.4g}")


def test_merger_rejects_partial_merge_groups(tower):
    with pytest.raises(ValueError):
        tower.merger(torch.zeros(6, int(tower.config.hidden_size)))


# --- 5 / 6: correction ------------------------------------------------------------------------ #

def test_correct_rows_matches_correct_forward(tower, pixels, pixels_base):
    """`correct_rows` (carry only the corrected rows) against `correct_forward` (carry the whole
    [T, D] stream and scatter): same rows, same values. Shapes differ between the two GEMM
    batches, so single-threaded for the bitwise claim."""
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

            _, cache_a = tower.approx_forward(ctx_base["hidden_states"], 0, DEPTH, ctx_base, {}, "v")
            rows, _ = tower.correct_rows(ctx_full["hidden_states"], gidx, ctx_full, cache_a, "v",
                                         span=(g0 * 4, g1 * 4))

            _, cache_b = tower.approx_forward(ctx_base["hidden_states"], 0, DEPTH, ctx_base, {}, "v")
            mask = torch.zeros(gctx["seq_len"], dtype=torch.bool)
            mask[g0 * 4:g1 * 4] = True
            stream = torch.where(mask.unsqueeze(-1), ctx_full["hidden_states"],
                                 ctx_base["hidden_states"])
            full, _ = tower.correct_forward(stream, gidx, 0, DEPTH, ctx_full, cache_b, "v")
        gap = (rows - full[g0 * 4:g1 * 4]).abs().max().item()
        print(f"\n[5] correct_rows vs correct_forward[rows]: "
              f"bitwise={torch.equal(rows, full[g0 * 4:g1 * 4])} max_abs {gap:.3e}")
        assert torch.equal(rows, full[g0 * 4:g1 * 4]), gap
    finally:
        torch.set_num_threads(threads)


def test_groups1_correction_reproduces_full_resolution(tower, pixels, pixels_base):
    """g=1: one band, corrected after everything arrived -- no staleness anywhere, so the merged
    output must be the full-resolution tower forward. Exact in exact arithmetic; the correction
    runs a different sequence of kernels (cached K/V, gathered rows) from the one-shot forward,
    so this asserts a relative tolerance and prints the gap rather than claiming bitwise."""
    px, grid = pixels
    px_base, _ = pixels_base
    with torch.no_grad():
        want = tower.reference_forward(px, grid)
        gctx = tower.prepare_grid(grid, px.device)
        ctx_full = tower.prepare_full_tokens(px, grid, gctx)
        ctx_base = tower.prepare_full_tokens(px_base, grid, gctx)
        n_groups = gctx["seq_len"] // tower.spatial_merge_unit
        _, cache = tower.approx_forward(ctx_base["hidden_states"], 0, DEPTH, ctx_base, {}, "v")
        rows, _ = tower.correct_rows(ctx_full["hidden_states"], torch.arange(n_groups), ctx_full,
                                     cache, "v", span=(0, gctx["seq_len"]))
        got = tower.merger(rows)
    rel = float((got - want).norm() / want.norm())
    print(f"\n[6] g=1 identity: rel-L2 {rel:.3e} max_abs {(got - want).abs().max().item():.3e}")
    assert rel < 1e-5, (rel, (got - want).abs().max().item())


# --- 7 / 8: the fork's own additions ----------------------------------------------------------- #

def test_qk_norm_is_load_bearing(tower, pixels):
    """The per-head q/k RMSNorm is the ONE module-level addition over the GLM-4.6V fork. If the
    fork dropped it, gates 2-6 would still pass (they compare against a stock tower that carries
    the same modules but would never be asked for them). Perturb `q_norm.weight` and the fork's
    output must move; the reference is the fork itself, so this is a direct test of the wiring
    rather than of the value."""
    px, grid = pixels
    w = tower.blocks[0].attn.q_norm.weight
    with torch.no_grad():
        before = tower.reference_forward(px, grid)
        saved = w.detach().clone()
        w.add_(0.05)
        after = tower.reference_forward(px, grid)
        w.copy_(saved)
        restored = tower.reference_forward(px, grid)
    moved = float((after - before).abs().max())
    print(f"\n[7] q_norm perturbation moves the output by max_abs {moved:.4g}")
    assert moved > 1e-3, moved
    assert torch.equal(restored, before)


def test_tower_rejects_a_glm46v_tower(stock):
    from appcorr.models.glm53.vision.backbone import ApproxCorrectGlm5NextVisionTower
    stock.post_conv_layernorm = torch.nn.Identity()
    try:
        with pytest.raises(ValueError, match="post_conv_layernorm"):
            ApproxCorrectGlm5NextVisionTower(stock)
    finally:
        del stock.post_conv_layernorm
