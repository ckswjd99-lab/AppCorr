"""Does the Qwen3.5 vision fork reproduce the stock tower?

Three checks, cheapest first, each a strict identity rather than an approximation:

  1. `forward`  -- the fork's plain path against stock's block loop. Catches a wrong norm, a
     dropped scaling, a mis-shaped QKV reshape.
  2. `approx`   -- the full-depth approximate pass. Must equal (1): approx differs from forward
     only by what it caches, never by what it computes.
  3. `correct`  -- one round correcting EVERY merge group. Recomputing 100% of tokens is a full
     forward by construction, so it must equal (1) too. This is the check that has caught real
     bugs in every prior fork; the two above mostly catch typos.

bf16 reduction order is shape-dependent, so (3) genuinely cannot be bit-exact against (1) -- it
splits the sequence differently. Everything runs in fp32 to remove that confound rather than
inventing a tolerance to hide behind.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig
from appcorr.models.qwen35.vision.backbone import ApproxCorrectQwen35VisionTower


def main() -> int:
    torch.manual_seed(0)
    # A small tower with the real architectural shape (depth trimmed for speed; depth is not what
    # any of these identities depend on).
    cfg = Qwen3_5MoeVisionConfig(depth=4, hidden_size=128, num_heads=8, intermediate_size=256,
                                 out_hidden_size=128, patch_size=16, spatial_merge_size=2,
                                 temporal_patch_size=2, num_position_embeddings=2304)
    torch.set_default_dtype(torch.float32)
    stock = Qwen3_5MoeVisionModel(cfg).eval()
    for p in stock.parameters():
        p.data.normal_(0, 0.02)

    grid_thw = torch.tensor([[1, 8, 8]])                      # 64 patches -> 16 merge groups
    n_patch = int(grid_thw.prod(-1).sum())
    pixel_values = torch.randn(n_patch, cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size ** 2)

    with torch.no_grad():
        ref = stock(pixel_values, grid_thw).last_hidden_state

    tower = ApproxCorrectQwen35VisionTower(stock).eval()
    ok = True
    with torch.no_grad():
        ctx = tower.prepare_full_tokens(pixel_values, grid_thw)
        n_l = len(tower.blocks)

        # (1) plain forward
        x = ctx["hidden_states"]
        for b in tower.blocks:
            x = b(x, ctx["segment_ranges"], ctx["position_embeddings"])
        d1 = (x - ref).abs().max().item()

        # (2) approx over full depth
        cache = {}
        x2, cache = tower.approx_forward(ctx["hidden_states"], 0, n_l, ctx, cache, "v",
                                         collect_attn_mean=True)
        d2 = (x2 - ref).abs().max().item()

        # (3) correct EVERY merge group -- a full forward by construction
        n_groups = ctx["seq_len"] // tower.spatial_merge_unit
        x3, cache = tower.correct_forward(ctx["hidden_states"], torch.arange(n_groups),
                                          0, n_l, ctx, cache, "v")
        d3 = (x3 - ref).abs().max().item()

        cache = tower.finalize_attn_layermean(cache, "v", n_l)
        am = cache["v_attn_layermean"]

        # (4) THE check that actually exercises correction. Everything above fed the same pixels to
        # both paths, so a `correct()` that silently did nothing would still pass all three -- the
        # exact failure CLAUDE.md warns about ("two conditions agreeing to every digit is the same
        # signal"). Here the approximate pass runs on DEGRADED pixels and the correction round then
        # arrives with the real ones, which is the setting AppCorr is actually used in. Correcting
        # every merge group overwrites every cached K/V, so the result must equal a stock forward on
        # the full-resolution input -- again an identity, not an approximation.
        px_degraded = pixel_values + 0.5 * torch.randn_like(pixel_values)
        ctx_d = tower.prepare_full_tokens(px_degraded, grid_thw)
        cache_b = {}
        x_approx, cache_b = tower.approx_forward(ctx_d["hidden_states"], 0, n_l, ctx_d, cache_b, "b")
        d_floor = (x_approx - ref).abs().max().item()   # must be LARGE -- proves the input differs
        x_corr, cache_b = tower.correct_forward(ctx["hidden_states"], torch.arange(n_groups),
                                                0, n_l, ctx, cache_b, "b")
        d4 = (x_corr - ref).abs().max().item()

    tol = 2e-4  # fp32 accumulated over 4 blocks; a real defect is orders of magnitude larger
    print(f"  ----  degraded approx differs from stock by {d_floor:.3e} (must be >> tol, else the "
          f"correction test is vacuous)")
    if d_floor < 1e-2:
        print("  FAIL  degraded input barely moved the output -- test (4) proves nothing")
        ok = False
    for name, d in (("forward == stock", d1), ("approx == stock", d2), ("correct(all) == stock", d3),
                    ("degraded+correct(all) == stock", d4)):
        flag = "PASS" if d < tol else "FAIL"
        ok &= d < tol
        print(f"  {flag}  {name:<26} max|diff| = {d:.3e}")

    # The score has to be a usable ranking signal, not just present.
    fin = bool(torch.isfinite(am).all())
    spread = (am.max() - am.min()).item()
    print(f"  {'PASS' if fin else 'FAIL'}  attn_layermean finite      shape={tuple(am.shape)}")
    print(f"  {'PASS' if spread > 0 else 'FAIL'}  attn_layermean varies      spread = {spread:.3e}")
    ok &= fin and spread > 0

    print("\n" + ("ALL GATES PASS" if ok else "GATE FAILURE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
