"""
progressive_correct.py -- Phase 5: PROGRESSIVE FINALIZATION via the actual cheap first-order
correction on the Qwen2.5-VL vision encoder (the validated `ApproxCorrectQwen25VLVisionTower`
fork), replacing the re-encoding UPPER BOUND used in `progressive.py`.

Mechanism (mirrors `progressive.progressive_finalized_embeds`, but cheap):
  1. approx pass on the coarse BASE image  -> caches per-layer K/V + block-out sums (base_only).
  2. For each band g (top->bottom, ONE SHARED growing cache): correct only band g's merge-groups
     against that cache. At round g the cache holds:
        - bands 1..g-1 : corrected in prior rounds (each against the cache state at ITS round)
        - band g       : recomputed now from true (full) patch values
        - bands g+1..G : still base (stale)
     -> band g's finalized tokens carry the ACCUMULATED bidirectional staleness that the
        re-encoding upper bound does NOT have. The gap between this and the upper bound is exactly
        the cost of using a cheap correction instead of a full re-encode.
  3. merge -> progressive-corrected visual embeds (raster / LLM order), same interface as
     `progressive.progressive_finalized_embeds` so the accuracy driver is unchanged.

The `full` reference and validation encodings come from stock `get_image_features` so the fork is
checked against the unmodified model, never against itself.
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from appcorr.models.qwen25vl.vision.backbone import ApproxCorrectQwen25VLVisionTower


def get_vision_tower(model):
    """Locate the stock vision tower (transformers 5.13: `model.model.visual`)."""
    vt = getattr(getattr(model, "model", model), "visual", None) or getattr(model, "visual", None)
    if vt is None:
        raise AttributeError("could not find the vision tower on the model")
    return vt


def build_tower(model):
    return ApproxCorrectQwen25VLVisionTower(get_vision_tower(model))


@torch.inference_mode()
def encode_pixels(processor, image_np: np.ndarray, device, dtype):
    """Raw uint8 [H,W,3] -> (pixel_values [Npatch, C], grid_thw [1,3])."""
    from PIL import Image
    out = processor.image_processor(images=[Image.fromarray(image_np)], return_tensors="pt")
    return out["pixel_values"].to(device, dtype=dtype), out["image_grid_thw"].to(device)


@torch.inference_mode()
def stock_embeds(model, pixel_values, grid_thw):
    """Stock (unmodified) merged visual tokens in raster/LLM order [Nv, H]."""
    embeds = model.model.get_image_features(pixel_values, grid_thw).pooler_output
    return torch.cat(list(embeds), dim=0)


def _band_token_idx(tower, ctx, group_idx):
    """Permuted-sequence row indices for a band's merge-groups (same map correct_forward uses)."""
    inv = ctx["inv_window_index"]
    dest = inv[group_idx.to(inv.device)]
    unit = tower.spatial_merge_unit
    return (dest.unsqueeze(1) * unit + torch.arange(unit, device=dest.device)).flatten()


@torch.inference_mode()
def progressive_corrected_embeds(model, tower, processor, full_np, base_np, bands, device):
    """Cheap-correction analogue of progressive.progressive_finalized_embeds.
    Returns (prog, full, base) merged embeds [Nv, H], raster order."""
    dtype = model.dtype
    pv_full, grid = encode_pixels(processor, full_np, device, dtype)
    pv_base, grid_b = encode_pixels(processor, base_np, device, dtype)
    assert torch.equal(grid, grid_b), (grid, grid_b)  # same resolution -> identical grid tensors

    ctx = tower.prepare_full_tokens(pv_full, grid)       # full patch embeds + grid tensors (for correct)
    ctx_base = tower.prepare_full_tokens(pv_base, grid)   # base patch embeds (for approx)
    L = len(tower.blocks)

    # (1) approx pass on the base image -> shared cache + base_only merged tokens
    cache = {}
    h_base, cache = tower.approx_forward(ctx_base["hidden_states"], 0, L, ctx_base, cache, "v")
    base_embeds = tower.get_merged_output(h_base, ctx)

    # (2) progressive per-band correction against the ONE growing cache
    h_prog = h_base.clone()
    for band in bands:
        group_idx = torch.arange(band.tok_start, band.tok_end, device=device)
        x_corr, cache = tower.correct_forward(ctx["hidden_states"].clone(), group_idx, 0, L, ctx, cache, "v")
        tok = _band_token_idx(tower, ctx, group_idx)
        h_prog[tok] = x_corr[tok]  # freeze band g's finalized hidden

    prog_embeds = tower.get_merged_output(h_prog, ctx)
    full_embeds = stock_embeds(model, pv_full, grid)     # stock reference (not the fork)
    return prog_embeds, full_embeds, base_embeds


@torch.inference_mode()
def correct_all_one_round(model, tower, processor, full_np, base_np, device):
    """VALIDATION: approx(base) then correct ALL merge-groups in one round must reproduce the stock
    full-image encoding (every token's K/V refreshed -> full bidirectional attention). Returns
    (correct_all_embeds, stock_full_embeds, base_embeds, stock_base_embeds)."""
    dtype = model.dtype
    pv_full, grid = encode_pixels(processor, full_np, device, dtype)
    pv_base, _ = encode_pixels(processor, base_np, device, dtype)
    ctx = tower.prepare_full_tokens(pv_full, grid)
    ctx_base = tower.prepare_full_tokens(pv_base, grid)
    L = len(tower.blocks)

    cache = {}
    h_base, cache = tower.approx_forward(ctx_base["hidden_states"], 0, L, ctx_base, cache, "v")
    base_embeds = tower.get_merged_output(h_base, ctx_base)

    n_groups = ctx["seq_len"] // tower.spatial_merge_unit
    group_all = torch.arange(n_groups, device=device)
    x_corr, cache = tower.correct_forward(ctx["hidden_states"].clone(), group_all, 0, L, ctx, cache, "v")
    correct_all = tower.get_merged_output(x_corr, ctx)

    return (correct_all, stock_embeds(model, pv_full, grid),
            base_embeds, stock_embeds(model, pv_base, grid))
