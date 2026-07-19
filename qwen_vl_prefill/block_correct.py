"""
block_correct.py -- 2D-block-granularity progressive correction (generalizes the 1D horizontal-band
scheme in progressive_correct.py). Motivated by the spatial-dependence + causal-order findings:
grounding dependence/order sensitivity is 2D-local, so correcting in 2D blocks with a spatial-
NEIGHBOR overlap (refresh the arrived blocks a target block actually depends on) should recover more
of the -2.56pp grounding cheap-correction overhead than 1D trailing bands.

A group is an arbitrary set of merged-token (raster) indices -- the vision fork's correct_forward
takes any group_idx, so 2D blocks (non-contiguous in raster order) work exactly like bands. The LLM
prefill stays raster here (we isolate the vision-correction granularity effect).

  block_groups(grid, P, Q)  -> list of token-index arrays (block-raster order) + block-grid centers
  build_refresh_sets(...)   -> per-step list of group indices to re-correct together (self + overlap)
      policy "trailing": self + the `overlap` immediately-preceding groups (reproduces 1D bands)
      policy "nearest" : self + the `overlap` spatially-nearest ALREADY-ARRIVED groups (2D neighbors)
  block_corrected_embeds(...) -> (prog, full, base) merged embeds, same interface as progressive_correct
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_vl_prefill.progressive_correct import encode_pixels, stock_embeds, _band_token_idx


def block_groups(grid_thw, merge_size, P, Q):
    """P row-splits x Q col-splits of the merged grid -> per-block raster token indices + centers.
    Block-raster order (row-major over blocks). 1D bands == P=G, Q=1."""
    _, hp, wp = [int(x) for x in grid_thw[0].tolist()]
    mh, mw = hp // merge_size, wp // merge_size
    r_edges = [(mh * p) // P for p in range(P + 1)]
    c_edges = [(mw * q) // Q for q in range(Q + 1)]
    groups, centers = [], []
    for bp in range(P):
        for bq in range(Q):
            r0, r1 = r_edges[bp], r_edges[bp + 1]
            c0, c1 = c_edges[bq], c_edges[bq + 1]
            if r1 <= r0 or c1 <= c0:
                continue
            toks = np.array([r * mw + c for r in range(r0, r1) for c in range(c0, c1)], dtype=np.int64)
            groups.append(toks)
            centers.append((bp, bq))
    return groups, np.array(centers, dtype=np.float64)


def build_refresh_sets(centers, overlap, policy):
    """Per-step (block-raster order) list of group indices to re-correct together (always incl self).
    trailing: self + `overlap` immediately-preceding groups. nearest: self + `overlap` spatially-
    nearest already-processed groups (2D neighbors, using block-grid centroid distance)."""
    n = len(centers)
    sets = []
    for i in range(n):
        if policy == "trailing":
            sel = list(range(max(0, i - overlap), i + 1))
        elif policy == "nearest":
            arrived = np.arange(i)
            if overlap > 0 and len(arrived):
                d = np.hypot(*(centers[arrived] - centers[i]).T)
                sel = sorted([i] + list(arrived[np.argsort(d)[:overlap]]))
            else:
                sel = [i]
        else:
            raise ValueError(policy)
        sets.append(sel)
    return sets


@torch.inference_mode()
def setup_correction(model, tower, processor, full_np, base_np, device):
    """Per-image shared work (encode + base approx pass), so a sweep of correction CONFIGS reuses it.
    Returns dict with ctx, base_cache (post-approx), h_base, and the config-independent base/full embeds."""
    dtype = model.dtype
    pv_full, grid = encode_pixels(processor, full_np, device, dtype)
    pv_base, grid_b = encode_pixels(processor, base_np, device, dtype)
    assert torch.equal(grid, grid_b)
    ctx = tower.prepare_full_tokens(pv_full, grid)
    ctx_base = tower.prepare_full_tokens(pv_base, grid)
    L = len(tower.blocks)
    cache = {}
    h_base, cache = tower.approx_forward(ctx_base["hidden_states"], 0, L, ctx_base, cache, "v")
    return {"ctx": ctx, "L": L, "base_cache": cache, "h_base": h_base, "grid": grid,
            "base_embeds": tower.get_merged_output(h_base, ctx),
            "full_embeds": stock_embeds(model, pv_full, grid)}


@torch.inference_mode()
def run_correction_config(tower, setup, groups, refresh_sets, device):
    """Run one correction config from a cloned copy of the shared post-approx base cache."""
    ctx, L = setup["ctx"], setup["L"]
    cache = {k: v.clone() for k, v in setup["base_cache"].items()}
    h_prog = setup["h_base"].clone()
    for i in range(len(groups)):
        gidx = np.unique(np.concatenate([groups[j] for j in refresh_sets[i]]))
        group_idx = torch.as_tensor(gidx, device=device)
        x_corr, cache = tower.correct_forward(ctx["hidden_states"].clone(), group_idx, 0, L, ctx, cache, "v")
        tok = _band_token_idx(tower, ctx, group_idx)
        h_prog[tok] = x_corr[tok]
    return tower.get_merged_output(h_prog, ctx)


@torch.inference_mode()
def block_corrected_embeds(model, tower, processor, full_np, base_np, groups, refresh_sets, device):
    """Progressive correction over arbitrary token-groups with per-step re-refresh sets.
    Returns (prog, full, base) merged embeds [Nv, H], raster order."""
    dtype = model.dtype
    pv_full, grid = encode_pixels(processor, full_np, device, dtype)
    pv_base, grid_b = encode_pixels(processor, base_np, device, dtype)
    assert torch.equal(grid, grid_b)
    ctx = tower.prepare_full_tokens(pv_full, grid)
    ctx_base = tower.prepare_full_tokens(pv_base, grid)
    L = len(tower.blocks)

    cache = {}
    h_base, cache = tower.approx_forward(ctx_base["hidden_states"], 0, L, ctx_base, cache, "v")
    base_embeds = tower.get_merged_output(h_base, ctx)

    h_prog = h_base.clone()
    for i in range(len(groups)):
        gidx = np.unique(np.concatenate([groups[j] for j in refresh_sets[i]]))
        group_idx = torch.as_tensor(gidx, device=device)
        x_corr, cache = tower.correct_forward(ctx["hidden_states"].clone(), group_idx, 0, L, ctx, cache, "v")
        tok = _band_token_idx(tower, ctx, group_idx)
        h_prog[tok] = x_corr[tok]

    prog_embeds = tower.get_merged_output(h_prog, ctx)
    full_embeds = stock_embeds(model, pv_full, grid)
    return prog_embeds, full_embeds, base_embeds
