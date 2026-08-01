"""
reverse_order_correct.py -- Phase D: bottom-up depth-staggered LLM prefill correction.

Motivation (mirrors DINOv3's L2-L1-L0 tail interleaving, ported to Qwen2.5-VL's causal LLM prefill):
the current scheme (`progressive_correct.py`) makes the VISION TOWER progressive (bidirectional ViT,
per-band cheap correction) but leaves the causal LLM prefill STOCK EXACT (append-only chunked
prefill, each visual group prefilled once, in raster/top-to-bottom order, for free -- no LLM-level
approximation at all). All of the -3.12pp RefCOCO grounding cost measured there comes from the
VISION TOWER's OWN bidirectional correction staleness (Phase 5b: -0.56pp inherent finalization +
-2.56pp cheap-correction overhead), never from the LLM.

This experiment asks a different question: what if the LLM prefill is ALSO made approximate-then-
correct, with corrections staggered in REVERSE spatial order (bottom band first, top band last)
against progressively deeper LLM checkpoints? THE CAUSAL MASK / TOKEN SEQUENCE POSITION ORDER DOES
NOT CHANGE -- groups still sit at raster positions pre_text < g0(top) < g1 < g2 < g3(bottom) <
post_text in the sequence, so g0 is still structurally blind to g1-g3 regardless of processing order
(that is a property of the fixed causal mask, not of scheduling). What changes is WHICH group's LLM
contribution is finalized (K/V written) at WHICH wall-clock step, and how deep each correction goes:

  1. base pass: base (low-res, see `native_pyramid` below) vision tokens for ALL G groups + exact
     pre/post text, threaded through the causal LLM's first `L//G` layers (full-sequence `approx()`
     at each of those layers, one call per layer, matching monolithic-prefill's own per-layer
     mechanism -- see `prefill.py`'s docstring on why chunked/staged calls reproduce monolithic
     prefill given identical embeddings+position_ids).
  2. group G-1 (bottom band, arrives first): vision-encode it exactly (existing per-band ViT
     `correct_forward`, UNCHANGED -- this is the "vision encoder does the same progressive encoding
     as now" requirement) -> correct its LLM tokens through layers [0, L//G) -> merge into the
     running frontier -> advance the WHOLE sequence (`approx()`) to depth 2L//G.
  3. group G-2: correct [0, 2L//G) -> advance to 3L//G. ... group 1: correct [0, (G-1)L//G) ->
     advance to L. group 0 (top band, arrives last): correct [0, L).
  4. Because group 0's correction only updates group 0's OWN K/V (nothing downstream is retroactively
     fixed), and `post_text` (the referring-expression query + generation prefix, which sits AFTER
     every visual group) attends to group 0's K/V at every layer, post_text's own hidden states --
     computed during every earlier "advance" step using group 0's still-base K/V -- are now stale.
     post_text therefore gets ONE additional correction pass, [0, L), COMBINED with group 0's own
     final round (same `token_idx` batch: `sort(group0_idx ++ post_text_idx)`) -- cheaper than, and
     numerically equivalent to, re-correcting post_text after every round (nothing reads post_text's
     hidden state before decoding starts).

Per-group correction mechanism (mirrors `progressive_correct.progressive_corrected_embeds`'s vision-
side pattern, now applied to the LLM decoder stack): EACH group's correction round is a SEPARATE,
SELF-CONTAINED sweep that restarts from the embedding-level tensor `x0` (this group's fresh embedding
spliced in at its own token_idx, base vision elsewhere, exact text always) and threads that ONE
tensor through `layer[i].correct(...)` for i in [0, depth). Only the round's OWN token_idx slice of
the final output is kept (merged into the persistent `x_frontier` accumulator); every other position
in that round's output is discarded. This is safe because `correct()`'s attention computation reads
a position's context SOLELY from the persistent, growing `cache_feature[f"{tag}_kv"]` tensor (which
each earlier group's round already wrote into), never from `x` itself -- `x`'s value at non-token_idx
positions only feeds the (here-unused) `blocks_out_sum` full-tensor reconstruction. Concretely this
means a later-arriving group's correction round correctly attends to every EARLIER group's true,
already-corrected K/V (written into the shared cache by that group's own round), while an
EARLIER-arriving (but structurally LATER-position) group is permanently based on whatever context
existed for the positions before it AT THE TIME it was corrected -- this is the scheme's actual
accuracy cost, and it is real: unlike the current top-down scheme (append-only causal chunking,
provably bit-exact vs. monolithic prefill, see `equivalence_test.py`), THIS schedule is NOT
exactness-preserving in general.

Exactness DOES hold in one degenerate case, and it is the sanity gate this module ships
(`assert_num_groups_1_is_exact`): with `num_groups=1` there is only one (whole-image) group, so the
base pass's 0-depth "start" is immediately fully superseded by that group's own [0, L) correction
(attending truly causally to pre_text, and to itself with full internal causal structure) combined
with post_text's [0, L) correction in the same final round -- reducing, position by position, to
plain monolithic prefill. Verify this BEFORE trusting any G>1 accuracy number.

Laplacian-pyramid note (AGENTS.md): `progressive.make_base`/`progressive_correct.py`'s existing base
construction blurs the image AFTER `smart_resize` (i.e. at the model's canvas resolution), not at the
image's native resolution -- this violates AGENTS.md's pyramid-construction rule ("build pyramid
levels in native coordinates before any model/evaluator scaling"). `native_pyramid` below fixes this
for THIS experiment: it builds the low-frequency base from the image's OWN native resolution, THEN
applies `smart_resize` to both the true image and the base. Because of this fix, numbers produced
here are not bit-comparable to the existing base_only/progressive figures in
`analysis/qwen_vl_prefill/README.md` (which used the canvas-resolution base) -- re-run whichever
baseline you need for a fair comparison, using `native_pyramid` for all arms.
"""

import sys
from pathlib import Path

import numpy as np
import torch

for _p in Path(__file__).resolve().parents[1:3]:  # analysis/ (qwen_vl_prefill) + repo root (appcorr)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from appcorr.models.qwen25vl.llm.decoder_layer import ApproxCorrectQwen25VLDecoderLayer

from qwen_vl_prefill import introspect as I
from qwen_vl_prefill import progressive as PR
from qwen_vl_prefill import progressive_correct as PC


# --------------------------------------------------------------------------------------------
# AGENTS.md-compliant Laplacian-pyramid base: build in native coordinates, THEN canvas-scale.
# --------------------------------------------------------------------------------------------

def native_pyramid(image, base_factor, smart_resize_fn, factor, min_px, max_px):
    """Build the low-frequency base from `image`'s OWN native resolution (downsample by
    `base_factor` then upsample back, bilinear, at NATIVE size), then apply the model's own
    `smart_resize` canvas scaling to both the true image and the native-resolution base -- never
    scale-to-canvas first and blur second (that changes the approximate image content; see AGENTS.md
    and this module's docstring). `image`: PIL Image at native resolution.

    Returns (full_np, base_np): both uint8 [H,W,3] at the IDENTICAL smart-resized canvas
    resolution (so both encode to the same grid_thw)."""
    from PIL import Image
    w0, h0 = image.size
    th, tw = smart_resize_fn(h0, w0, factor=factor, min_pixels=min_px, max_pixels=max_px)
    full_r = image.resize((tw, th), Image.BILINEAR)
    if base_factor <= 1:
        base_r = full_r
    else:
        small = image.resize((max(1, w0 // base_factor), max(1, h0 // base_factor)), Image.BILINEAR)
        base_native = small.resize((w0, h0), Image.BILINEAR)   # native-res low-frequency base
        base_r = base_native.resize((tw, th), Image.BILINEAR)  # THEN canvas-scale (never the reverse)
    return np.array(full_r, dtype=np.uint8), np.array(base_r, dtype=np.uint8)


# --------------------------------------------------------------------------------------------
# LLM decoder fork construction
# --------------------------------------------------------------------------------------------

def build_llm_layers(model):
    """One `ApproxCorrectQwen25VLDecoderLayer` per stock text-decoder layer, reused across an
    entire request (construction is cheap -- it wraps existing stock submodules, no new params)."""
    lm = model.model.language_model
    rotary_emb = lm.rotary_emb
    layers = [ApproxCorrectQwen25VLDecoderLayer.from_stock(layer, rotary_emb) for layer in lm.layers]
    return layers


# --------------------------------------------------------------------------------------------
# Core mechanism
# --------------------------------------------------------------------------------------------

@torch.inference_mode()
def reverse_order_prefill(model, tower, llm_layers, processor, prepared, position_ids,
                          full_np, base_np, bands, device):
    """Bottom-up depth-staggered progressive LLM prefill. See module docstring for the mechanism.

    Args:
        tower: `ApproxCorrectQwen25VLVisionTower` wrapping the stock vision tower (see
            `progressive_correct.build_tower`).
        llm_layers: from `build_llm_layers(model)`.
        prepared: `introspect.PreparedInputs` for this (image, prompt).
        position_ids: `introspect.compute_position_ids(model, prepared)`, `[3,1,T]`.
        full_np, base_np: from `native_pyramid`, both smart-resized to the SAME canvas resolution.
        bands: `progressive.band_layout(...)`, raster order (bands[0] = top ... bands[-1] = bottom).

    Returns: (last_hidden_normed [1,1,H] -- post_text's LAST position, post-final-norm,
              cache_feature -- populated `{f"L{i}_kv"}` per LLM layer, full [B,Hkv,N,2,Dh],
              layout -- dict with img_start/img_end/N for the caller's decode step).
    """
    L = len(llm_layers)
    G = len(bands)
    assert L % G == 0, f"this driver assumes L (={L}) divisible by num_groups (={G}) for clean depth quarters"
    checkpoints = [k * L // G for k in range(G + 1)]  # e.g. G=4,L=36 -> [0,9,18,27,36]

    dtype = model.dtype
    lm = model.model.language_model

    # ---- vision: base approx over the WHOLE image (unchanged mechanism) ----
    pv_full, grid = PC.encode_pixels(processor, full_np, device, dtype)
    pv_base, grid_b = PC.encode_pixels(processor, base_np, device, dtype)
    assert torch.equal(grid, grid_b), "full/base must share grid_thw (same canvas resolution)"
    ctx = tower.prepare_full_tokens(pv_full, grid)
    ctx_base = tower.prepare_full_tokens(pv_base, grid)
    Lv = len(tower.blocks)
    vcache = {}
    h_base, vcache = tower.approx_forward(ctx_base["hidden_states"], 0, Lv, ctx_base, vcache, "v")
    h_vis = h_base.clone()  # accumulator, permuted-sequence order (matches progressive_correct.py)

    layout = I.token_layout(prepared)
    img_start, img_end, N = layout.img_start, layout.img_end, prepared.seq_len
    post_text_idx = torch.arange(img_end, N, device=device)

    # ---- LLM: fixed embedding-level "restart" reference (base vision everywhere, exact text) ----
    base_vis_embeds = tower.get_merged_output(h_vis, ctx)  # [Nv,H] raster order, all-base initially
    x0 = lm.embed_tokens(prepared.input_ids)                # [1,N,H], exact text
    img_mask = prepared.image_mask.unsqueeze(-1).expand_as(x0)
    x0 = x0.masked_scatter(img_mask, base_vis_embeds.to(x0.dtype))

    cache_feature = {}

    # ---- base pass: whole sequence through the first checkpoint depth ----
    x_frontier = x0
    for i in range(0, checkpoints[1]):
        x_frontier, cache_feature = llm_layers[i].approx(x_frontier, position_ids, cache_feature, tag=f"L{i}")

    # ---- reversed-order group correction: bottom band first, top band last ----
    order = list(reversed(range(G)))  # e.g. G=4 -> [3,2,1,0] (band index, 0=top ... G-1=bottom)
    for step, gi in enumerate(order):
        band = bands[gi]

        # vision-side: correct this band exactly against the vision tower's own shared cache
        # (unchanged mechanism -- same call `progressive_correct.py` makes per band).
        group_idx = torch.arange(band.tok_start, band.tok_end, device=device)
        x_corr_v, vcache = tower.correct_forward(ctx["hidden_states"].clone(), group_idx, 0, Lv, ctx, vcache, "v")
        tok_v = PC._band_token_idx(tower, ctx, group_idx)
        h_vis[tok_v] = x_corr_v[tok_v]
        band_embeds = tower.get_merged_output(h_vis, ctx)[band.tok_start:band.tok_end]  # this band, now exact

        band_pos = torch.arange(img_start + band.tok_start, img_start + band.tok_end, device=device)
        is_last_group = (step == G - 1)
        # This round's token_idx is the band's own positions, plus post_text (folded in only on the
        # final round -- see docstring). x0 already has exact text everywhere (text is never
        # approximated), so post_text needs no embedding-level override -- only the band's own
        # positions get their fresh (just vision-corrected) layer-0 embedding.
        token_idx = torch.cat([band_pos, post_text_idx]).sort().values if is_last_group else band_pos
        x_round = x0.clone()
        x_round[:, band_pos] = band_embeds.to(x0.dtype)

        depth_target = checkpoints[step + 1]
        x = x_round
        for i in range(0, depth_target):
            x, cache_feature = llm_layers[i].correct(x, token_idx, cache_feature, tag=f"L{i}", position_ids=position_ids)

        x_frontier = x_frontier.clone()
        x_frontier[:, token_idx] = x[:, token_idx]

        if not is_last_group:
            next_depth = checkpoints[step + 2]
            for i in range(depth_target, next_depth):
                x_frontier, cache_feature = llm_layers[i].approx(x_frontier, position_ids, cache_feature, tag=f"L{i}")

    last_hidden = lm.norm(x_frontier[:, -1:])
    return last_hidden, cache_feature, {"img_start": img_start, "img_end": img_end, "N": N}


# --------------------------------------------------------------------------------------------
# Decode handoff: convert the fork's per-layer K/V into a stock DynamicCache, decode unmodified.
# --------------------------------------------------------------------------------------------

@torch.inference_mode()
def greedy_decode_from_cache(model, last_hidden, cache_feature, num_layers, N, position_ids,
                             max_new_tokens, eos_ids, device):
    """Reproduces `progressive.greedy_generate_from_embeds`'s decode loop, but starting from an
    ALREADY-PREFILLED cache (this module's `reverse_order_prefill` output) instead of a fresh
    monolithic/chunked prefill. Action/text generation itself is completely stock: only the
    hand-off (fork cache_feature -> `DynamicCache`) is new."""
    from transformers import DynamicCache

    cache = DynamicCache()
    for i in range(num_layers):
        kv = cache_feature[f"L{i}_kv"]  # [B, Hkv, N, 2, Dh]
        k, v = kv.unbind(dim=3)
        cache.update(k.contiguous(), v.contiguous(), i)

    lm = model.model.language_model
    cur_pos = position_ids[:, :, -1:]
    seq = N
    ids = []
    nt = model.lm_head(last_hidden)[:, -1].argmax(-1)  # first generated token, from the prefill's own last hidden
    for _ in range(max_new_tokens):
        tid = int(nt.item())
        if tid in eos_ids:
            break
        ids.append(tid)
        emb = lm.embed_tokens(nt[:, None])
        cur_pos = cur_pos + 1
        out = lm(inputs_embeds=emb, position_ids=cur_pos, past_key_values=cache,
                 cache_position=torch.arange(seq, seq + 1, device=device), use_cache=True)
        cache = out.past_key_values
        last_hidden = out.last_hidden_state[:, -1:]
        nt = model.lm_head(last_hidden)[:, -1].argmax(-1)
        seq += 1
    return ids
