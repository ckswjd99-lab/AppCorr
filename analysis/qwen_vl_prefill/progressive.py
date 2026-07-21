"""
progressive.py -- Phase 4 (base/residual band decomposition aligned to visual-token groups) +
Phase 6 (monotonic progressive finalization via re-encoding), for the accuracy measurement.

The vision encoder is bidirectional, so a visual-token group finalized EARLY (from a partial image)
differs from its full-image value. This module produces the progressively-finalized visual
embeddings so we can measure the downstream accuracy cost of that staleness, WITHOUT the first-order
correction yet -- we use actual re-encoding of partial reconstructions, which is the UPPER BOUND of
what a cheap correction could achieve.

Visual-token order is raster (row-major) over the merged token grid, so a contiguous visual-token
group == a horizontal BAND of image rows. As residuals stream top-to-bottom, band g is sharpened
and its tokens finalized when residual group g arrives; at that point the image is:
    full-resolution in bands 1..g (top), coarse base in bands g+1..G (bottom).
We encode THAT image and freeze band g's tokens (they attend to the still-coarse bottom -> staleness).

Grid facts (verified): grid_thw = [1, Hp, Wp] patch grid; merged grid = (Hp//merge) x (Wp//merge);
each merged token covers `patch_size*merge` (=28px) of the smart-resized image; token index =
merged_row*merged_w + merged_col.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class Band:
    group: int
    tok_start: int      # visual-token index range [tok_start, tok_end)
    tok_end: int
    px_row_start: int   # pixel-row range in the smart-resized image [px_row_start, px_row_end)
    px_row_end: int


def band_layout(grid_thw: torch.Tensor, merge_size: int, patch_size: int, num_groups: int) -> List[Band]:
    """Split the merged token grid into num_groups contiguous horizontal bands (whole merged-rows),
    each aligned to a contiguous visual-token range AND a contiguous pixel-row range."""
    _, hp, wp = [int(x) for x in grid_thw[0].tolist()]
    merged_h, merged_w = hp // merge_size, wp // merge_size
    merged_px = patch_size * merge_size  # pixels per merged-token row/col
    bands = []
    row_edges = [(merged_h * g) // num_groups for g in range(num_groups + 1)]
    for g in range(num_groups):
        ra, rb = row_edges[g], row_edges[g + 1]
        if rb <= ra:
            continue
        bands.append(Band(
            group=g,
            tok_start=ra * merged_w, tok_end=rb * merged_w,
            px_row_start=ra * merged_px, px_row_end=rb * merged_px,
        ))
    return bands


def make_base(image_np: np.ndarray, factor: int) -> np.ndarray:
    """Low-frequency base: downsample by `factor` then upsample back (bilinear), a proxy for the
    Laplacian pyramid base. image_np: [H, W, 3] uint8."""
    if factor <= 1:
        return image_np.copy()
    from PIL import Image
    h, w = image_np.shape[:2]
    im = Image.fromarray(image_np)
    small = im.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
    return np.array(small.resize((w, h), Image.BILINEAR), dtype=np.uint8)


def reconstruct_upto(full_np: np.ndarray, base_np: np.ndarray, px_row_end: int) -> np.ndarray:
    """Image at a finalization step: full-resolution rows [0, px_row_end), coarse base below."""
    img = base_np.copy()
    img[:px_row_end] = full_np[:px_row_end]
    return img


@torch.inference_mode()
def encode_image(model, processor, image_np: np.ndarray, device):
    """Run vision encoder (+ merger) on a raw uint8 [H,W,3] image, return (visual_tokens [Nv,H], grid_thw)."""
    from PIL import Image
    ip_out = processor.image_processor(images=[Image.fromarray(image_np)], return_tensors="pt")
    pv = ip_out["pixel_values"].to(device, dtype=model.dtype)
    grid = ip_out["image_grid_thw"].to(device)
    embeds = model.model.get_image_features(pv, grid).pooler_output
    return torch.cat(list(embeds), dim=0), grid


@torch.inference_mode()
def progressive_finalized_embeds(model, processor, full_np, base_np, bands, device):
    """Assemble the progressively-finalized visual embeddings: band g's tokens come from encoding
    the reconstruction that is full in bands 1..g and base below. Returns [Nv, H] and also the
    full-image and base-only embeds (computed along the way, for the 3-way comparison).
    (band G's reconstruction == the full image, so full embeds are reused from it.)"""
    # base-only embeds (all bands from the coarse base)
    base_embeds, _ = encode_image(model, processor, base_np, device)
    full_embeds = None
    prog = base_embeds.clone()
    for band in bands:
        img_g = reconstruct_upto(full_np, base_np, band.px_row_end)
        emb_g, _ = encode_image(model, processor, img_g, device)
        prog[band.tok_start:band.tok_end] = emb_g[band.tok_start:band.tok_end]
        if band.px_row_end >= full_np.shape[0]:  # this reconstruction == full image
            full_embeds = emb_g
    if full_embeds is None:  # last band didn't reach the bottom (shouldn't happen with whole-row bands)
        full_embeds, _ = encode_image(model, processor, full_np, device)
    return prog, full_embeds, base_embeds


@torch.inference_mode()
def greedy_generate_from_embeds(model, inputs_embeds, position_ids, max_new_tokens, eos_ids, device):
    """Greedy decode starting from a custom multimodal inputs_embeds (lets us generate from arbitrary
    visual embeddings -- full / base / progressive -- through the identical LLM path). Returns token ids."""
    from transformers import DynamicCache
    lm = model.model.language_model
    cache = DynamicCache()
    T = inputs_embeds.shape[1]
    out = lm(inputs_embeds=inputs_embeds, position_ids=position_ids, past_key_values=cache,
             cache_position=torch.arange(T, device=device), use_cache=True)
    cache = out.past_key_values
    last_hidden = out.last_hidden_state[:, -1:]
    cur_pos = position_ids[:, :, -1:]
    seq = T
    ids = []
    for _ in range(max_new_tokens):
        nt = model.lm_head(last_hidden)[:, -1].argmax(-1)  # [1]
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
        seq += 1
    return ids
