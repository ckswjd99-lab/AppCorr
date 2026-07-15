"""
qwen25vl_flops_estimate.py

Theoretical FLOPs estimate for approx_forward (fixed cost, happens once per request
regardless of keep_rate) vs correct_forward (cost scales with keep_rate/recompute rate),
for Qwen2.5-VL-32B on a representative RefCOCO image. Pure arithmetic -- no GPU, no
dataset, no model weights loaded. Sweep over keep_rate to see how correction cost scales
and where it crosses the fixed approx cost.

Architecture numbers (from the actual 32B config + Qwen2_5_VLVisionConfig defaults,
confirmed against this investigation's own logged values -- num_heads=16 vision /
num_heads=40 LLM, GQA num_key_value_heads=8, fullatt_block_indexes={7,15,23,31}):
  vision: hidden=1280, heads=16, head_dim=80, intermediate=3456, depth=32,
          patch_size=14, spatial_merge_size=2 (merge_unit=4), window_size=112
  llm:    hidden=5120, heads=40, head_dim=128, num_key_value_heads=8 (GQA groups=5),
          intermediate=27648, depth=64

Representative image: 448x672 (by far the single most common smart-resized grid shape
seen across this whole session's RefCOCO logs) -> patches: 32x48=1536 raw patches ->
384 merge-groups (2x2 merge). Text (permanent group) token count measured directly from
the real chat template + GROUNDING_PROMPT_TMPL for a typical expression: 72 tokens.

Key modeling choices (see module docstring comments at each formula for why):
  - Standard 2*M*K*N convention for linear-layer FLOPs (multiply-add = 2 FLOPs).
  - Attention FLOPs use 4*Q*N_k*D (QK^T + softmax@V, summed over heads via D=H*d_h) --
    GQA doesn't change this formula (KV heads get repeated before the matmul either way).
  - Windowed vision layers (28 of 32) use N_k=64 (one 112px window = 8x8=64 raw patches,
    matching window_size/patch_size), NOT the full image -- only the 4 full-attention
    layers (7/15/23/31) use N_k=1536 (the whole image). This exactly mirrors the real
    architecture (see appcorr/models/qwen25vl/vision/backbone.py).
  - Correction is assumed to select merge-groups uniformly at random for FLOPs purposes
    (a fixed keep_rate fraction of every window gets corrected on average) -- the REAL
    pscore-based selection is content-dependent, not uniform, but total corrected-patch
    count (and hence total FLOPs) is the same either way; only the per-window distribution
    differs, which doesn't change the FLOPs sum (windowed attention cost is linear in
    corrected-query-count).
  - LLM causal masking is NOT applied to the attention FLOPs count (N_k is treated as the
    full sequence length for every query, both in approx and correct) -- a deliberate
    simplification applied uniformly to both approx and correct so the RATIO between them
    stays meaningful, at the cost of a shared constant-factor overestimate vs the true
    causal-masked FLOPs count.
  - The patch merger (2-layer MLP over merge-groups, runs once after all 32 vision blocks)
    is a small, keep_rate-INDEPENDENT cost (it always processes the full image's current
    merge-group states to build the spliced LLM input) -- omitted as a fixed, common
    addend that doesn't affect the approx-vs-correct comparison.
  - keep_rate=0.0 is a special case matching the real pipeline exactly: `refcoco_attn_fused_eval.py`
    guards `correct_forward` behind `if selected:`, so zero selected merge-groups means
    correct_forward is never called at all (not even a permanent-group-only correction) --
    this is the true approx-only floor, correction cost = 0. For keep_rate>0, every
    correction round includes the permanent group (72 text tokens) PLUS the keep_rate
    fraction of image tokens, matching `executor.correct_forward`'s
    `token_idx = cat([permanent_group_idx, image_token_positions]).unique()`.

Run:
    python analysis/experiments/qwen25vl_flops_estimate.py
"""

import argparse


def linear_flops(n, d_in, d_out):
    """2*n*d_in*d_out (multiply-add counted as 2 FLOPs), the standard convention."""
    return 2 * n * d_in * d_out


def attn_flops(n_q, n_k, d_model):
    """QK^T + softmax(...)@V, summed over heads via d_model=H*head_dim. GQA-invariant:
    KV heads get repeated to match Q heads before the matmul, so the FLOPs count is the
    same as if there were no GQA reduction (the reduction only saves memory/bandwidth for
    the K/V projection itself, handled separately in the QKV-projection cost)."""
    return 2 * n_q * n_k * d_model + 2 * n_q * n_k * d_model


class VisionArch:
    hidden = 1280
    heads = 16
    head_dim = 80
    intermediate = 3456
    depth = 32
    fullatt_layers = {7, 15, 23, 31}
    patch_size = 14
    spatial_merge_size = 2
    window_size = 112

    @property
    def window_patches(self):
        return (self.window_size // self.patch_size) ** 2  # 8*8 = 64


class LLMArch:
    hidden = 5120
    heads = 40
    head_dim = 128
    num_key_value_heads = 8
    intermediate = 27648
    depth = 64


def vision_block_flops(q_c, n_k, arch: VisionArch):
    """One vision transformer block, q_c queries (fresh Q/K/V computed for these only),
    attending to n_k keys (either the full image [full-attn layers] or one window
    [windowed layers], both fixed per-layer regardless of approx/correct)."""
    d = arch.hidden
    qkv = linear_flops(q_c, d, 3 * d)          # fused QKV projection
    attn = attn_flops(q_c, n_k, d)
    out_proj = linear_flops(q_c, d, d)
    mlp = 3 * linear_flops(q_c, d, arch.intermediate)  # SwiGLU: gate + up + down
    return qkv + attn + out_proj + mlp


def vision_tower_flops(q_c_raw_patches, n_v_raw_patches, arch: VisionArch):
    """Full 32-block vision tower cost for q_c_raw_patches queries (all raw patches for
    approx; keep_rate * n_v_raw_patches for correct), n_v_raw_patches = total raw patches
    in the image (fixed, determines full-attn layers' key count)."""
    n_full = len(arch.fullatt_layers)
    n_windowed = arch.depth - n_full
    full_cost = n_full * vision_block_flops(q_c_raw_patches, n_v_raw_patches, arch)
    windowed_cost = n_windowed * vision_block_flops(q_c_raw_patches, arch.window_patches, arch)
    return full_cost + windowed_cost


def llm_layer_flops(q_c, n_k, arch: LLMArch):
    """One LLM decoder layer, q_c queries, n_k keys (always the full current sequence
    length, since correction reads from the full KV cache)."""
    d = arch.hidden
    d_kv = arch.num_key_value_heads * arch.head_dim
    qkv = linear_flops(q_c, d, d) + 2 * linear_flops(q_c, d, d_kv)  # Q: D->D, K/V: D->d_kv each
    attn = attn_flops(q_c, n_k, d)
    out_proj = linear_flops(q_c, d, d)
    mlp = 3 * linear_flops(q_c, d, arch.intermediate)  # SwiGLU: gate + up + down
    return qkv + attn + out_proj + mlp


def llm_decoder_flops(q_c, n_llm, arch: LLMArch):
    return arch.depth * llm_layer_flops(q_c, n_llm, arch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-h", type=int, default=448)
    p.add_argument("--image-w", type=int, default=672)
    p.add_argument("--text-tokens", type=int, default=72, help="permanent-group (non-image) "
                    "token count -- measured directly from the real chat template + "
                    "GROUNDING_PROMPT_TMPL for a typical RefCOCO expression.")
    p.add_argument("--keep-rates", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0])
    args = p.parse_args()

    v_arch, l_arch = VisionArch(), LLMArch()
    n_v_raw = (args.image_h // v_arch.patch_size) * (args.image_w // v_arch.patch_size)
    merge_unit = v_arch.spatial_merge_size ** 2
    n_merge_groups = n_v_raw // merge_unit
    n_llm = args.text_tokens + n_merge_groups

    print(f"[flops] image {args.image_h}x{args.image_w} -> {n_v_raw} raw patches -> "
          f"{n_merge_groups} merge-groups; LLM sequence length = {args.text_tokens} text + "
          f"{n_merge_groups} image = {n_llm} tokens\n")

    # Approx: fixed cost, independent of keep_rate. ALL raw patches / ALL LLM tokens.
    approx_vision = vision_tower_flops(n_v_raw, n_v_raw, v_arch)
    approx_llm = llm_decoder_flops(n_llm, n_llm, l_arch)
    approx_total = approx_vision + approx_llm
    print(f"[flops] approx_forward (FIXED, independent of keep_rate):")
    print(f"    vision: {approx_vision/1e9:10.2f} GFLOPs")
    print(f"    llm:    {approx_llm/1e9:10.2f} GFLOPs")
    print(f"    total:  {approx_total/1e9:10.2f} GFLOPs\n")

    print(f"[flops] correct_forward sweep over keep_rate (recompute rate):")
    print(f"    {'keep_rate':>10s} {'vision GFLOPs':>15s} {'llm GFLOPs':>13s} "
          f"{'correct total':>15s} {'correct/approx':>15s} {'grand total':>13s} {'grand/approx':>13s}")
    for kr in args.keep_rates:
        if kr == 0.0:
            # Matches the real pipeline exactly: `if selected:` guards correct_forward, so
            # zero selected merge-groups means correct_forward is never called at all --
            # not even the permanent-group-only case. This is the true "approx-only" floor.
            correct_vision = correct_llm = correct_total = 0.0
        else:
            q_c_raw = kr * n_v_raw
            q_c_llm = args.text_tokens + kr * n_merge_groups  # permanent group always corrected
            correct_vision = vision_tower_flops(q_c_raw, n_v_raw, v_arch)
            correct_llm = llm_decoder_flops(q_c_llm, n_llm, l_arch)
            correct_total = correct_vision + correct_llm
        grand_total = approx_total + correct_total
        print(f"    {kr:10.2f} {correct_vision/1e9:15.2f} {correct_llm/1e9:13.2f} "
              f"{correct_total/1e9:15.2f} {correct_total/approx_total:15.3f} "
              f"{grand_total/1e9:13.2f} {grand_total/approx_total:13.3f}")


if __name__ == "__main__":
    main()
