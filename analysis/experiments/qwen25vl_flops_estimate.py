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

Interleaved (multi-round, `--rounds G>1`) correction, e.g. the `interleaved_g4` configs used
elsewhere in this investigation: the total keep_rate is assumed split EVENLY across G rounds
(kr/G merge-groups arrive per round) -- a simplifying assumption for a symmetric-round FLOPs
estimate, not a claim about how any specific grouping strategy schedules its rounds.
  - VISION correction cost is PROVABLY INVARIANT to round count: `vision_tower_flops` is linear
    in query count (every term is proportional to q_c, no fixed per-call overhead), so summing
    G rounds of (kr/G * n_v_raw) queries each gives exactly the same total as one round of
    (kr * n_v_raw) queries. Interleaving does not change vision's total FLOPs.
  - LLM correction cost is NOT invariant to round count, and this is the whole point of modeling
    interleaving separately: `executor.correct_forward` includes the FULL permanent group (72
    text tokens) in `token_idx` on EVERY round (`torch.cat([permanent_group_idx,
    image_token_positions]).unique()`), not just once total. G rounds means the 72-token
    permanent-group refresh cost is paid G times, not amortized -- this is real, necessary work
    (the final pre-generation hidden state must reflect whatever image tokens have arrived so
    far, every round), not an implementation inefficiency, but it IS extra FLOPs that single-shot
    (G=1) correction doesn't pay.

Run:
    python analysis/experiments/qwen25vl_flops_estimate.py
    python analysis/experiments/qwen25vl_flops_estimate.py --rounds 1 2 4 8
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


def correct_forward_flops(kr, rounds, n_v_raw, n_merge_groups, n_llm, text_tokens, v_arch, l_arch,
                           repeat_text=True):
    """Total correct_forward FLOPs for a given keep_rate, split evenly across `rounds`
    interleaved correction rounds (rounds=1 == single-shot, the original behavior).

    repeat_text=True (the real current implementation): every round's token_idx includes the
    full permanent group (text_tokens), matching `executor.correct_forward`'s
    `token_idx = cat([permanent_group_idx, image_token_positions]).unique()` -- the text tokens'
    hidden states are re-corrected every round so they reflect whichever image tokens have
    arrived so far.

    repeat_text=False (an UNIMPLEMENTED optimization, modeled here for comparison only): only
    the LAST round corrects the permanent group; earlier rounds correct ONLY that round's
    arriving image tokens. Valid if intermediate rounds' text-token hidden states are never
    read (nothing consumes them until the final pre-generation state) -- defers all of the
    permanent-group refresh cost to a single pass instead of paying it G times. Rounds 1..G-1
    do image-only corrections of (kr/rounds)*n_merge_groups queries each; round G does
    text_tokens + (kr/rounds)*n_merge_groups (the last batch of image tokens, plus the one and
    only permanent-group refresh).
    Returns (vision_flops, llm_flops, total_flops)."""
    if kr == 0.0:
        return 0.0, 0.0, 0.0
    # Vision: round-count-invariant (linear in query count, no per-round fixed cost) -- compute
    # directly from the total kr rather than summing rounds, though the sum would be identical.
    vision = vision_tower_flops(kr * n_v_raw, n_v_raw, v_arch)
    # LLM: NOT round-invariant when repeat_text=True -- every round re-pays the full
    # permanent-group (text_tokens) cost. When repeat_text=False, only the last round does.
    llm = 0.0
    per_round_image = (kr / rounds) * n_merge_groups
    for r in range(rounds):
        is_last = (r == rounds - 1)
        q_c_llm_round = per_round_image + (text_tokens if (repeat_text or is_last) else 0)
        llm += llm_decoder_flops(q_c_llm_round, n_llm, l_arch)
    return vision, llm, vision + llm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-h", type=int, default=448)
    p.add_argument("--image-w", type=int, default=672)
    p.add_argument("--text-tokens", type=int, default=72, help="permanent-group (non-image) "
                    "token count -- measured directly from the real chat template + "
                    "GROUNDING_PROMPT_TMPL for a typical RefCOCO expression.")
    p.add_argument("--keep-rates", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--rounds", type=int, nargs="+", default=[1],
                    help="Number of interleaved correction rounds to sweep (1 = single-shot, "
                    "the original behavior; e.g. 4 for an interleaved_g4-style schedule). "
                    "Total keep_rate is split evenly across rounds.")
    p.add_argument("--text-modes", choices=["repeat", "defer"], nargs="+", default=["repeat"],
                    help="repeat: every round re-corrects the permanent group (the real current "
                    "implementation). defer: only the LAST round corrects the permanent group "
                    "(an unimplemented optimization, modeled for comparison only -- see "
                    "correct_forward_flops docstring). Pass both to compare side by side.")
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

    for text_mode in args.text_modes:
        repeat_text = (text_mode == "repeat")
        for rounds in args.rounds:
            if rounds == 1 and text_mode == "defer":
                continue  # defer vs repeat are identical at rounds=1 (only one round exists anyway)
            label = "single-shot" if rounds == 1 else f"interleaved, {rounds} rounds, text={text_mode}"
            print(f"[flops] correct_forward sweep over keep_rate ({label}):")
            print(f"    {'keep_rate':>10s} {'vision GFLOPs':>15s} {'llm GFLOPs':>13s} "
                  f"{'correct total':>15s} {'correct/approx':>15s} {'grand total':>13s} {'grand/approx':>13s}")
            for kr in args.keep_rates:
                correct_vision, correct_llm, correct_total = correct_forward_flops(
                    kr, rounds, n_v_raw, n_merge_groups, n_llm, args.text_tokens, v_arch, l_arch,
                    repeat_text=repeat_text)
                grand_total = approx_total + correct_total
                print(f"    {kr:10.2f} {correct_vision/1e9:15.2f} {correct_llm/1e9:13.2f} "
                      f"{correct_total/1e9:15.2f} {correct_total/approx_total:15.3f} "
                      f"{grand_total/1e9:13.2f} {grand_total/approx_total:13.3f}")
            print()

    if len(args.rounds) > 1:
        print(f"[flops] interleaving overhead at fixed keep_rate points, repeat-text vs defer-text "
              f"(extra LLM cost from re-paying the {args.text_tokens}-token permanent group every "
              f"round vs only once, at the last round):")
        header = f"    {'keep_rate':>10s} " + " ".join(
            f"{('G='+str(g)+'/'+m):>16s}" for g in args.rounds for m in (["repeat", "defer"] if g > 1 else ["repeat"]))
        print(header)
        for kr in [0.35, 0.5, 0.65, 1.0]:
            if kr not in args.keep_rates:
                continue
            base_total = None
            cells = []
            for rounds in args.rounds:
                modes = ["repeat", "defer"] if rounds > 1 else ["repeat"]
                for m in modes:
                    _, _, total = correct_forward_flops(
                        kr, rounds, n_v_raw, n_merge_groups, n_llm, args.text_tokens, v_arch, l_arch,
                        repeat_text=(m == "repeat"))
                    if base_total is None:
                        base_total = total
                    cells.append(f"{total/base_total:15.3f}x")
            print(f"    {kr:10.2f} " + " ".join(cells))


if __name__ == "__main__":
    main()
