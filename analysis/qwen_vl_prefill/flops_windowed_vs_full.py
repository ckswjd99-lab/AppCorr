"""
flops_windowed_vs_full.py -- compute breakdown of the Qwen2.5-VL vision encoder: windowed (28) vs
full-attention (4) layers, and the "full-all + windowed-1/4" approximation.

Finding: per-token linear (QKV+outproj+SwiGLU FFN = 4d^2 + 3 d d_ffn) is identical for both layer
types and DOMINATES; the windowed attention matrix (2 w d, w=64 patches/window) is negligible, and
the full attention matrix (2 N d) is only 3-10% of the encoder even though it is the quadratic term.
So windowed layers are 79-85% of encoder compute (there are 7x more of them). Doing full-attention on
all tokens but windowed layers (+their FFN) on only 1/4 -> ~2.5x speedup (59-64% saved).

Consequence: the 4 full-attention layers attend to ALL N tokens, so the 3/4 that skipped windowed
processing enter them as base/stale -> full layers mix fresh 1/4 + stale 3/4 = exactly the
progressive/cheap-correction staleness (accuracy ~0 on VQA, -3.12pp on RefCOCO grounding). Windowed
layers are purely local (zero cross-window contamination when sub-selected); ALL cross-window
staleness -- and hence the -2.56pp grounding cheap-correction overhead -- lives in the 4 full layers.
"""
d, d_ffn, w, L_full, L_win = 1280, 3420, 64, 4, 28  # Qwen2.5-VL vision config
lin = 4 * d * d + 3 * d * d_ffn          # per-token proj + SwiGLU FFN (both layer types)
win_attn = 2 * w * d                     # per-token windowed attention matrix

def report(N):
    lw = lin + win_attn
    lf = lin + 2 * N * d
    tot = L_win * N * lw + L_full * N * lf
    new = L_win * (N // 4) * lw + L_full * N * lf   # full-all + windowed-1/4
    return dict(N=N, ratio=lf / lw, tot_G=tot / 1e9,
                win_pct=100 * L_win * N * lw / tot, full_pct=100 * L_full * N * lf / tot,
                speedup=tot / new, saved=100 * (1 - new / tot))

if __name__ == "__main__":
    for N in (1700, 4096, 6656):
        r = report(N)
        print(f"N={r['N']:5d}: full/win layer {r['ratio']:.2f}x | enc {r['tot_G']:.0f}G "
              f"(win {r['win_pct']:.0f}% full {r['full_pct']:.0f}%) | 1/4-win scheme {r['speedup']:.2f}x ({r['saved']:.0f}% saved)")
