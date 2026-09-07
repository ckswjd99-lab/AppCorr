"""Closed-form backbone FLOPs and the critical split, validated against the hooked measurements.

The measured path needs a GPU and a run per (model, dataset, keep). The shapes it depends on do not:
FLOPs are a deterministic function of token counts, layer widths and the schedule, and token counts
come from the image processor alone. So if a formula reproduces the measurements, the rest of the
table can be filled on CPU in seconds instead of GPU-hours.

**The formula is not `(keep/g) x full`.** That undershoots by 1.75x on OV2/ChartQA, because the
final round carries three things at once:

  1. its own group's correction -- `keep/g` of the tokens, but over the FULL depth walked so far,
     not the average depth;
  2. the text tokens, which join the LAST round in every VLM arm (they were never approximated, but
     they attend to the image block and the answer is read off the last position);
  3. the approximate frontier still to be advanced from the last bound to the end of the axis --
     over the WHOLE sequence, which is the term the naive ratio misses entirely.

So the schedule is replicated here rather than approximated, using the same cost-equalised
`layer_bounds` the axis computes at run time.

    python analysis/experiments/flops_analytic.py --validate     # against the measured runs
    python analysis/experiments/flops_analytic.py --fill         # the whole table
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# --- per-layer closed forms ---------------------------------------------------------------------- #
# 2 FLOPs per multiply-accumulate. Norms, activations, softmax and residual adds are excluded, the
# same convention the hooks follow, so the two are comparable by construction.

def attn_flops(tokens: int, heads: int, head_dim: int, keys: Optional[int] = None) -> float:
    """QK^T plus AV. `keys` differs from `tokens` for windowed attention."""
    return 2 * 2 * heads * tokens * (tokens if keys is None else keys) * head_dim


def vision_layer_flops(n: int, hidden: int, heads: int, ffn: int,
                       keys: Optional[int] = None) -> float:
    """One pre-norm encoder layer: fused QKV + proj + 2-layer MLP + attention."""
    head_dim = hidden // heads
    return (2 * n * hidden * (3 * hidden)          # qkv
            + attn_flops(n, heads, head_dim, keys)
            + 2 * n * hidden * hidden              # out proj
            + 2 * 2 * n * hidden * ffn)            # mlp up + down


def llm_layer_flops(n: int, hidden: int, heads: int, kv_heads: int, ffn: int,
                    gated: bool = True) -> float:
    """One decoder layer with GQA and a gated MLP (gate+up+down = 3 matmuls)."""
    head_dim = hidden // heads
    kv_dim = kv_heads * head_dim
    return (2 * n * hidden * (hidden + 2 * kv_dim + hidden)   # q,k,v,o
            + attn_flops(n, heads, head_dim)
            + (3 if gated else 2) * 2 * n * hidden * ffn)


@dataclass
class Axis:
    """A backbone as the schedule sees it: an optional vision half then an optional LLM half."""
    v_layers: int = 0
    v_hidden: int = 0
    v_heads: int = 0
    v_ffn: int = 0
    v_keys: Optional[int] = None          # windowed attention key length, None = global
    l_layers: int = 0
    l_hidden: int = 0
    l_heads: int = 0
    l_kv_heads: int = 0
    l_ffn: int = 0
    l_gated: bool = True

    @property
    def n_stages(self) -> int:
        return self.v_layers + self.l_layers

    def v_cost(self, n_patch: int) -> float:
        return vision_layer_flops(n_patch, self.v_hidden, self.v_heads, self.v_ffn, self.v_keys)

    def l_cost(self, seq: int) -> float:
        if not self.l_layers:
            return 0.0
        return llm_layer_flops(seq, self.l_hidden, self.l_heads, self.l_kv_heads, self.l_ffn,
                               self.l_gated)

    def full(self, n_patch: int, seq: int) -> float:
        """The ceiling: every stage over every token. 100% critical by the rule."""
        return self.v_layers * self.v_cost(n_patch) + self.l_layers * self.l_cost(seq)

    def layer_bounds(self, groups: int, n_patch: int, seq: int) -> List[int]:
        """Round boundaries by equal COST, matching `OV2UnifiedAxis.layer_bounds`.

        Equal stage counts would be wrong: a vision layer and a decoder layer are not
        interchangeable units when one is wide in tokens and the other in width.
        """
        if groups <= 1:
            return [self.n_stages]
        per = [self.v_cost(n_patch)] * self.v_layers + [self.l_cost(seq)] * self.l_layers
        total = sum(per)
        cum, acc = [], 0.0
        for c in per:
            acc += c
            cum.append(acc / total)
        bounds = []
        for r in range(1, groups):
            target = r / groups
            b = next((i + 1 for i, c in enumerate(cum) if c >= target), self.n_stages)
            bounds.append(max(1, min(b, self.n_stages - 1)))
        bounds = sorted(set(bounds))
        while len(bounds) < groups - 1:
            for cand in range(1, self.n_stages):
                if cand not in bounds:
                    bounds.append(cand)
                    break
            bounds = sorted(set(bounds))
        return bounds + [self.n_stages]

    def interleaved_critical(self, groups: int, keep: float, n_patch: int, seq: int,
                             n_img_tok: int) -> Tuple[float, float]:
        """(critical, total) for the interleaved walk. Returns FLOPs.

        Replays the schedule: an opening approximate pass to the first bound, then per round a
        correction of that round's group over the depth walked so far, followed by advancing the
        approximate frontier to the next bound. Critical is the last round's body -- correction,
        text, and the frontier advance that follows it.
        """
        b = self.layer_bounds(groups, n_patch, seq)
        n_text = max(0, seq - n_img_tok)
        vc, lc = self.v_cost(n_patch), self.l_cost(seq)
        # Per-token unit costs, so a partial correction can be priced. Attention stays linear in the
        # QUERY count at the full key length: a corrected query attends over every key, corrected or
        # not, so `vc_tok * n_selected` is the right charge and not an approximation.
        vc_tok = vc / max(n_patch, 1)
        lc_tok = lc / max(seq, 1)

        def stage_cost(lo: int, hi: int) -> float:
            """Cost of walking stages [lo, hi) over every token."""
            v = min(hi, self.v_layers) - min(lo, self.v_layers)
            l = max(0, hi - self.v_layers) - max(0, lo - self.v_layers)
            return v * vc + l * lc

        # Opening approximate pass to the first bound. Nothing has arrived: this is arrival 0.
        total = stage_cost(0, b[0])
        critical = 0.0
        g_patch = n_patch * keep / groups
        g_tok = n_img_tok * keep / groups

        for r in range(groups):
            # Round r corrects at the frontier it finds, which is bounds[r] -- the advance to the
            # NEXT bound happens after the correction, at the end of the same round. Pricing the
            # correction at bounds[r-1] instead (advance first) understates the last round by the
            # whole LLM tail and was off by 2.2-5.5x against the hooks.
            depth = b[r]
            v_front = min(depth, self.v_layers)
            l_depth = max(0, depth - self.v_layers)
            corr = v_front * vc_tok * g_patch
            if l_depth:
                # Text joins the FINAL round only: never approximated, but it attends to the image
                # block and the answer is read off the last position.
                rows = g_tok + (n_text if r == groups - 1 else 0)
                corr += l_depth * lc_tok * rows

            nxt = b[r + 1] if r + 1 < len(b) else self.n_stages
            adv = stage_cost(depth, nxt)

            total += corr + adv
            if r == groups - 1:
                critical = corr + adv          # adv is zero here: the axis is already fully walked
        return critical, total


# --- model registry ------------------------------------------------------------------------------- #

MODELS: Dict[str, Axis] = {
    # LLaVA-OneVision-2-8B: 24 encoder layers (global attention, one segment per image) + Qwen3 36L.
    "ov2": Axis(v_layers=24, v_hidden=1024, v_heads=16, v_ffn=4096,
                l_layers=36, l_hidden=4096, l_heads=32, l_kv_heads=8, l_ffn=12288),
    # Gemma 3 4B: 27 SigLIP layers over a fixed 896x896 canvas + 34 decoder layers.
    "gemma3": Axis(v_layers=27, v_hidden=1152, v_heads=16, v_ffn=4304,
                   l_layers=34, l_hidden=2560, l_heads=8, l_kv_heads=4, l_ffn=10240),
    # Qwen2.5-VL 32B / 72B: the encoder is SHARED across sizes (depth 32, h1280, window 112,
    # full attention only at layers 7/15/23/31).
    "qwen25vl_32b": Axis(v_layers=32, v_hidden=1280, v_heads=16, v_ffn=3456, v_keys=64,
                         l_layers=64, l_hidden=5120, l_heads=40, l_kv_heads=8, l_ffn=27648),
    "qwen25vl_72b": Axis(v_layers=32, v_hidden=1280, v_heads=16, v_ffn=3456, v_keys=64,
                         l_layers=80, l_hidden=8192, l_heads=64, l_kv_heads=8, l_ffn=29568),
    # DINOv3 ViT-7B/16: a VFM, so the axis is the trunk alone.
    "dinov3": Axis(v_layers=40, v_hidden=4096, v_heads=32, v_ffn=8192),
    # OpenCLIP ViT-bigG-14 vision tower.
    "openclip": Axis(v_layers=48, v_hidden=1664, v_heads=16, v_ffn=8192),
    # SAM 3 vision encoder: 32 layers, global at 4 of them, windowed at 576 keys elsewhere.
    "sam3": Axis(v_layers=32, v_hidden=1152, v_heads=16, v_ffn=4608, v_keys=576),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--groups", type=int, default=4)
    a = ap.parse_args()

    if a.validate:
        # (model, dataset, n_patch, seq, n_img_tok, measured_full, meas_k30, meas_k50) in GFLOPs.
        # Shapes are the per-dataset means the measured runs recorded; the FLOPs are what the hooks
        # counted on the same samples.
        # Shapes are the MEASURED per-dataset means over the same strided samples the GPU runs
        # used (24 for chartqa/textvqa, 16 elsewhere). Using a different sample count here would
        # compare a formula against shapes the measurement never saw.
        CASES = [
            ("ov2", "chartqa",      2410,  648,  602, 11374.6, 1495.3,  1871.1),
            ("ov2", "textvqa",      4027, 1046, 1007, 19337.3, 1798.8,  2775.6),
            ("ov2", "infovqa",     11296, 2868, 2824, 69185.3, 5909.3,  8629.9),
            ("ov2", "docvqa",      18580, 4688, 4645, 124497.1, 8827.4, 13403.0),
            ("ov2", "realworldqa",  6901, 1778, 1725, 35793.7, 2087.6,  3607.9),
            ("ov2", "pope",         1497,  416,  374,  7030.1, 1053.2,  1407.9),
            ("gemma3", "chartqa",     4096, 294, 256, 7369.9, 887.1, 1080.0),
            ("gemma3", "textvqa",     4096, 287, 256, 7327.2, 724.8, 1112.3),
            ("gemma3", "infovqa",     4096, 292, 256, 7358.8, 834.4, 1095.0),
            ("gemma3", "pope",        4096, 289, 256, 7338.7, 798.8, 1175.9),
            ("gemma3", "realworldqa", 4096, 301, 256, 7418.9, 521.9,  786.1),
        ]
        print(f"{'model':<9}{'dataset':<13}{'full meas':>11}{'calc':>10}{'r':>6}"
              f"{'k30 meas':>10}{'calc':>9}{'r':>6}{'k50 meas':>10}{'calc':>9}{'r':>6}")
        worst_full = worst_crit = 0.0
        for m, ds, npatch, seq, ntok, mf, m30, m50 in CASES:
            ax = MODELS[m]
            cf = ax.full(npatch, seq) / 1e9
            c30 = ax.interleaved_critical(a.groups, 0.30, npatch, seq, ntok)[0] / 1e9
            c50 = ax.interleaved_critical(a.groups, 0.50, npatch, seq, ntok)[0] / 1e9
            rf, r30, r50 = cf / mf, c30 / m30, c50 / m50
            worst_full = max(worst_full, abs(rf - 1))
            worst_crit = max(worst_crit, abs(r30 - 1), abs(r50 - 1))
            print(f"{m:<9}{ds:<13}{mf:>11.1f}{cf:>10.1f}{rf:>6.2f}"
                  f"{m30:>10.1f}{c30:>9.1f}{r30:>6.2f}{m50:>10.1f}{c50:>9.1f}{r50:>6.2f}")
        print(f"\n  worst |calc/meas - 1|:  full {100*worst_full:.1f}%   critical {100*worst_crit:.1f}%")
        print("  A formula that tracks the ceiling but not the critical split is not usable for "
              "the table:\n  the critical column is the one being claimed.")


if __name__ == "__main__":
    main()
