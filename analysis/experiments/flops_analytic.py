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


# --- Qwen3.5 hybrid decoder ----------------------------------------------------------------------- #
#
# The `Axis` above prices a UNIFORM decoder (GQA + a gated MLP at every layer). Qwen3.5 is neither:
# three layers in four are Gated DeltaNet (linear attention, no quadratic term, a depthwise conv and
# a recurrent scan the generic hooks cannot see) and every layer's MLP is a 256-expert MoE. So it
# gets its own entry, whose per-token formulas are the SAME ones `appcorr/flops/hooks.py` charges --
# `_qwen35_experts_flops` and `_qwen35_deltanet_core_flops` copied here term for term (copied, not
# imported: this module must stay a CPU-only, torch-free closed form) -- so the two are comparable
# by construction and the reconciliation below is a real check rather than a tautology.
#
# Dims come from the HF configs under /NHNHOME/huggingface/hub (text_config): `layer_types` /
# `full_attention_interval` for the layer mix, `linear_{num_key,num_value}_heads` and
# `linear_{key,value}_head_dim` + `linear_conv_kernel_dim` for the GDN block, `moe_intermediate_size`
# / `shared_expert_intermediate_size` / `num_experts` / `num_experts_per_tok` for the MoE (the 4B is
# dense: `intermediate_size`, no experts).


@dataclass
class Qwen35Decoder:
    """One Qwen3.5 text decoder, priced per token and per corrected row.

    Conventions match the hooks exactly: 2 FLOPs per MAC, no norms/activations/softmax/bias,
    attention charged as `2 * 2 * H_q * Sq * Sk * D` (query heads, so GQA is not under-counted),
    experts charged on the ROUTED count (top_k per token) plus the router's own projection,
    lm_head and the embedding table excluded (the hooked reports install on the vision tower and
    the language model only, so `lm_head` sits outside the subtree).
    """
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    n_full: int                  # layers with softmax attention
    n_linear: int                # layers with Gated DeltaNet
    lin_k_heads: int
    lin_v_heads: int
    lin_k_dim: int
    lin_v_dim: int
    conv_kernel: int
    num_experts: int = 0
    top_k: int = 0
    moe_inter: int = 0
    shared_inter: int = 0
    dense_inter: int = 0         # 4B: an ordinary gated MLP instead of the MoE block
    vocab: int = 248320

    def __post_init__(self):
        assert self.n_full + self.n_linear == self.layers, (self.n_full, self.n_linear, self.layers)

    # -- per-token, per-layer pieces ---------------------------------------------------------
    @property
    def key_dim(self) -> int:
        return self.lin_k_heads * self.lin_k_dim

    @property
    def value_dim(self) -> int:
        return self.lin_v_heads * self.lin_v_dim

    def mlp_tok(self) -> float:
        """MoE block, per token. `2 * top_k * 3 * I * H` for the routed experts and
        `2 * H * num_experts` for the router are `hooks._qwen35_experts_flops` divided by the token
        count; the shared expert's gate/up/down and its 1-wide gate are ordinary Linears the
        generic hooks see."""
        if self.dense_inter:
            return 3 * 2 * self.hidden * self.dense_inter
        return (2 * self.top_k * 3 * self.moe_inter * self.hidden
                + 2 * self.hidden * self.num_experts
                + 3 * 2 * self.hidden * self.shared_inter
                + 2 * self.hidden)

    def gdn_proj_tok(self) -> float:
        """The Gated DeltaNet layer's GEMMs: in_proj_qkv [H x 2K+V], in_proj_z [H x V],
        in_proj_b and in_proj_a [H x num_v_heads], out_proj [V x H]."""
        return (2 * self.hidden * (2 * self.key_dim + self.value_dim)
                + 2 * self.hidden * self.value_dim
                + 2 * 2 * self.hidden * self.lin_v_heads
                + 2 * self.value_dim * self.hidden)

    def gdn_conv_tok(self) -> float:
        """Depthwise causal conv over the 2K+V mixed channels: `hooks._conv_flops` with
        in_channels/groups = 1, i.e. 2 * C * K per output column (the K-1 padding columns are a
        boundary term, not charged)."""
        return 2 * (2 * self.key_dim + self.value_dim) * self.conv_kernel

    def gdn_scan_tok(self) -> float:
        """`hooks._qwen35_deltanet_core_flops` per token: the state update (k (x) v) and the
        readout (q . S), 2 * dk * dv MACs per value head."""
        return 2 * self.lin_v_heads * 2 * self.lin_k_dim * self.lin_v_dim

    def full_proj_tok(self) -> float:
        """q_proj is [H x heads*dh*2] (query and its output gate in one GEMM), k/v are
        [H x kv_heads*dh], o is [heads*dh x H]."""
        return (2 * self.hidden * (self.heads * self.head_dim * 2
                                   + 2 * self.kv_heads * self.head_dim)
                + 2 * (self.heads * self.head_dim) * self.hidden)

    def attn(self, q_tokens: float, keys: float) -> float:
        """QK^T + AV at the query heads, the `record_attention` convention."""
        return 2 * 2 * self.heads * q_tokens * keys * self.head_dim

    # -- the three closed forms --------------------------------------------------------------
    def prefill_flops(self, n: float) -> float:
        """A stock prefill of `n` prompt rows in ONE forward: every layer over every row, with the
        softmax layers' quadratic term at Sq = Sk = n (SDPA is charged as Sq*Sk whether or not the
        mask is causal -- the hooks' convention, so the ceiling arm matches this)."""
        gdn = self.gdn_proj_tok() + self.gdn_conv_tok() + self.gdn_scan_tok()
        per_tok = (self.layers * self.mlp_tok() + self.n_linear * gdn
                   + self.n_full * self.full_proj_tok())
        return per_tok * n + self.n_full * self.attn(n, n)

    def corrected_row_flops(self, p: float) -> float:
        """One rewritten prompt row at position `p`: every layer's GEMMs once, and on the softmax
        layers an attention against the p+1 keys at or before it (write-before-read inside a round
        makes the row's own key visible, hence p+1). The GDN conv and scan are NOT here -- a
        corrected row's recurrent contribution comes from the round's window re-scan, which is
        charged once per round by `rescan_flops`."""
        return (self.layers * self.mlp_tok()
                + self.n_linear * self.gdn_proj_tok()
                + self.n_full * (self.full_proj_tok() + self.attn(1, p + 1)))

    def rescan_flops(self, window_len: float) -> float:
        """The round's Gated DeltaNet re-scan of `[s, e)`: conv + delta-rule scan over the whole
        window on every linear layer, however few of its rows the round actually corrected. This
        is real overhead of the schedule, not an implementation detail, so it is counted."""
        return self.n_linear * window_len * (self.gdn_conv_tok() + self.gdn_scan_tok())

    def lm_head_flops(self, rows: float = 1) -> float:
        """Reported separately and NOT added anywhere: the hooked reference installs on the vision
        tower + language model, so `lm_head` is outside the subtree in every measured number."""
        return 2 * rows * self.hidden * self.vocab

    # -- the schedule ------------------------------------------------------------------------
    def interleaved_cost(self, n: int, lo: int, n_groups: int, chunks) -> Dict[str, float]:
        """Replay `stats["chunks"]` of an interleaved run -> {"total", "crit"} decoder FLOPs.

        `chunks` is what `qwen_vl_axis.streaming_forward(llm_schedule="interleaved")` records and
        the driver stores per row: `("approx", 0, n-1)` for the t=0 pass over the whole
        approximate prompt (the last row is held back), then `("correct", s, e, |P_r|)` per round.

        Critical = everything that can only start once the last band's pixels are in: the final
        round's corrected rows and its re-scan, plus the held-back row `n-1`, which the engine
        computes in the step that follows and which no arrival can precede. Earlier rounds and the
        approximate pass overlapped with transmission (`appcorr/flops/counter.py`'s rule).

        A round's rows are priced at the MEAN position of its window: `corrected_row_flops` is
        affine in `p`, so the sum over |P_r| rows spread across `[s, e)` is exact at keep=1 (the
        rows ARE the window) and unbiased under keep<1, where the selection within a band is not
        positionally ordered.
        """
        total = 0.0
        last = 0.0
        n_correct = 0
        for c in chunks:
            kind = c[0]
            if kind == "approx":
                s, e = int(c[1]), int(c[2])
                assert s == 0 and e == n - 1, f"approx pass {s, e} is not the held-back prompt {n}"
                cost = self.prefill_flops(e - s)
            elif kind == "correct":
                s, e, rows = int(c[1]), int(c[2]), int(c[3])
                assert lo <= s < e <= n - 1, f"window {s, e} outside [{lo}, {n - 1})"
                assert rows <= e - s, f"{rows} corrected rows in a {e - s}-row window"
                cost = rows * self.corrected_row_flops((s + e - 1) / 2.0) + self.rescan_flops(e - s)
                n_correct += 1
            else:
                raise ValueError(f"unknown chunk record {c!r}")
            total += cost
            last = cost
        assert n_correct >= 1, "an interleaved run has at least one `correct` round"
        assert n_groups > 0
        tail = self.corrected_row_flops(n - 1)      # the held-back row, in the first engine step
        return {"total": total + tail, "crit": last + tail}


def qwen35_from_config(path: str) -> Qwen35Decoder:
    """Build the entry from an HF `config.json` (the snapshot dirs under /NHNHOME/huggingface/hub);
    `MODELS35` below is what this returns for the three checkpoints, frozen so the module needs no
    filesystem."""
    import json as _json
    cfg = _json.load(open(path))
    t = cfg.get("text_config", cfg)
    lt = t.get("layer_types")
    if lt is None:                      # older configs only carry the interval
        iv = int(t["full_attention_interval"])
        lt = ["full_attention" if (i + 1) % iv == 0 else "linear_attention"
              for i in range(int(t["num_hidden_layers"]))]
    return Qwen35Decoder(
        layers=int(t["num_hidden_layers"]), hidden=int(t["hidden_size"]),
        heads=int(t["num_attention_heads"]), kv_heads=int(t["num_key_value_heads"]),
        head_dim=int(t.get("head_dim", t["hidden_size"] // t["num_attention_heads"])),
        n_full=sum(1 for x in lt if x == "full_attention"),
        n_linear=sum(1 for x in lt if x == "linear_attention"),
        lin_k_heads=int(t["linear_num_key_heads"]), lin_v_heads=int(t["linear_num_value_heads"]),
        lin_k_dim=int(t["linear_key_head_dim"]), lin_v_dim=int(t["linear_value_head_dim"]),
        conv_kernel=int(t["linear_conv_kernel_dim"]),
        num_experts=int(t.get("num_experts", 0)), top_k=int(t.get("num_experts_per_tok", 0)),
        moe_inter=int(t.get("moe_intermediate_size", 0)),
        shared_inter=int(t.get("shared_expert_intermediate_size", 0)),
        dense_inter=int(t.get("intermediate_size", 0)) if "num_experts" not in t else 0,
        vocab=int(t.get("vocab_size", 248320)))


# Read off the snapshots on 2026-09-10 (models--Qwen--Qwen3.5-{35B-A3B,122B-A10B-FP8,4B}).
MODELS35: Dict[str, Qwen35Decoder] = {
    "qwen35_35b": Qwen35Decoder(layers=40, hidden=2048, heads=16, kv_heads=2, head_dim=256,
                                n_full=10, n_linear=30, lin_k_heads=16, lin_v_heads=32,
                                lin_k_dim=128, lin_v_dim=128, conv_kernel=4,
                                num_experts=256, top_k=8, moe_inter=512, shared_inter=512),
    "qwen35_122b": Qwen35Decoder(layers=48, hidden=3072, heads=32, kv_heads=2, head_dim=256,
                                 n_full=12, n_linear=36, lin_k_heads=16, lin_v_heads=64,
                                 lin_k_dim=128, lin_v_dim=128, conv_kernel=4,
                                 num_experts=256, top_k=8, moe_inter=1024, shared_inter=1024),
    "qwen35_4b": Qwen35Decoder(layers=32, hidden=2560, heads=16, kv_heads=4, head_dim=256,
                               n_full=8, n_linear=24, lin_k_heads=16, lin_v_heads=32,
                               lin_k_dim=128, lin_v_dim=128, conv_kernel=4, dense_inter=9216),
}


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


# Prompt lengths of the 12 strided samples `flops_report_qwen35.load_samples` feeds -- the SAME
# images the hooked numbers below were measured on, recomputed on CPU from the HF processor
# (2026-09-10). Per-sample, not a mean: the softmax term is quadratic in N, so a mean of shapes
# would not be a mean of costs.
QWEN35_SHAPES = {
    "refcoco":     [325, 418, 319, 320, 395, 421, 361, 360, 361, 318, 358, 318],
    "chartqa":     [549, 545, 547, 458, 467, 466, 483, 465, 467, 466, 466, 463],
    "textvqa":     [705, 768, 803, 707, 1055, 1053, 708, 607, 609, 804, 703, 735],
    "realworldqa": [1363, 1337, 1359, 1362, 1344, 1397, 1361, 1813, 1340, 1340, 1791, 1344],
}
# Hooked means over those samples, GFLOPs. `full` = the ceiling forward, vision tower + decoder,
# from analysis/results/flops/inprocess_flops.json["qwen35_moe"] (sourced from
# qwen35_flops_attnfix_legacy4.json -- the 2026-08-31 re-measure WITH `hooks.patch_attention`;
# the older analysis/results/flops/qwen35_flops.json is the PRE-fix file and its `full` is 10-26%
# low, which shows up here as a decoder cost that falls with N). `V` = the tower alone, measured
# in isolation because `patch_attention` is a global SDPA patch
# (analysis/results/flops/qwen35_vision_share.json, same `load_samples(ds, 12)`). The decoder-side
# reference this module reconciles against is their difference.
QWEN35_MEASURED = {
    "refcoco":     {"full": 2948.4, "V": 1168.6},
    "chartqa":     {"full": 4347.5, "V": 1904.4},
    "textvqa":     {"full": 7520.5, "V": 3610.3},
    "realworldqa": {"full": 15855.5, "V": 8458.1},
}


def validate_qwen35(model_key: str = "qwen35_35b") -> float:
    """Gate F: the closed-form decoder prefill against the hooked ceiling minus the hooked tower.
    Returns the worst |calc/meas - 1|."""
    dec = MODELS35[model_key]
    print(f"{'dataset':<14}{'N (mean)':>10}{'meas full':>11}{'meas V':>10}{'meas L':>10}"
          f"{'calc L':>10}{'ratio':>9}")
    worst = 0.0
    for ds, ns in QWEN35_SHAPES.items():
        m = QWEN35_MEASURED[ds]
        meas = m["full"] - m["V"]
        calc = sum(dec.prefill_flops(n) for n in ns) / len(ns) / 1e9
        worst = max(worst, abs(calc / meas - 1))
        print(f"{ds:<14}{sum(ns) / len(ns):>10.1f}{m['full']:>11.1f}{m['V']:>10.1f}"
              f"{meas:>10.1f}{calc:>10.1f}{calc / meas:>9.5f}")
    print(f"\n  worst |calc/meas - 1| on the decoder prefill: {100 * worst:.3f}%  (gate F: < 1%)")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--validate-qwen35", action="store_true",
                    help="gate F for the Qwen3.5 hybrid entry: closed-form decoder prefill vs the "
                         "hooked ceiling minus the hooked vision tower")
    ap.add_argument("--model35", default="qwen35_35b", choices=sorted(MODELS35))
    ap.add_argument("--groups", type=int, default=4)
    a = ap.parse_args()

    if a.validate_qwen35:
        validate_qwen35(a.model35)

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
