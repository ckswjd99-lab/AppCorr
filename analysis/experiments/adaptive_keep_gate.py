"""Gates for `keep="auto"` -- bucketized threshold selection ("adaptive k"), CPU only.

The adaptive arm changes exactly ONE thing about the keep<1 path: how many groups band r
corrects. The score, the candidates, the top-by-score pick, the merge and the messages are the
fixed arm's. These gates are the assertions that make that sentence true rather than intended:

  A  identity with the fixed quota. With `pscore_score="mse"` (the fixed arm's own score) at a
     theta whose bucketized count m' equals the fixed quota in EVERY band, the adaptive arm and
     the fixed arm must select the SAME groups and hand the LLM BITWISE the same embeddings.
     Thetas are found by scanning, not asserted a priori: a global theta that lands on an equal
     per-band count is a property of the score distribution, so the gate reports how many it
     found and fails if it found none.
  B  the limits. theta -> +inf selects exactly ceil(G_r / bucket) groups in every band (the
     floor: a band the threshold rejects entirely still corrects its best 1/bucket) and
     theta -> -inf/0 selects all of them, at which point the embeddings must agree with the
     keep=1.0 arm.
  C  the lattice. Every band's count is on the 1/bucket lattice and `bucket_quota` agrees with
     the count the axis actually used, at bucket in {4, 8} and on a band size that is NOT a
     multiple of the bucket (where the ceilings bite).
  D  the raw-pixel scale. The processor's own constants (read from a LOADED processor, not
     hard-coded) and the channel-major patch layout `_pixel_std` assumes, checked against
     `pixel_values` of an image with known per-channel levels.

A-C run on the randomly initialised CPU models of qwen_axis_cpu_unittest (real HF classes, real
forks, fp32): the quota rule is integer logic, so that is where it is cheapest to gate. D needs
only a processor (config json, no weights, no GPU).

  PYTHONPATH=$PWD python analysis/experiments/adaptive_keep_gate.py --family qwen35
  PYTHONPATH=$PWD HF_HUB_OFFLINE=1 HF_HUB_CACHE=/NHNHOME/huggingface/hub \\
      python analysis/experiments/adaptive_keep_gate.py --processor Qwen/Qwen3.5-35B-A3B
"""
import argparse, json, math, os, sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from appcorr.models.qwen_vl_axis import bucket_quota          # noqa: E402
from analysis.experiments.qwen_vllm_accuracy import make_axis  # noqa: E402
from analysis.experiments.qwen_axis_cpu_unittest import make_inputs, tiny_models  # noqa: E402


def varied_inputs(model, h, w, seed):
    """`make_inputs` with a residual whose SIZE varies per merge group.

    Its own residual is i.i.d. gaussian over ~6k elements per group, so every group's energy
    concentrates within a couple of percent of 1 after the mean-1 normalisation -- a score
    distribution so flat that a global threshold is either above all of it or below all of it,
    and the identity gate would only ever see its two trivial limits. Scaling each group's
    residual by a per-group factor gives the spread a real image has.
    """
    inputs, px_base = make_inputs(model, h, w, seed)
    unit = int(model.config.vision_config.spatial_merge_size) ** 2
    n_groups = (h * w) // unit
    g = torch.Generator().manual_seed(seed + 1000)
    scale = (0.2 + 1.8 * torch.rand(n_groups, 1, generator=g)).repeat_interleave(unit, 0)
    px_full = inputs["pixel_values"]
    return inputs, px_full + scale * (px_base - px_full)


def scan_thetas(f, max_evals=240):
    """Every distinct per-band count pattern the threshold can produce, found by BISECTION.

    A log grid cannot be trusted here: where the score distribution is narrow, every transition
    hides between two grid points (measured: a fixed grid of 115 thetas found 3 of the patterns
    on the tiny model, the bisection finds all of them). `f(theta) -> tuple(counts)` is treated
    as a black box -- the gate never reads the axis's score vector, so the quantity under test
    is not shared with the harness that picks the thresholds.
    """
    seen, evals = {}, [0]

    def ev(t):
        if t not in seen:
            evals[0] += 1
            seen[t] = f(t)
        return seen[t]

    def split(lo, hi, d=0):
        if evals[0] >= max_evals or d > 48 or ev(lo) == ev(hi):
            return
        mid = math.sqrt(lo * hi) if lo > 0 else (lo + hi) / 2.0
        if mid <= lo or mid >= hi:
            return
        split(lo, mid, d + 1)
        split(mid, hi, d + 1)

    ev(float("inf")); ev(-float("inf")); ev(0.0)
    split(1e-12, 1e12)
    return sorted(seen.items(), key=lambda kv: kv[0])


def bands_of(axis, groups, n_groups):
    return [(a, b) for a, b in axis._bands(groups, n_groups) if b > a]


@torch.no_grad()
def run(axis, inputs, px_base, groups, keep, **knobs):
    for k, v in knobs.items():
        setattr(axis, k, v)
    lg, kv, st = axis.streaming_forward(inputs, px_base, groups, keep=keep)
    sel = torch.cat([g.cpu() for g in st["group_idx"]]).sort().values if st["group_idx"] else \
        torch.empty(0, dtype=torch.long)
    return st, sel


def bitwise(a, b):
    return bool(a.shape == b.shape and torch.equal(a, b))


def gate_abc(axis, inputs, px_base, groups, bucket, report, tag):
    n_groups = int(inputs["mm_token_type_ids"].sum())
    bands = bands_of(axis, groups, n_groups)
    sizes = [b - a for a, b in bands]
    row = {"case": tag, "groups": groups, "bucket": bucket, "n_groups": n_groups,
           "band_sizes": sizes}

    # -- B: the two limits -------------------------------------------------------------------
    st_hi, sel_hi = run(axis, inputs, px_base, groups, "auto",
                        pscore_threshold=float("inf"), pscore_bucket=bucket, pscore_score="mse")
    floor = [bucket_quota(0, g, bucket) for g in sizes]
    row["theta_inf_counts"] = list(st_hi["band_selected"])
    row["theta_inf_expected"] = floor
    row["B_floor_ok"] = list(st_hi["band_selected"]) == floor == [-(-g // bucket) for g in sizes]
    row["theta_inf_keep_realised"] = st_hi["keep_realised"]

    st_lo, sel_lo = run(axis, inputs, px_base, groups, "auto",
                        pscore_threshold=-float("inf"), pscore_bucket=bucket, pscore_score="mse")
    row["theta_neginf_counts"] = list(st_lo["band_selected"])
    row["B_all_ok"] = list(st_lo["band_selected"]) == sizes and st_lo["keep_realised"] == 1.0
    st_one, _ = run(axis, inputs, px_base, groups, 1.0)
    row["B_all_embeds_bitwise_vs_keep1"] = bitwise(st_lo["image_embeds"], st_one["image_embeds"])
    row["B_all_embeds_max_abs"] = float((st_lo["image_embeds"].float()
                                         - st_one["image_embeds"].float()).abs().max())
    # theta = 0 is the same limit for a score that is a product of two non-negative factors
    st_z, _ = run(axis, inputs, px_base, groups, "auto", pscore_threshold=0.0,
                  pscore_bucket=bucket, pscore_score="mse")
    row["B_theta0_all"] = list(st_z["band_selected"]) == sizes

    # -- C: the lattice + monotonicity in theta ----------------------------------------------
    lattice_ok, quota_ok = True, True

    def counts_at(th):
        nonlocal lattice_ok, quota_ok
        st, _ = run(axis, inputs, px_base, groups, "auto", pscore_threshold=th,
                    pscore_bucket=bucket, pscore_score="mse")
        c, over = list(st["band_selected"]), list(st["band_over"])
        for m, o, g in zip(c, over, sizes):
            if m != bucket_quota(o, g, bucket):
                quota_ok = False
            # on the lattice: m' = ceil(q * g / bucket) for some q in [1, bucket]
            if m not in {-(-q * g // bucket) for q in range(1, bucket + 1)}:
                lattice_ok = False
        return tuple(c)

    scan = scan_thetas(counts_at)
    thetas = [t for t, _ in scan]
    counts = [list(c) for _, c in scan]
    row["C_quota_matches_rule"] = quota_ok
    row["C_on_lattice"] = lattice_ok
    # theta ascending -> counts non-increasing, band by band
    row["C_monotone_in_theta"] = all(
        all(x >= y for x, y in zip(counts[i], counts[i + 1])) for i in range(len(counts) - 1))
    row["C_thetas_evaluated"] = len(scan)
    row["C_distinct_count_patterns"] = len({tuple(c) for c in counts})
    row["C_realised_k_range"] = [round(min(sum(c) for c in counts) / n_groups, 4),
                                 round(max(sum(c) for c in counts) / n_groups, 4)]

    # -- A: identity with the fixed quota ------------------------------------------------------
    matches, fails = [], []
    seen = set()
    for th, c in zip(thetas, counts):
        if len(set(c)) != 1 or tuple(c) in seen:
            continue
        v = c[0]
        n_sel = v * len(bands)
        keep = n_sel / n_groups
        # the fixed path's own split of round(keep * G) must reproduce [v] * groups exactly
        ns = max(1, int(round(keep * n_groups)))
        fixed_q = [ns // groups + (1 if r < ns % groups else 0) for r in range(groups)]
        if fixed_q != [v] * groups:
            continue
        seen.add(tuple(c))
        st_a, sel_a = run(axis, inputs, px_base, groups, "auto", pscore_threshold=th,
                          pscore_bucket=bucket, pscore_score="mse")
        st_f, sel_f = run(axis, inputs, px_base, groups, keep)
        ok_sel = bitwise(sel_a, sel_f)
        ok_emb = bitwise(st_a["image_embeds"], st_f["image_embeds"])
        ok_cg = int(st_a["corrected_groups"]) == int(st_f["corrected_groups"]) == n_sel
        (matches if (ok_sel and ok_emb and ok_cg) else fails).append(
            {"theta": th, "per_band": v, "keep": round(keep, 6), "n_selected": int(sel_a.numel()),
             "same_groups": ok_sel, "embeds_bitwise": ok_emb, "corrected_groups_ok": ok_cg})
    row["A_matches"] = matches
    row["A_failures"] = fails
    # the two limits are matches too, but they are gate B; gate A is only meaningful if a
    # NON-trivial quota (neither the floor nor everything) was reproduced bitwise
    row["A_nontrivial"] = [m for m in matches
                           if m["per_band"] not in (floor[0], sizes[0])]
    row["A_ok"] = bool(row["A_nontrivial"]) and not fails

    # -- smoke: the rms score runs and is a different ranking ---------------------------------
    st_r, sel_r = run(axis, inputs, px_base, groups, "auto", pscore_threshold=float("inf"),
                      pscore_bucket=bucket, pscore_score="rms")
    row["rms_units_no_processor"] = st_r["pscore_rms_units"]
    row["rms_floor_counts"] = list(st_r["band_selected"])
    row["rms_selection_differs_from_mse"] = not bitwise(sel_r, sel_hi)

    row["PASS"] = bool(row["B_floor_ok"] and row["B_all_ok"] and row["B_theta0_all"]
                       and row["C_quota_matches_rule"] and row["C_on_lattice"]
                       and row["C_monotone_in_theta"] and row["A_ok"]
                       and row["B_all_embeds_bitwise_vs_keep1"])
    report.append(row)
    print(f"[{tag}] g={groups} bucket={bucket} bands={sizes}  "
          f"A: {len(matches)} theta(s) reproduce the fixed quota bitwise "
          f"({len(row['A_nontrivial'])} non-trivial, k="
          f"{[m['keep'] for m in row['A_nontrivial']]}), {len(fails)} fail  "
          f"B: floor={row['theta_inf_counts']} (want {floor}), all={row['B_all_ok']} "
          f"embeds_vs_k1={row['B_all_embeds_bitwise_vs_keep1']}  "
          f"C: lattice={row['C_on_lattice']} rule={row['C_quota_matches_rule']} "
          f"monotone={row['C_monotone_in_theta']} patterns={row['C_distinct_count_patterns']}"
          f"  -> {'PASS' if row['PASS'] else 'FAIL'}", flush=True)
    return row


def gate_d(model_id, report):
    """The processor's constants and the channel-major patch layout, from a LOADED processor."""
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    ip = proc.image_processor
    mean = [float(v) for v in ip.image_mean]
    std = [float(v) for v in ip.image_std]
    levels = (10, 120, 240)                    # a flat RGB image: one known level per channel
    img = Image.fromarray(np.stack([np.full((224, 224), v, np.uint8) for v in levels], -1), "RGB")
    px = proc.image_processor(images=[img], return_tensors="pt")["pixel_values"]
    P, T = int(ip.patch_size), int(ip.temporal_patch_size)
    blk = T * P * P
    got = [float(px[0, c * blk:(c + 1) * blk].mean()) for c in range(3)]
    want = [(v * float(ip.rescale_factor) - mean[c]) / std[c] for c, v in enumerate(levels)]
    row = {"case": "D_processor", "model": model_id, "processor": type(ip).__name__,
           "image_mean": mean, "image_std": std, "rescale_factor": float(ip.rescale_factor),
           "do_rescale": bool(ip.do_rescale), "do_normalize": bool(ip.do_normalize),
           "patch_size": P, "temporal_patch_size": T, "row_dim": int(px.shape[-1]),
           "channel_block": blk,
           "one_gray_level_in_normalised_units": [float(ip.rescale_factor) / s for s in std],
           "flat_image_levels": list(levels), "per_channel_block_mean": got,
           "per_channel_expected": want,
           "max_abs_layout_error": max(abs(a - b) for a, b in zip(got, want))}
    # the same vector `_pixel_std` builds, checked against the processor's constants directly
    from appcorr.models.qwen_vl_axis import QwenVLStreamingAxis

    class _Probe(QwenVLStreamingAxis):
        def __init__(self, p):
            torch.nn.Module.__init__(self)
            self.processor, self._pixel_std_cache = p, None
    vec = _Probe(proc)._pixel_std(int(px.shape[-1]), "cpu", torch.float32)
    ref = torch.tensor(std).repeat_interleave(blk)
    row["pixel_std_vector_ok"] = bool(torch.equal(vec, ref))
    row["PASS"] = bool(row["max_abs_layout_error"] < 1e-5 and row["pixel_std_vector_ok"])
    report.append(row)
    print(f"[D] {model_id}: {type(ip).__name__} mean={mean} std={std} "
          f"rescale={float(ip.rescale_factor):.8f} patch={P} T={T} row_dim={int(px.shape[-1])} "
          f"-> one gray level = {row['one_gray_level_in_normalised_units']} normalised units; "
          f"channel-major layout error {row['max_abs_layout_error']:.2e}; "
          f"_pixel_std matches std x{blk}: {row['pixel_std_vector_ok']}  "
          f"-> {'PASS' if row['PASS'] else 'FAIL'}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen25vl", "qwen35"], default="qwen35")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--buckets", type=int, nargs="+", default=[8, 4])
    ap.add_argument("--grids", type=int, nargs="+", default=[8, 16, 8, 12],
                    help="pairs (h, w) of patch rows; 8 16 tiles the bucket exactly (bands of 8), "
                         "8 12 does not (bands of 6, where the two ceilings bite)")
    ap.add_argument("--processor", default=None,
                    help="model id whose PROCESSOR (config only, no weights) gate D reads")
    ap.add_argument("--out", default="analysis/results/vllm_stream/adaptive_keep_gate.json")
    a = ap.parse_args()
    torch.manual_seed(0)

    model = tiny_models()[a.family]
    axis = make_axis(a.family, model, None)
    report = []
    grids = list(zip(a.grids[0::2], a.grids[1::2]))
    for (h, w) in grids:
        inputs, px_base = varied_inputs(model, h, w, 0)
        for bucket in a.buckets:
            gate_abc(axis, inputs, px_base, a.groups, bucket, report,
                     f"{a.family}-{h}x{w}")
    if a.processor:
        gate_d(a.processor, report)
    out = {"_family": a.family, "_groups": a.groups, "_grids": grids, "rows": report,
           "PASS": all(r["PASS"] for r in report)}
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(f"wrote {a.out}: {'ADAPTIVE_GATE_PASS' if out['PASS'] else 'ADAPTIVE_GATE_FAIL'}")
    sys.exit(0 if out["PASS"] else 1)


if __name__ == "__main__":
    main()
