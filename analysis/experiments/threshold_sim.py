"""Offline calibration of the "adaptive k" threshold: what k a given theta actually buys.

CPU. Reads one or more `pscore_dump.py` npz files and, for a grid of thetas and either energy
factor, replays the RUNTIME's per-band rule -- it imports `appcorr.models.qwen_vl_axis.
bucket_quota` rather than restating it, so the calibration and the arm cannot drift apart -- and
reports the realised k per image (mean, p95, min, max, and how often it pins to either clamp).
`--target-k` inverts it: the theta whose mean realised k equals a target, by bisection (the rule
is monotone non-increasing in theta, so the inverse is well defined up to the score lattice).

Two knobs decide what is being thresholded, and they must match the arm:

  --score {rms,mse}    rms = RMS residual in RAW [0,1] pixel units x mean-1 received attention
                       (absolute: the only one a GLOBAL theta is meaningful for).
                       mse = the fixed-k arm's per-image mean-1 energy x the same attention;
                       thresholding it only ever responds to the SHAPE of an image's score
                       distribution, never to how degraded its base is. Both at equal mean k is
                       the comparison the design memo asks for.
  --pscore {deferred,eager}   deferred (the campaign default) ranks BAND 0 on the energy factor
                       alone -- the attention column sum has not run when band 0 is chosen -- so
                       band 0 is thresholded on a different quantity from bands 1... This is the
                       arm's existing signal choice, not something adaptive k introduces; it is
                       simulated, not averaged away.

  PYTHONPATH=$PWD python analysis/experiments/threshold_sim.py \\
      --npz analysis/results/pscore_dump/vstar_qwen3.5-35b-a3b_g4_l2pyr_n36.npz \\
      --target-k 0.5 0.25
  PYTHONPATH=$PWD python analysis/experiments/threshold_sim.py --selftest
"""
import argparse, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from appcorr.models.qwen_vl_axis import bucket_quota, bucket_quota_lattice   # noqa: E402  (the runtime's own rules)

LATTICE = None          # --lattice: captured graph sizes; the LAST band is lifted onto them
SUFFIX = {}             # --suffix-json: image_id -> text-suffix rows of the final round


def _quota(n_over, n_band, bucket, img, r, last):
    if LATTICE and r == last:
        return bucket_quota_lattice(n_over, n_band, bucket, LATTICE, SUFFIX.get(int(img["image_id"]), 0))
    return bucket_quota(n_over, n_band, bucket)


# --------------------------------------------------------------------------------------------- #
# the simulator
# --------------------------------------------------------------------------------------------- #

def load_npz(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    offs = z["offsets"]
    per_image = []
    for j in range(len(offs) - 1):
        a, b = int(offs[j]), int(offs[j + 1])
        per_image.append({"rms": z["rms"][a:b], "mse": z["mse"][a:b], "attn": z["attn"][a:b],
                          "band": z["band"][a:b], "image_id": int(z["image_id"][j])})
    return meta, per_image


def image_scores(img, score: str, pscore: str):
    """[(band_id, scores)] -- the vector each band is thresholded on, the arm's own choice."""
    energy = img["rms"] if score == "rms" else img["mse"]
    full = energy * img["attn"]
    out = []
    for r in sorted(set(int(b) for b in img["band"])):
        m = img["band"] == r
        # deferred: band 0 is chosen before the attention column sum has run
        out.append((r, (energy if (pscore == "deferred" and r == 0) else full)[m]))
    return out


def realised_k(images, theta: float, bucket: int, score: str, pscore: str) -> np.ndarray:
    """Per-image realised k under the bucket rule: selected groups / groups."""
    out = np.empty(len(images), np.float64)
    for j, img in enumerate(images):
        sel = tot = 0
        bands = image_scores(img, score, pscore)
        last = max((r for r, s in bands if s.size), default=-1)
        for r, s in bands:
            n_over = int(np.count_nonzero(s >= theta))
            sel += _quota(n_over, s.size, bucket, img, r, last)
            tot += s.size
        out[j] = sel / max(1, tot)
    return out


def stats(k: np.ndarray, floor_k: float) -> dict:
    return {"mean": float(k.mean()), "p95": float(np.percentile(k, 95)),
            "p05": float(np.percentile(k, 5)), "min": float(k.min()), "max": float(k.max()),
            "at_floor": float(np.mean(k <= floor_k + 1e-12)),
            "at_one": float(np.mean(k >= 1.0 - 1e-12))}


def floor_of(images, bucket: int) -> float:
    """The smallest k the rule can produce for this set (theta -> inf), as a mean over images."""
    ks = []
    for img in images:
        sel = tot = 0
        rs = sorted(set(int(b) for b in img["band"]))
        last = max((r for r in rs if np.count_nonzero(img["band"] == r)), default=-1)
        for r in rs:
            g = int(np.count_nonzero(img["band"] == r))
            sel += _quota(0, g, bucket, img, r, last)
            tot += g
        ks.append(sel / max(1, tot))
    return float(np.mean(ks))


def solve_theta(images, target: float, bucket: int, score: str, pscore: str,
                lo=1e-9, hi=1e9, iters=60):
    """Theta whose MEAN realised k is `target` (bisection; k is non-increasing in theta)."""
    f = lambda t: float(realised_k(images, t, bucket, score, pscore).mean())
    if f(lo) < target or f(hi) > target:
        return None, f(lo), f(hi)
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        if f(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo, f(lo), f(hi)


def theta_grid(images, score, pscore, n=25):
    """A log grid that spans the score distribution actually present (percentile-anchored:
    a fixed decade grid steps straight over a narrow distribution)."""
    allv = np.concatenate([s for img in images for _, s in image_scores(img, score, pscore)])
    allv = allv[allv > 0]
    if allv.size == 0:
        return np.array([1.0])
    lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.9)
    return np.geomspace(max(lo, 1e-12), max(hi, lo * 10), n)


# --------------------------------------------------------------------------------------------- #
# the text report
# --------------------------------------------------------------------------------------------- #

def table(meta, images, bucket, scores, pscore, targets, grid_n):
    L = []
    n_g = [sum(1 for _ in img["band"]) for img in images]
    L.append(f"# {meta['dataset']}  {meta['model']}  g={meta['groups']}  "
             f"level {meta['level']} / {meta['degrade_filter']}  "
             f"{len(images)} images, {sum(n_g)} merge groups "
             f"({min(n_g)}-{max(n_g)} per image)")
    L.append(f"# rms units: {meta['rms_units']}  image_std={meta['image_std']}  "
             f"bucket=1/{bucket}  pscore={pscore}")
    fl = floor_of(images, bucket)
    L.append(f"# floor (theta -> inf) mean k = {fl:.4f};  theta -> 0 gives k = 1.0")
    for score in scores:
        L.append("")
        L.append(f"## score = {score} x attn"
                 + ("  (band 0: energy alone -- deferred)" if pscore == "deferred" else ""))
        L.append(f"{'theta':>12} {'mean k':>8} {'p05':>7} {'p95':>7} {'min':>7} {'max':>7} "
                 f"{'@floor':>7} {'@1.0':>6}")
        for th in theta_grid(images, score, pscore, grid_n):
            st = stats(realised_k(images, th, bucket, score, pscore), fl)
            L.append(f"{th:>12.4g} {st['mean']:>8.4f} {st['p05']:>7.3f} {st['p95']:>7.3f} "
                     f"{st['min']:>7.3f} {st['max']:>7.3f} {st['at_floor']:>7.2f} "
                     f"{st['at_one']:>6.2f}")
        for tgt in targets:
            th, klo, khi = solve_theta(images, tgt, bucket, score, pscore)
            if th is None:
                L.append(f"  target mean k = {tgt:.2f}: UNREACHABLE "
                         f"(k spans {khi:.4f}..{klo:.4f} over the bisection bracket)")
                continue
            k = realised_k(images, th, bucket, score, pscore)
            st = stats(k, fl)
            L.append(f"  target mean k = {tgt:.2f}  ->  theta = {th:.6g}   "
                     f"realised mean {st['mean']:.4f}  p05 {st['p05']:.3f}  p95 {st['p95']:.3f}  "
                     f"[{st['min']:.3f}, {st['max']:.3f}]  at floor {st['at_floor']*100:.0f}%  "
                     f"at 1.0 {st['at_one']*100:.0f}%")
    return "\n".join(L)


# --------------------------------------------------------------------------------------------- #
# unit tests (synthetic scores -- no dump needed)
# --------------------------------------------------------------------------------------------- #

def synth(n_bands=4, per_band=8, seed=0, scale=None):
    rng = np.random.default_rng(seed)
    band = np.repeat(np.arange(n_bands, dtype=np.int16), per_band)
    n = band.size
    return {"rms": rng.uniform(0.001, 0.2, n).astype(np.float32),
            "mse": rng.uniform(0.1, 4.0, n).astype(np.float32),
            "attn": rng.uniform(0.2, 3.0, n).astype(np.float32),
            "band": band, "image_id": 0}


def selftest() -> int:
    bad = []

    def chk(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
        if not cond:
            bad.append(name)

    # -- the bucket rule itself -------------------------------------------------------------
    chk("ceiling, band size on the lattice (8 groups, bucket 8)",
        [bucket_quota(m, 8, 8) for m in range(9)] == [1, 1, 2, 3, 4, 5, 6, 7, 8],
        str([bucket_quota(m, 8, 8) for m in range(9)]))
    chk("ceiling, band size OFF the lattice (6 groups, bucket 8)",
        [bucket_quota(m, 6, 8) for m in range(7)] == [1, 2, 3, 3, 5, 6, 6],
        str([bucket_quota(m, 6, 8) for m in range(7)]))
    chk("ceiling, 12 groups bucket 8",
        [bucket_quota(m, 12, 8) for m in range(13)] == [2, 2, 3, 3, 5, 6, 6, 8, 9, 9, 11, 12, 12],
        str([bucket_quota(m, 12, 8) for m in range(13)]))
    chk("lower clamp: 0 over threshold still corrects ceil(G/bucket)",
        all(bucket_quota(0, g, 8) == -(-g // 8) for g in range(1, 65)))
    chk("upper clamp: never above the band, never below 1",
        all(1 <= bucket_quota(m, g, 8) <= g
            for g in range(1, 65) for m in range(0, g + 1)))
    chk("all over threshold -> the whole band",
        all(bucket_quota(g, g, 8) == g for g in range(1, 65)))
    chk("monotone non-decreasing in the count",
        all(bucket_quota(m, g, b) <= bucket_quota(m + 1, g, b)
            for b in (2, 4, 8, 16) for g in range(1, 33) for m in range(g)))
    chk("on the lattice for every band size / bucket",
        all(bucket_quota(m, g, b) in {-(-q * g // b) for q in range(1, b + 1)}
            for b in (2, 4, 8, 16) for g in range(1, 33) for m in range(g + 1)))
    chk("bucket=1 is all-or-the-whole-band", all(bucket_quota(m, g, 1) == g
                                                 for g in range(1, 17) for m in range(g + 1)))
    chk("empty band", bucket_quota(0, 0, 8) == 0)

    # -- the simulator on synthetic scores ---------------------------------------------------
    imgs = [synth(4, 8, s) for s in range(7)]
    for score in ("rms", "mse"):
        for pscore in ("deferred", "eager"):
            k_inf = realised_k(imgs, np.inf, 8, score, pscore)
            k_zero = realised_k(imgs, 0.0, 8, score, pscore)
            chk(f"theta -> inf gives 1/bucket ({score}/{pscore})",
                np.allclose(k_inf, 1 / 8), f"{k_inf[:3]}")
            chk(f"theta -> 0 gives 1.0 ({score}/{pscore})", np.allclose(k_zero, 1.0))
            ths = np.geomspace(1e-6, 1e3, 40)
            ks = np.stack([realised_k(imgs, t, 8, score, pscore) for t in ths])
            chk(f"monotone non-increasing in theta ({score}/{pscore})",
                bool(np.all(np.diff(ks, axis=0) <= 1e-12)))
            chk(f"k is always on the per-image lattice ({score}/{pscore})",
                bool(np.all(np.isin(np.round(ks * 32).astype(int),
                                    np.arange(4, 33)))))  # 4 bands x [1..8]
    # a band NOT a multiple of the bucket (6 groups): floor is ceil(6/8) = 1 of 24
    imgs6 = [synth(4, 6, s) for s in range(5)]
    chk("off-lattice band floor", np.allclose(realised_k(imgs6, np.inf, 8, "rms", "eager"), 4 / 24))
    chk("off-lattice band ceiling", np.allclose(realised_k(imgs6, 0.0, 8, "rms", "eager"), 1.0))
    chk("floor_of matches theta -> inf",
        abs(floor_of(imgs6, 8) - float(realised_k(imgs6, np.inf, 8, "rms", "eager").mean())) < 1e-12)

    # -- the inverse -------------------------------------------------------------------------
    for tgt in (0.5, 0.25):
        th, _, _ = solve_theta(imgs, tgt, 8, "rms", "eager")
        got = float(realised_k(imgs, th, 8, "rms", "eager").mean()) if th else float("nan")
        chk(f"solve_theta hits mean k = {tgt}", th is not None and abs(got - tgt) <= 0.03,
            f"theta={th:.6g} -> {got:.4f}" if th else "unreachable")
    th, _, _ = solve_theta(imgs, 0.01, 8, "rms", "eager")
    chk("a target below the floor is reported unreachable, not faked", th is None)

    # -- deferred vs eager really is a different band-0 signal --------------------------------
    d = realised_k(imgs, 1.0, 8, "mse", "deferred")
    e = realised_k(imgs, 1.0, 8, "mse", "eager")
    chk("deferred band 0 differs from eager", bool(np.any(d != e)), f"{d[:3]} vs {e[:3]}")
    print(f"\n{'THRESHOLD_SIM_SELFTEST_PASS' if not bad else 'THRESHOLD_SIM_SELFTEST_FAIL ' + str(bad)}")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="*", default=[])
    ap.add_argument("--bucket", type=int, default=8)
    ap.add_argument("--scores", nargs="+", choices=["rms", "mse"], default=["rms", "mse"])
    ap.add_argument("--pscore", choices=["deferred", "eager"], default="deferred")
    ap.add_argument("--target-k", type=float, nargs="*", default=[0.5, 0.25])
    ap.add_argument("--grid", type=int, default=25)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None, help="also write the text table here")
    ap.add_argument("--lattice", type=str, default=None,
                    help="opt-in: comma-separated captured graph sizes; the last band's count is "
                         "lifted onto them with the image's text suffix (bucket_quota_lattice), "
                         "exactly as the runtime does with axis.pscore_lattice")
    ap.add_argument("--suffix-json", type=str, default=None,
                    help="{dataset: {image_id: text-suffix rows}} for --lattice (final-round rows "
                         "besides the band's groups)")
    a = ap.parse_args()
    global LATTICE, SUFFIX
    if a.lattice:
        LATTICE = tuple(sorted({int(v) for v in a.lattice.split(",") if v.strip()}))
    if a.suffix_json:
        with open(a.suffix_json) as fh:
            allsuf = json.load(fh)
        SUFFIX = {}
        for path in a.npz:
            meta_ds = json.loads(str(np.load(path, allow_pickle=False)["meta"])).get("dataset")
            SUFFIX.update({int(k): int(v) for k, v in allsuf.get(meta_ds, {}).items()})
    if a.selftest:
        sys.exit(selftest())
    if not a.npz:
        raise SystemExit("--npz (a pscore_dump.py file) or --selftest")
    txt = []
    for p in a.npz:
        meta, images = load_npz(p)
        txt.append(table(meta, images, a.bucket, a.scores, a.pscore, a.target_k, a.grid))
    out = "\n\n".join(txt)
    print(out)
    if a.out:
        open(a.out, "w").write(out + "\n")


if __name__ == "__main__":
    main()
