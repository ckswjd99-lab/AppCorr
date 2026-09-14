"""Quick TTFT probe for the eval table's Lat. / Crit. Lat. columns (FLOPs-style, minutes not hours).

Runs `qwen_vllm_accuracy.py --concurrency 1` (no queueing, one request in flight) for the ceiling
and the streaming arms at each keep on a few evenly spaced images per dataset and pins the medians
into analysis/results/latency/inprocess_latency.json with the SAME key names as
analysis/results/flops/inprocess_flops.json, so make_eval_table.py reads both the same way:

  full          ceiling TTFT from t0 (one-shot tower + full prefill; 100% critical, the denominator)
  total_k{keep} streaming TTFT from t0 -- every pass serialized as if the whole image were present
                at t0 (the analogue of Comp.: total work, transmission overlap NOT credited)
  k{keep}       streaming TTFT measured from the moment the LAST image band's pixels are consumed,
                i.e. the start of its vision correction (the push of band g-2 precedes it; the push
                syncs the GPU) -- what a user waits after the final byte: last-band correction +
                its push + its prefill + the trailing text + the first decode step (the analogue
                of Crit. Comp., which charges the last band's correction the same way)

t0 = the driver clock at "inputs on cuda:0, before vision()" (`t_start` in the driver); TTFT ends
at the engine's first-token timestamp relayed through the bridge. Both include the driver-side
vision pass, the socket hop and the engine's chunked prefill; decode is excluded. The first
`--warmup` rows of every arm are dropped (cold kernels/allocator), medians over the rest.

Usage (122B on the live server, all seven table datasets):
  python analysis/experiments/latency_probe.py --family qwen35 --model Qwen/Qwen3.5-122B-A10B-FP8 \
      --port 5591 --key qwen35_122b --samples 40 --warmup 4 --keeps 1.0 0.50 0.25 \
      --datasets visdrone_det:pyr visdrone_count:pyr vstar:pyr textvqa:pyr refcoco:pyr \
                 realworldqa:box chartqa:box
The dataset:filter pairs follow the campaign's degrade-filter per dataset.

GLM-5.3-Flash is the same call with `--family glm53 --model zai-org/GLM-5.3-Flash --key glm53`
(slug `glm-5.3-flash`); note its server is the TP=2 one, so anchor the probe only after
`glm53_tp_gate.sh` arm B passes -- until then the numbers would be a TP config's, not ours.

GLM-4.6V (106B-A12B FP8, one B200, slug `glm-4.6v-fp8`) is the same call with the family and the
model swapped:
  python analysis/experiments/latency_probe.py --family glm46v --model zai-org/GLM-4.6V-FP8 \
      --port 5591 --key glm46v --samples 36 --warmup 4 --keeps 1.0 0.50 0.25 \
      --push-delay-ms 150 --llm-schedule interleaved --datasets vstar:pyr realworldqa:box
"""
import argparse, json, os, statistics as st, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Mirrors `qwen_vllm_accuracy.SCHEDULE_TAG` (copied, not imported: that module pulls in torch and
# this one only launches subprocesses). Keep the two in step.
SCHEDULE_TAG = {"unified_staged": "interleaved_unified"}


def parse_keep(s: str):
    """`--keeps` token: a float budget, or `auto:<theta>` for the bucketized-threshold arm."""
    if s.startswith("auto:"):
        return ("auto", float(s[5:]))
    return float(s)


def ktag(k) -> str:
    """Row-file / json key suffix: `k0.50` for a fixed keep, `auto0.0251` for theta (driver's arm_tag)."""
    return (f"auto{k[1]:g}" + ("_lat" if LATTICE else "")) if isinstance(k, tuple) else f"k{k:.2f}"


def klt1(k) -> bool:
    return isinstance(k, tuple) or k < 1.0


def kdriver(k) -> list:
    if isinstance(k, tuple):
        args = ["--keep", "auto", "--pscore-threshold", f"{k[1]}", "--pscore-bucket", "8",
                "--pscore-score", "rms"]
        if LATTICE:
            args += ["--pscore-lattice", LATTICE]
        return args
    return ["--keep", f"{k}"]


LATTICE: str = ""      # set from --pscore-lattice; forwarded to the driver for every auto keep
OUT_JSON = os.path.join(ROOT, "analysis", "results", "latency", "inprocess_latency.json")
README = [
    "Single-request (concurrency 1) time-to-first-token medians, ms, for the eval table's Lat. /",
    "Crit. Lat. columns. Keys mirror inprocess_flops.json: 'full' = ceiling TTFT from t0 (one-shot",
    "tower + full prefill, the denominator); 'total_k*' = streaming TTFT from t0 with every pass",
    "serialized (no transmission credit; analogue of Comp.); 'k*' = streaming TTFT from the start",
    "of the LAST band's vision correction (= the push of band g-2, which syncs the GPU): last-band",
    "correction + push + prefill + trailing text + first decode -- what waits on the final byte",
    "(analogue of Crit. Comp.). 'detail' keeps ttft_last_chunk_ms (from the last push alone) too.",
    "From 2026-09-09 'k*' is measured with the bands spaced 150 ms apart (APPCORR_PUSH_DELAY_MS,",
    "'k*_push_delay_ms'), anchored at the last band's pixel arrival (push g-2 issue + delay); the",
    "fast-producer value (chunks queue behind the previous step, tower overlaps the engine) lives",
    "in detail.streaming_k*.ttft_last_band_ms. 'full' and 'total_k*' are fast-producer numbers.",
    "t0 = driver inputs on the",
    "GPU before the vision pass; first token = engine timestamp via the bridge. Decode excluded.",
    "Each entry's 'detail' keeps the per-arm medians of the driver's timing fields.",
    "Reproduce: python analysis/experiments/latency_probe.py (see its docstring).",
]
FIELDS = ["t_vision_ms", "t_open_ms", "t_last_push_ms", "ttft_open_ms", "ttft_last_chunk_ms",
          "ttft_last_band_ms", "last_correct_step_ms", "ttft_start_ms", "t_client_done_ms",
          "prompt_tokens", "gen_tokens"]


def rows_of(path, warmup, groups, delay_ms=0.0):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "skip" not in r and r.get("ttft_start_ms") is not None]
    for r in rows:
        sched = str(r.get("llm_schedule", ""))
        if sched == "unified_staged" and len(r.get("t_bands_ms") or []) >= groups:
            # Unified axis: the bands before the crossing send nothing, so neither the push
            # list nor the correct list is one-per-band. The driver stamps the start of every
            # band's processing (t_bands_ms, after the spacing sleep) -- the same moment the
            # other conventions below reconstruct from their message times.
            tb = r["t_bands_ms"]
            tr, td = r.get("t_recv_ms") or [], r.get("t_done_ms") or []
            ts = r.get("t_sent_ms") or r.get("t_pushes_ms") or []
            r["ttft_last_band_ms"] = r["ttft_start_ms"] - tb[groups - 1]
            if ts and ts[-1] is not None:
                r["last_band_correct_ms"] = ts[-1] - tb[groups - 1]
            if tr and ts and tr[-1] is not None and ts[-1] is not None:
                r["last_chunk_wait_ms"] = tr[-1] - ts[-1]
            if td and tr and td[-1] is not None and tr[-1] is not None:
                r["last_correct_step_ms"] = td[-1] - tr[-1]
                r["last_chunk_to_ft_ms"] = r["ttft_start_ms"] - td[-1]
            continue
        if sched.startswith("interleaved"):
            # The interleaved schedule sends g+1 messages: the whole approximate prompt at t=0
            # and one `correct` per band. Its Crit. Lat. uses the STREAMING anchor -- the last
            # band's pixel arrival = the issue of the previous correct + the band spacing (so the
            # two schedules are compared on one convention; from that moment the driver still
            # owes the last band's vision correction, the correct message, its decoder step over
            # k/g of the image rows + the text suffix, and the first decode step). The
            # decomposition keeps transport (t_recv - t_send), the server's drain + correct step
            # (t_done - t_recv) and the hold-back step (first token - t_done) apart.
            tp = r.get("t_pushes_ms") or []
            ts = r.get("t_sent_ms") or tp
            tr, td = r.get("t_recv_ms") or [], r.get("t_done_ms") or []
            if len(tp) >= 2 and tp[-2] is not None and delay_ms > 0:
                r["ttft_last_band_ms"] = r["ttft_start_ms"] - tp[-2] - delay_ms
            elif len(ts) >= 2 and ts[-2] is not None:
                r["ttft_last_band_ms"] = r["ttft_start_ms"] - ts[-2]
            else:
                r["ttft_last_band_ms"] = r["ttft_start_ms"]
            if len(ts) >= 2 and ts[-1] is not None and tp[-2] is not None:
                r["last_band_correct_ms"] = ts[-1] - tp[-2] - delay_ms
            if tr and ts and tr[-1] is not None and ts[-1] is not None:
                r["last_chunk_wait_ms"] = tr[-1] - ts[-1]
            if td and tr and td[-1] is not None and tr[-1] is not None:
                r["last_correct_step_ms"] = td[-1] - tr[-1]
                r["last_chunk_to_ft_ms"] = r["ttft_start_ms"] - td[-1]
            elif tr and tr[-1] is not None:
                r["last_chunk_to_ft_ms"] = r["ttft_start_ms"] - tr[-1]
            continue
        if delay_ms > 0 and r.get("t_pushes_ms") and len(r["t_pushes_ms"]) >= groups:
            # spaced-arrival arm (APPCORR_PUSH_DELAY_MS): band g-1's correction starts when the
            # sleep after push g-2 ends = its pixels' arrival; the last chunk's transfer wait
            # and the last band's correction are kept separately for the decomposition
            tp, ts, tr = r["t_pushes_ms"], r["t_sent_ms"], r["t_recv_ms"]
            r["ttft_last_band_ms"] = r["ttft_start_ms"] - tp[groups - 2] - delay_ms
            r["last_band_correct_ms"] = ts[groups - 1] - tp[groups - 2] - delay_ms
            r["last_chunk_wait_ms"] = tr[groups - 1] - ts[groups - 1]
            r["last_chunk_to_ft_ms"] = r["ttft_start_ms"] - tr[groups - 1]
            r["max_chunk_wait_ms"] = max(b - a for a, b in zip(ts, tr))
            continue
        # first token (driver clock) minus the departure of band g-2's chunk = the moment the
        # GPU finished band g-2, i.e. the start of band g-1's correction (t_sent_ms; rows from
        # before 2026-09-09 only have the CPU issue time t_pushes_ms, which runs ahead of the
        # GPU); a one-chunk arm (ceiling) has no band structure -> from t0
        tp = r.get("t_sent_ms") or r.get("t_pushes_ms")
        if tp and len(tp) >= groups and tp[groups - 2] is not None:
            r["ttft_last_band_ms"] = r["ttft_start_ms"] - tp[groups - 2]
        else:
            r["ttft_last_band_ms"] = r["ttft_start_ms"]
    return rows[warmup:]


def med(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return round(st.median(v), 1) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen25vl", "qwen35", "glm46v", "glm53"], required=True,
                    help="forwarded to qwen_vllm_accuracy.py (which owns the axis dispatch)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=5591)
    ap.add_argument("--key", required=True, help="model key in inprocess_latency.json, e.g. qwen35_122b")
    ap.add_argument("--datasets", nargs="+", required=True, help="dataset[:degrade-filter] ...")
    ap.add_argument("--keeps", type=parse_keep, nargs="+", default=[1.0, 0.50, 0.25],
                    help="fixed keeps as floats, or `auto:<theta>` for the adaptive arm (keep=auto, "
                         "--pscore-threshold theta, bucket 8, rms score); keys are k0.50 / auto0.0251")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=None, help="driver output dir (default results/latency/probe_<key>)")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--pscore-lattice", type=str, default=None,
                    help="opt-in lattice mode for the auto keeps (forwarded to the driver as "
                         "--pscore-lattice; keys gain `_lat`)")
    ap.add_argument("--pscore", choices=["deferred", "eager"], default="deferred",
                    help="forwarded to the driver (keep<1 arms)")
    ap.add_argument("--push-delay-ms", type=float, default=0.0,
                    help="APPCORR_PUSH_DELAY_MS for the streaming arms: space the bands like a slow "
                         "link so chunk queueing / tower-engine overlap do not enter the critical "
                         "window (2026-09-09 rule: 150). Writes k{keep} from the pixel-arrival "
                         "anchor and detail streaming_k*_d<ms>; leaves 'full'/'total_k*' alone")
    ap.add_argument("--skip-ceiling", action="store_true", help="streaming arms only")
    ap.add_argument("--llm-schedule",
                    choices=["streaming", "interleaved", "interleaved_staged", "unified_staged"],
                    default="streaming",
                    help="forwarded to the driver; 'interleaved' probes the correct-op schedule "
                         "and reads its rows (whose arm files are named interleaved_*); "
                         "'unified_staged' is the tower-inside-the-staging form (memo §7.12), "
                         "whose rows are named interleaved_unified_*")
    a = ap.parse_args()
    global LATTICE
    LATTICE = a.pscore_lattice or ""
    out = a.out or os.path.join(ROOT, "analysis", "results", "latency", f"probe_{a.key}")
    os.makedirs(out, exist_ok=True)
    slug = a.model.split("/")[-1].lower()
    env = dict(os.environ, PYTHONPATH=ROOT, HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"))
    drv = [a.python, os.path.join(ROOT, "analysis", "experiments", "qwen_vllm_accuracy.py"),
           "--family", a.family, "--model", a.model, "--port", str(a.port), "--groups", str(a.groups),
           "--level", str(a.level), "--workers", str(a.workers), "--load", "vision",
           "--samples", str(a.samples), "--concurrency", "1", "--out", out, "--pscore", a.pscore]

    def arm_path(ds, arm, keep=None):
        suf = "" if arm != "streaming" else f"_g{a.groups}" + (f"_{ktag(keep)}" if klt1(keep) else "")
        # driver's `arm_tag` (incl. its schedule -> row-file-tag mapping)
        tag = SCHEDULE_TAG.get(a.llm_schedule, a.llm_schedule) if arm == "streaming" else arm
        return os.path.join(out, f"{ds}_{slug}_{tag}{suf}.jsonl")

    def run(ds, filt, arm, keep=1.0):
        p = arm_path(ds, arm, keep)
        if os.path.exists(p):
            os.remove(p)          # the driver resumes from existing rows; a probe must be fresh
        cmd = drv + ["--dataset", ds, "--degrade-filter", filt, "--arms", arm] + kdriver(keep) + [
            "--llm-schedule", a.llm_schedule]
        log = os.path.join(out, f"log_{ds}_{arm}_{ktag(keep)}.log")
        e = dict(env)
        if arm == "streaming" and a.push_delay_ms > 0:
            e["APPCORR_PUSH_DELAY_MS"] = f"{a.push_delay_ms:g}"
        t0 = time.time()
        with open(log, "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=e, cwd=ROOT,
                                 timeout=a.timeout)
        print(f"  {ds:15s} {arm:10s} {ktag(keep)} rc={rc} {time.time() - t0:5.0f}s", flush=True)

    specs = [(d.split(":")[0], d.split(":")[1] if ":" in d else "box") for d in a.datasets]
    if not a.aggregate_only:
        for ds, filt in specs:
            if not a.skip_ceiling:
                run(ds, filt, "ceiling")
            for k in a.keeps:
                run(ds, filt, "streaming", k)

    J = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {"_README": README}
    J["_README"] = README
    M = J.setdefault(a.key, {})
    M["_note"] = (f"{a.model} via the vLLM stream server on port {a.port}, --samples {a.samples} "
                  f"evenly spaced, --warmup {a.warmup} dropped, concurrency 1, groups {a.groups}, "
                  f"L{a.level}; per-dataset degrade filter as listed in each entry. "
                  f"Measured {time.strftime('%Y-%m-%d')}.")
    D = a.push_delay_ms
    dsuf = f"_d{D:g}" if D > 0 else ""
    for ds, filt in specs:
        e = M.setdefault(ds, {})
        e["filter"] = filt
        e.setdefault("detail", {})
        p = arm_path(ds, "ceiling")
        if os.path.exists(p):
            r = rows_of(p, a.warmup, a.groups)
            e["full"], e["n"] = med(r, "ttft_start_ms"), len(r)
            e["detail"]["ceiling"] = {k: med(r, k) for k in FIELDS}
        for k in a.keeps:
            p = arm_path(ds, "streaming", k)
            if not os.path.exists(p):
                continue
            r = rows_of(p, a.warmup, a.groups, D)
            kt = ktag(k)
            e[kt] = med(r, "ttft_last_band_ms")
            if isinstance(k, tuple):
                # adaptive arm: the per-image budget is in the rows, not the file name; keep the
                # realised-k distribution next to the latency so the two can be read together
                kr = sorted(float(x["keep_realised"]) for x in r if x.get("keep_realised") is not None)
                if kr:
                    e[f"{kt}_keep_realised"] = {"mean": sum(kr) / len(kr), "p05": kr[int(0.05 * len(kr))],
                                                "p50": kr[len(kr) // 2], "p95": kr[int(0.95 * len(kr))]}
                    e[f"{kt}_ttft_last_band_p95_ms"] = sorted(
                        float(x["ttft_last_band_ms"]) for x in r)[int(0.95 * len(r))]
            if D > 0:
                # spaced arrival: TTFT-from-t0 contains the injected spacing, so 'total_k*'
                # (the everything-serialized Lat.) keeps its fast-producer value
                e[f"{kt}_push_delay_ms"] = D
                e["detail"][f"streaming_{kt}{dsuf}"] = {kk: med(r, kk) for kk in FIELDS + [
                    "last_band_correct_ms", "last_chunk_wait_ms", "last_chunk_to_ft_ms",
                    "max_chunk_wait_ms"]}
            else:
                e[f"total_{kt}"] = med(r, "ttft_start_ms")
                e["detail"][f"streaming_{kt}"] = {kk: med(r, kk) for kk in FIELDS}
        full = e.get("full")
        print(f"{ds:15s} full {full} ms | " + " | ".join(
            f"{ktag(k)}: total {e.get(f'total_{ktag(k)}')} crit {e.get(ktag(k))}" for k in a.keeps)
              + (f" | n={e.get('n')}" if full else ""), flush=True)
    json.dump(J, open(OUT_JSON, "w"), indent=1)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
