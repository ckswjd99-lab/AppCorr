"""Emit the evaluation table as LaTeX from whatever results exist on disk.

Run it any time. Cells with no measurement print `--`, so the table is always current and never
carries a number that was typed by hand -- which is the point: transcribing these by eye already
put a wrong MMMU row into one progress report.

    python analysis/experiments/make_eval_table.py                  # LaTeX, keeps 25/50
    python analysis/experiments/make_eval_table.py --keeps 0.30 0.50
    python analysis/experiments/make_eval_table.py --format md      # readable while working
    python analysis/experiments/make_eval_table.py --status         # what is still missing
    python analysis/experiments/make_eval_table.py --table latency  # the latency table (2026-09-09
                                                                    # split; --with-latency keeps
                                                                    # the old combined eval form)

Where the numbers come from:

  accuracy   analysis/results/{model}_{dataset}/{arm}.json -> summary.accuracy, written by the
             oracle drivers. `ceiling`, `floor`, `interleaved_g{g}_k{keep}`.
  FLOPs      analysis/results/flops/*.json for the offload-driven models, written by the worker;
             analysis/results/flops/inprocess_flops.json for the ones driven in process.
  latency    analysis/results/latency/inprocess_latency.json (analysis/experiments/latency_probe.py):
             single-request TTFT medians, ms. Same key names as the FLOPs file -- `full` = ceiling
             TTFT from t0, `total_k*` = streaming TTFT from t0 (Lat., the Comp. analogue),
             `k*` = streaming TTFT from the previous band's chunk departure, i.e. its GPU
             completion (Crit. Lat., the Crit. Comp. analogue).
             Models without a measured serving path show "--". Rendered by `--table latency`
             (its own table, with the `detail` medians and the concurrency sweep from
             conc_sweep_20260909.json); the eval table itself carries accuracy + compute only.

Two conventions the table depends on, both enforced here rather than left to the reader:

  * Crit.\\ Comp. is per INSTRUCTION. Offload arms differ in batch size within a single model --
    NYU's ceiling runs at 1 and its interleaved arms at 8 -- so every offload value is divided by
    its recorded batch size. Skipping that made NYU's critical exceed its own ceiling.
  * A VFM runs one backbone regardless of task, so its rows repeat the same FLOPs by construction.
    They are emitted repeatedly because the table has one row per task, not because they were
    measured per task.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "analysis", "results")
FLOPS_DIR = os.path.join(RESULTS, "flops")
LAT_DIR = os.path.join(RESULTS, "latency")

# Ours accuracy cells at or above this preservation get a light-gray background (user request
# 2026-08-31). Needs \usepackage[table]{xcolor} in the paper preamble; emit_md strips the macro.
SHADE_PRES = 98.0
SHADE_MACRO = r"\cellcolor{gray!15}"

# Qwen3.5 pyr-filter campaign (2026-08-31, B200 box): jsonl per arm, appended incrementally with
# resume. A cell is emitted only when the arm is COMPLETE (every split index present), so a table
# generated mid-run never shows a partial number; oom-skips count toward completeness but are
# excluded from the mean. RefCOCO reports mean(ok) as Acc@0.5 and mean(val) as mIoU; TextVQA's
# headline is mean(val) -- the VQA soft score -- NOT the driver's running % (which prints the
# ok-rate, sc>=0.5).
QWEN35_PYR_DIR = os.path.join(RESULTS, "qwen35_accuracy_pyr")
QWEN35_PYR_EXPECTED = {"refcoco": 8811, "textvqa": 5000, "vstar": 191,
                       "visdrone_det": 448, "visdrone_count": 2350}
QWEN35_PYR_ROWS = {"RefCOCO val (Acc.@0.5)": ("refcoco", "ok"),
                   "RefCOCO val (mIoU)":     ("refcoco", "val"),
                   "TextVQA (VQA Acc.)":     ("textvqa", "val"),
                   "V*Bench (Acc.)":         ("vstar", "ok"),
                   "VisDrone Count (Exact Acc.)": ("visdrone_count", "ok"),
                   "VisDrone Count (Soft)":       ("visdrone_count", "val"),
                   "VisDrone Det (Acc.@0.5)":     ("visdrone_det", "ok"),
                   "VisDrone Det (mIoU)":         ("visdrone_det", "val")}

# Box-filter Qwen3.5 rows (RealWorldQA / ChartQA / VSR / MMVP / CV-Bench): same jsonl schema
# under analysis/results/qwen35_accuracy (35B: no slug; 122B: QWEN122B_SLUG). The LITERALS for
# these rows are the same files' historical values; the loader wins for every complete arm, so a
# re-measured arm (2026-09-09: keep<1 rows under the deferred patch score, eager files parked in
# _eager_pscore_20260909/) replaces its literal without a hand edit.
QWEN35_BOX_DIR = os.path.join(RESULTS, "qwen35_accuracy")
QWEN35_BOX_EXPECTED = {"realworldqa": 765, "chartqa": 2500, "vsr": 1222, "mmvp": 300,
                       "cvbench": 2638}
QWEN35_BOX_ROWS = {"RealWorldQA (Acc.)":      ("realworldqa", "ok"),
                   "ChartQA (Relaxed Acc.)":  ("chartqa", "ok"),
                   "VSR zeroshot (Acc.)":     ("vsr", "ok"),
                   "MMVP (Acc.)":             ("mmvp", "ok"),
                   "CV-Bench (Acc.)":         ("cvbench", "ok")}

# Muse Glimmer campaign jsonls (same schema, same driver family).
MG_PYR_DIR = os.path.join(RESULTS, "museglimmer_accuracy_pyr")
MG_PYR_EXPECTED = {"vstar": 191, "textvqa": 5000, "refcoco": 8811,
                   "visdrone_det": 448, "visdrone_count": 2350}
MG_PYR_ROWS = {"V*Bench (Acc.)":              ("vstar", "ok"),
               "TextVQA (VQA Acc.)":          ("textvqa", "val"),
               "RefCOCO val (Acc.@0.5)":      ("refcoco", "ok"),
               "RefCOCO val (mIoU)":          ("refcoco", "val"),
               "VisDrone Count (Exact Acc.)": ("visdrone_count", "ok"),
               "VisDrone Count (Soft)":       ("visdrone_count", "val"),
               "VisDrone Det (Acc.@0.5)":     ("visdrone_det", "ok"),
               "VisDrone Det (mIoU)":         ("visdrone_det", "val")}


# 122B probe (2026-09-01, NHN box): same jsonl schema, REDUCED SCALE (n=240 per arm, user-directed)
# under the Triton finegrained-fp8 fallback -- see the pilcrow footnote. Completeness is checked
# against the probe's own n, not the dataset size.
QWEN122B_DIR = os.path.join(RESULTS, "qwen35_122b_probe")
QWEN122B_EXPECTED = {"refcoco": 240, "textvqa": 240}
QWEN122B_SLUG = "_qwen3.5-122b-a10b-fp8"
QWEN35_4B_SLUG = "_qwen3.5-4b"        # the driver's slug for Qwen/Qwen3.5-4B
# 2026-09-08: the remaining 122B datasets run through the vLLM single-GPU form (AppCorr-vllm,
# analysis/experiments/qwen_vllm_accuracy.py, --concurrency 4 -> "_c4" file suffix). Same vision
# tower (bitwise, vision_only.py), same processor family and greedy rule; the engine's own kernel
# band vs HF is 3-4% of predictions on near-ties (gate PASS, q122b_gate_rwqa192_cmp.log). Full
# split only (the driver resumes into the same jsonl), so the loader's `expected` gate applies.
QWEN122B_VLLM_DIR = "/NHNHOME/share/cjpark/AppCorr-vllm/analysis/results/qwen_vllm_accuracy"
QWEN122B_VLLM_PYR_DIR = QWEN122B_VLLM_DIR + "_pyr"
QWEN122B_VLLM_SUFFIX = "_c4"


def qwen35_pyr_lit(dataset: str, metric: str, slug: str = "",
                   dir_: str = None, expected: Dict[str, int] = None,
                   suffix: str = "") -> Dict[str, float]:
    """LITERALS-shaped dict computed from the pyr campaign's jsonls; complete arms only.
    `suffix` is the driver's file-name tail after the arm tag (vLLM runs: "_c4")."""
    exp = (expected or QWEN35_PYR_EXPECTED).get(dataset)
    out: Dict[str, float] = {}
    if exp is None:
        return out
    QWEN35_PYR_DIR_ = dir_ or QWEN35_PYR_DIR
    for key, tag in (("floor", "floor"), ("ceiling", "ceiling"),
                     ("k0.25", "streaming_g4_k0.25"), ("k0.50", "streaming_g4_k0.50"),
                     ("stream", "streaming_g4")):
        p = os.path.join(QWEN35_PYR_DIR_, f"{dataset}{slug}_{tag}{suffix}.jsonl")
        if not os.path.exists(p) and suffix:
            # the served campaigns added the "_c4" tail later than their bounds / k=1 arms
            # (e.g. 122B RealWorldQA: floor / ceiling / streaming_g4 without it, k<1 with it)
            p = os.path.join(QWEN35_PYR_DIR_, f"{dataset}{slug}_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        if len({r["i"] for r in rows}) < exp:
            continue
        sc = [r for r in rows if "skip" not in r]
        if not sc:
            continue
        out[key] = 100.0 * sum(float(r[metric]) for r in sc) / len(sc)
    return out

# (model label, [(dataset label, accuracy key, flops key)]). `accuracy key` is (dir_prefix, dataset)
# or None when no accuracy arm exists; `flops key` selects the FLOPs source.
SPEC = [
    ("Gemma 3 (4.3B)$^\\ddagger$", [
        ("ChartQA (Relaxed Acc.)", ("gemma3", "chartqa"),     ("inproc", "gemma3", "chartqa")),
        ("InfoVQA (ANLS)",         ("gemma3", "infovqa"),     ("inproc", "gemma3", "infovqa")),
        ("TextVQA (VQA Acc.)",     ("gemma3", "textvqa"),     ("inproc", "gemma3", "textvqa")),
        ("POPE (Acc.)",            ("gemma3", "pope"),        ("inproc", "gemma3", "pope")),
        ("RealWorldQA (Acc.)",     ("gemma3", "realworldqa"), ("inproc", "gemma3", "realworldqa")),
        ("DocVQA (ANLS)",          ("gemma3", "docvqa"),      ("inproc", "gemma3", "docvqa")),
        ("GQA testdev (Exact Match)", ("gemma3", "gqa"),      ("inproc", "gemma3", "gqa")),
        ("MMMU val (Acc.)",        ("gemma3", "mmmu"),        ("inproc", "gemma3", "mmmu")),
        ("VSR zeroshot (Acc.)",    ("gemma3", "vsr"),         ("inproc", "gemma3", "vsr")),
    ]),
    # Qwen3.5's arm is STREAMING (vision approx/correct + chunked LLM prefill), not interleaved
    # correction: it progressively recomputes 100% of tokens and has no keep-rate knob -- the
    # operating point is the round count (g=4 measured). Its FLOPs sit under the k0.50 keys purely
    # so the existing column machinery renders them; the dagger footnote in the caption says so.
    # Accuracy cells stay empty until the dataset driver runs.
    ("LLaVA-OV2 (8.5B)$^\\ddagger$", [
        ("ChartQA (Relaxed Acc.)", ("ov2", "chartqa"),     ("inproc", "ov2", "chartqa")),
        ("InfoVQA (ANLS)",         ("ov2", "infovqa"),     ("inproc", "ov2", "infovqa")),
        ("TextVQA (VQA Acc.)",     ("ov2", "textvqa"),     ("inproc", "ov2", "textvqa")),
        ("DocVQA (ANLS)",          ("ov2", "docvqa"),      ("inproc", "ov2", "docvqa")),
        ("RealWorldQA (Acc.)",     ("ov2", "realworldqa"), ("inproc", "ov2", "realworldqa")),
        ("POPE (Acc.)",            ("ov2", "pope"),        ("inproc", "ov2", "pope")),
        ("GQA testdev (Exact Match)", ("ov2", "gqa"),      ("inproc", "ov2", "gqa")),
        ("MMMU val (Acc.)",        ("ov2", "mmmu"),        ("inproc", "ov2", "mmmu")),
        ("RefCOCO val (Acc.@0.5)", ("ov2", "refcoco"),     ("inproc", "ov2", "refcoco")),
        ("VSR zeroshot (Acc.)",    ("ov2", "vsr"),         ("inproc", "ov2", "vsr")),
        ("V*Bench (Acc.)",         None, ("inproc", "ov2", "vstar")),
    ]),
    ("Qwen2.5-VL (33.5B)$^\\S$", [
        ("RefCOCO val (Acc.@0.5)",    None, ("inproc", "qwen25vl_32b", "refcoco")),
        ("RefCOCO val (mIoU)",        None, ("inproc", "qwen25vl_32b", "refcoco")),
        ("GQA testdev (Exact Match)", None, ("inproc", "qwen25vl_32b", "gqa")),
        ("RealWorldQA (Acc.)",        None, ("inproc", "qwen25vl_32b", "realworldqa")),
        ("MMVP (Acc.)",               None, ("inproc", "qwen25vl_32b", "mmvp")),
        ("CV-Bench (Acc.)",           None, ("inproc", "qwen25vl_32b", "cvbench")),
        # VisDrone: neither accuracy nor FLOPs exist for this model. The spec hands the encoder
        # the NATIVE frame (270x480 etc.), which ProgressiveLPyramidPolicy rejects (not a multiple
        # of the 28px merge patch) -- the fix is a resize policy that would also define the
        # accuracy arms, so it is not a fill (2026-09-07).
        ("VisDrone Count (Exact Acc.)", None, None),
        ("VisDrone Det (Acc.@0.5)",     None, None),
        ("V*Bench (Acc.)",              None, ("inproc", "qwen25vl_32b", "vstar")),
    ]),
    # Qwen3.5-4B (dense; the same 13 rows as the 35B block, same drivers with slug
    # "_qwen3.5-4b"; added 2026-09-11, cells fill as the arms land)
    ("Qwen3.5 (4B)$^\\dagger$", [
        ("ChartQA (Relaxed Acc.)", None, ("inproc", "qwen35_4b", "chartqa")),
        ("RealWorldQA (Acc.)",     None, ("inproc", "qwen35_4b", "realworldqa")),
        ("VSR zeroshot (Acc.)",    None, ("inproc", "qwen35_4b", "vsr")),
        ("MMVP (Acc.)",            None, ("inproc", "qwen35_4b", "mmvp")),
        ("CV-Bench (Acc.)",        None, ("inproc", "qwen35_4b", "cvbench")),
        # Resolution-sensitive track (2026-08-31, B200 box): accuracy is measured under the
        # pyr filter (Option B) by qwen35_accuracy.py into qwen35_accuracy_pyr/ -- jsonl, not
        # the {arm}.json layout, so cells stay "--" until a loader is wired. FLOPs n=12.
        ("RefCOCO val (Acc.@0.5)", None, ("inproc", "qwen35_4b", "refcoco")),
        ("RefCOCO val (mIoU)",     None, ("inproc", "qwen35_4b", "refcoco")),
        ("TextVQA (VQA Acc.)",     None, ("inproc", "qwen35_4b", "textvqa")),
        ("VisDrone Count (Exact Acc.)", None, ("inproc", "qwen35_4b", "visdrone_count")),
        ("VisDrone Count (Soft)",       None, ("inproc", "qwen35_4b", "visdrone_count")),
        ("VisDrone Det (Acc.@0.5)",     None, ("inproc", "qwen35_4b", "visdrone_det")),
        ("VisDrone Det (mIoU)",         None, ("inproc", "qwen35_4b", "visdrone_det")),
        ("V*Bench (Acc.)",              None, ("inproc", "qwen35_4b", "vstar")),
    ]),
    ("Qwen3.5-MoE (35B-A3B)$^\\dagger$", [
        ("ChartQA (Relaxed Acc.)", None, ("inproc", "qwen35_moe", "chartqa")),
        ("RealWorldQA (Acc.)",     None, ("inproc", "qwen35_moe", "realworldqa")),
        ("VSR zeroshot (Acc.)",    None, ("inproc", "qwen35_moe", "vsr")),
        ("MMVP (Acc.)",            None, ("inproc", "qwen35_moe", "mmvp")),
        ("CV-Bench (Acc.)",        None, ("inproc", "qwen35_moe", "cvbench")),
        # Resolution-sensitive track (2026-08-31, B200 box): accuracy is measured under the
        # pyr filter (Option B) by qwen35_accuracy.py into qwen35_accuracy_pyr/ -- jsonl, not
        # the {arm}.json layout, so cells stay "--" until a loader is wired. FLOPs n=12.
        ("RefCOCO val (Acc.@0.5)", None, ("inproc", "qwen35_moe", "refcoco")),
        ("RefCOCO val (mIoU)",     None, ("inproc", "qwen35_moe", "refcoco")),
        ("TextVQA (VQA Acc.)",     None, ("inproc", "qwen35_moe", "textvqa")),
        ("VisDrone Count (Exact Acc.)", None, ("inproc", "qwen35_moe", "visdrone_count")),
        ("VisDrone Count (Soft)",       None, ("inproc", "qwen35_moe", "visdrone_count")),
        ("VisDrone Det (Acc.@0.5)",     None, ("inproc", "qwen35_moe", "visdrone_det")),
        ("VisDrone Det (mIoU)",         None, ("inproc", "qwen35_moe", "visdrone_det")),
        ("V*Bench (Acc.)",              None, ("inproc", "qwen35_moe", "vstar")),
    ]),
    # Gemma 4 31B: Ours = INTERLEAVED g=4 since 2026-08-31 (port-plan step 4 landed; walk
    # gate bitwise, identity gate in ceiling's flicker set). Accuracy cells prefer
    # interleaved_g4_k*.json and fall back to corrected_k*.json while reruns land; FLOPs in
    # inprocess_flops.json are the interleaved accounting (crit 17-35%, total 154-182% --
    # the one-shot structure's 90%+ crit / 103-106% total is archived in
    # analysis/results/flops/gemma4_flops*.json).
    ("Gemma 4 (31B)", [
        ("MMVP (Acc.)",            ("gemma4", "mmvp"),    ("inproc", "gemma4", "mmvp")),
        ("CV-Bench (Acc.)",        ("gemma4", "cvbench"), ("inproc", "gemma4", "cvbench")),
        # Resolution-sensitive track (GH200 campaign, post-Mistral). NOTE the port has bounds +
        # one-shot corrected only (interleaved/streaming are port-plan steps 4-6, not yet built),
        # so the Streaming column stays "--" for now by construction.
        ("RefCOCO val (Acc.@0.5)", ("gemma4", "refcoco"), ("inproc", "gemma4", "refcoco")),
        ("RefCOCO val (mIoU)",     ("gemma4", "refcoco"), ("inproc", "gemma4", "refcoco")),
        ("TextVQA (VQA Acc.)",     ("gemma4", "textvqa"), ("inproc", "gemma4", "textvqa")),
        ("VisDrone Count (Exact Acc.)", ("gemma4", "visdrone_count"), ("inproc", "gemma4", "visdrone_count")),
        ("VisDrone Count (Soft)",       ("gemma4", "visdrone_count"), ("inproc", "gemma4", "visdrone_count")),
        ("VisDrone Det (Acc.@0.5)",     ("gemma4", "visdrone_det"),   ("inproc", "gemma4", "visdrone_det")),
        ("VisDrone Det (mIoU)",         ("gemma4", "visdrone_det"),   ("inproc", "gemma4", "visdrone_det")),
    ]),
    # New 30B-class models (2026-08-28 sweep): bounds via the generic oracle; ours arms pending
    # their axis ports. WildVision is judge-only (prediction dumps) and has no accuracy row.
    # Mistral Ours columns (2026-09-07 fix): compute is the STREAMING arm (per-band correction
    # under the keep quota + arrival-ordered chunked prefill -- the mistral3_oracle name for the
    # progressive schedule every other row uses; keys remapped in inprocess_flops.json, the one-shot
    # figures stay in the per-arm files). Accuracy prefers streaming_g4_k*.json (RefCOCO/TextVQA at
    # 50%, VisDrone at 50%) and falls back to the one-shot corrected_k*.json elsewhere -- ddagger footnote.
    ("Mistral Small 3.1 (24B)$^\\ddagger$", [
        ("MMVP (Acc.)",            ("mistral24b", "mmvp"),    ("inproc", "mistral24b", "mmvp")),
        ("CV-Bench (Acc.)",        ("mistral24b", "cvbench"), ("inproc", "mistral24b", "cvbench")),
        # Resolution-sensitive track (2026-08-28 priority shift): grounding + OCR + drone-scale
        # tiny objects. File-based: cells fill as the GH200 campaign's arms land.
        ("RefCOCO val (Acc.@0.5)", ("mistral24b", "refcoco"), ("inproc", "mistral24b", "refcoco")),
        ("RefCOCO val (mIoU)",     ("mistral24b", "refcoco"), ("inproc", "mistral24b", "refcoco")),
        ("TextVQA (VQA Acc.)",     ("mistral24b", "textvqa"), ("inproc", "mistral24b", "textvqa")),
        ("VisDrone Count (Exact Acc.)", ("mistral24b", "visdrone_count"), ("inproc", "mistral24b", "visdrone_count")),
        ("VisDrone Count (Soft)",       ("mistral24b", "visdrone_count"), ("inproc", "mistral24b", "visdrone_count")),
        ("VisDrone Det (Acc.@0.5)",     ("mistral24b", "visdrone_det"),   ("inproc", "mistral24b", "visdrone_det")),
        ("VisDrone Det (mIoU)",         ("mistral24b", "visdrone_det"),   ("inproc", "mistral24b", "visdrone_det")),
        ("V*Bench (Acc.)",              None, ("inproc", "mistral24b", "vstar")),
    ]),
    ("Muse Glimmer (29.6B)", [
        ("MMVP (Acc.)",            ("museglimmer30b", "mmvp"),    ("inproc", "museglimmer30b", "mmvp")),
        ("CV-Bench (Acc.)",        ("museglimmer30b", "cvbench"), ("inproc", "museglimmer30b", "cvbench")),
        ("V*Bench (Acc.)",         None, ("inproc", "museglimmer30b", "vstar")),
        # 2026-09-01 MG campaign (user scope: RefCOCO/VisDrone/TextVQA, arms floor/ceiling/
        # streaming k1.0). visdrone_det is a capability-limit row (~1-2% every arm), kept per
        # the Mistral precedent. FLOPs: refcoco/textvqa from museglimmer_arms_flops.json.
        ("RefCOCO val (Acc.@0.5)", None, ("inproc", "museglimmer30b", "refcoco")),
        ("RefCOCO val (mIoU)",     None, ("inproc", "museglimmer30b", "refcoco")),
        ("TextVQA (VQA Acc.)",     None, ("inproc", "museglimmer30b", "textvqa")),
        ("VisDrone Count (Exact Acc.)", None, ("inproc", "museglimmer30b", "visdrone_count")),
        ("VisDrone Count (Soft)",       None, ("inproc", "museglimmer30b", "visdrone_count")),
        ("VisDrone Det (Acc.@0.5)",     None, ("inproc", "museglimmer30b", "visdrone_det")),
        ("VisDrone Det (mIoU)",         None, ("inproc", "museglimmer30b", "visdrone_det")),
    ]),
    # 122B-FP8: DeepGEMM mis-generates on this B200 (sm_100; per-row outputs bit-perfect, end
    # tokens drift -- transformers 5.13 documents it); accuracy IS measurable under the Triton
    # finegrained-fp8 fallback (TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1). Cells below are the
    # 2026-09-01 REDUCED-SCALE probe (n=240/arm, user-directed) via the QWEN122B_* loader; the
    # pilcrow footnote states both the fallback and the n. COMPUTE figures are shape-determined
    # and kernel-independent (n=12, attention term included, routed experts counted since the
    # 2026-09-10 FP8Experts hook fix -- see _note; the pre-fix entry is qwen35_122b_prefix).
    ("Qwen3.5-MoE (122B-A10B FP8)$^\\dagger$\\textsuperscript{\\P}", [
        ("ChartQA (Relaxed Acc.)", None, ("inproc", "qwen35_122b", "chartqa")),
        ("RealWorldQA (Acc.)",     None, ("inproc", "qwen35_122b", "realworldqa")),
        ("VSR zeroshot (Acc.)",    None, ("inproc", "qwen35_122b", "vsr")),
        ("MMVP (Acc.)",            None, ("inproc", "qwen35_122b", "mmvp")),
        ("CV-Bench (Acc.)",        None, ("inproc", "qwen35_122b", "cvbench")),
        # Resolution-sensitive track (2026-08-31, B200 box): accuracy is measured under the
        # pyr filter (Option B) by qwen35_accuracy.py into qwen35_accuracy_pyr/ -- jsonl, not
        # the {arm}.json layout, so cells stay "--" until a loader is wired. FLOPs n=12.
        ("RefCOCO val (Acc.@0.5)", None, ("inproc", "qwen35_122b", "refcoco")),
        ("RefCOCO val (mIoU)",     None, ("inproc", "qwen35_122b", "refcoco")),
        ("TextVQA (VQA Acc.)",     None, ("inproc", "qwen35_122b", "textvqa")),
        ("VisDrone Count (Exact Acc.)", None, ("inproc", "qwen35_122b", "visdrone_count")),
        ("VisDrone Count (Soft)",       None, ("inproc", "qwen35_122b", "visdrone_count")),
        ("VisDrone Det (Acc.@0.5)",     None, ("inproc", "qwen35_122b", "visdrone_det")),
        ("VisDrone Det (mIoU)",         None, ("inproc", "qwen35_122b", "visdrone_det")),
        ("V*Bench (Acc.)",              None, ("inproc", "qwen35_122b", "vstar")),
    ]),
    # OpenVLA FLOPs (2026-09-07): AppCorr-openvla/analysis/experiments/flops_report_openvla.py,
    # in-process, 4 sequential groups with cumulative vision correction. Backbone = both towers +
    # projector + 32 Llama layers; the 7-token action decode is excluded like every other model's
    # decode. k1.00/total_k1.00 are the CHUNKED causal prefill (no LLM approx pass, each position
    # once, text once; total ~127%). The accuracy campaign ran the interleaved schedule (frontiers
    # 32x4), gated bit-identical to chunked on final state / action bins (openvla_chunked_gate.py),
    # whose redundant cost (~244%) is kept as interleaved_total_k1.00. Only the streaming (k=1.00)
    # arm exists for the VLA, so the one-shot columns stay "--" by construction.
    # The table keeps the CUMULATIVE correction (every arrived patch re-corrected each round). A
    # cheaper NEW-ONLY variant (only the round's newly arrived patches corrected; 2026-09-07..08,
    # 500 ep/suite, one-camera env, libero_*_chunked_newonly_t50.jsonl) costs total ~113% /
    # critical ~32% but is not lossless: Spatial 77.2 (-4.6pp vs cumulative, CI excl. 0), Object
    # 85.6, Goal 74.6, Long 41.0 (-10.8pp vs ceiling) -- reported in the asterisk footnote only
    # (user decision 2026-09-08), flops keys newonly_k1.00 / newonly_total_k1.00 stay unused here.
    ("OpenVLA (7B)$^\\ast$", [
        ("LIBERO-Spatial (Success Rate)", None, ("inproc", "openvla", "libero_spatial")),
        ("LIBERO-Object (Success Rate)",  None, ("inproc", "openvla", "libero_object")),
        ("LIBERO-Goal (Success Rate)",    None, ("inproc", "openvla", "libero_goal")),
        ("LIBERO-Long (Success Rate)",    None, ("inproc", "openvla", "libero_10")),
    ]),
    ("DINOv3 (7B)", [
        (r"ImageNet-1k (Top-1 $\uparrow$)", None, ("offload", "dinov3_imagenet")),
        (r"COCO Detector (mAP $\uparrow$)", None, ("offload", "dinov3_coco")),
        (r"ADE20K m2f (mIoU $\uparrow$)",   None, ("offload", "dinov3_ade20k")),
        (r"NYUv2 (AbsRel $\downarrow$)",    None, ("offload", "dinov3_nyu")),
        (r"Co3Dv2 (Rot. deg $\downarrow$)", None, None),
    ]),
    ("SAM 3 (0.85B)", [
        ("COCO Tracker (Mask AP)",  None, ("inproc", "sam3", "coco")),
        ("COCO Detector (Mask AP)", None, ("inproc", "sam3", "coco")),
        ("LVIS Detector (Mask AP)", None, ("inproc", "sam3", "coco")),
        ("SA-Co crowded (cgF1)",    None, ("inproc", "sam3", "coco")),
        ("SA-Co sa1b (cgF1)",       None, ("inproc", "sam3", "coco")),
        ("SA-Co attributes (cgF1)", None, ("inproc", "sam3", "coco")),
        ("SA-Co metaclip (cgF1)",    None, ("inproc", "sam3", "coco")),
        ("SA-Co fg-sports (cgF1)",   None, ("inproc", "sam3", "coco")),
        ("SA-Co fg-food (cgF1)",     None, ("inproc", "sam3", "coco")),
        ("SA-Co wiki-common (cgF1)", None, ("inproc", "sam3", "coco")),
    ]),
    ("OpenCLIP (2.5B)", [
        ("ImageNet-1k (Top-1)",        None, ("offload", "openclip_imagenet")),
        ("ImageNet-1k (Top-5)",        None, ("offload", "openclip_imagenet")),
        # Same vision tower and same 224px canvas as the ImageNet rows, so the ceiling comes out
        # bit-identical (967.5 GF/image) -- measured rather than aliased, which is what makes that
        # agreement a cross-check instead of an assumption.
        ("COCO Ret. val2017 (i2t R@1)", None, ("offload", "openclip_cocoret")),
        ("COCO Ret. val2017 (t2i R@1)", None, ("offload", "openclip_cocoret")),
        # One-shot (g=1) at the same keep: the diagnosis decomposition's headline. CLIP is the one
        # model where interleaving costs real accuracy (staleness 5.76pp > selection 3.06pp at
        # keep=0.50); this row shows the trade the g=4 row hides. Measured on the full 5000-image
        # split, 2026-08-26 (docs/memo/openclip_staleness_decomposition.md).
        # FLOPs: the same worker accounting with num_groups=1 (openclip_cocoret_g1_k0.50.json,
        # 2026-09-07); the ceiling file is shared with the g=4 rows because a one-shot ceiling is
        # the same forward on the same canvas -- an identity, not a measurement to repeat.
        ("COCO Ret. one-shot g=1 (i2t R@1)", None, ("offload", "openclip_cocoret", "g1")),
        ("COCO Ret. one-shot g=1 (t2i R@1)", None, ("offload", "openclip_cocoret", "g1")),
    ]),
    ("VGGT-Omega (7B)", [
        (r"Co3Dv2 (Depth AbsRel $\downarrow$)", None, ("offload", "vggt_co3d")),
        (r"Co3Dv2 (Rot. deg $\downarrow$)",     None, ("offload", "vggt_co3d")),
        (r"Co3Dv2 ($\delta < 1.10$ $\uparrow$)", None, ("offload", "vggt_co3d")),
        (r"Co3Dv2 (3D Point Err. $\downarrow$)", None, ("offload", "vggt_co3d")),
        (r"Co3Dv2 (3D Inlier $<10\%$)",         None, ("offload", "vggt_co3d")),
    ]),
]

# Presentation order (user, 2026-08-31): VFMs first, then the VLMs, then the VLA. Sorting here
# instead of moving the literal blocks keeps each block's comments next to its rows.
_ROW_ORDER = ["DINOv3", "SAM 3", "OpenCLIP", "VGGT-Omega", "Gemma 3", "Gemma 4", "LLaVA-OV2",
              "Mistral Small", "Muse Glimmer", "Qwen2.5-VL", "Qwen3.5 (4B", "Qwen3.5-MoE (35B",
              "Qwen3.5-MoE (122B", "OpenVLA"]


def _row_order_key(entry):
    name = entry[0]
    for i, prefix in enumerate(_ROW_ORDER):
        if name.startswith(prefix):
            return i
    return len(_ROW_ORDER)  # unknown models sink to the bottom rather than crash


SPEC.sort(key=_row_order_key)

# Values that exist only in prose (other branches, published memos) and have no JSON to read.
# Kept separate from anything measured here so the two are never confused.
# SUSPECT, 2026-08-25: every Qwen2.5-VL value below came from `full_inference` in
# `qwen25vl_executor.py`, which called the stock model without `mm_token_type_ids`. transformers
# then takes `can_compute_mrope = False`, `compute_3d_position_ids` returns None, and the text model
# falls back to plain 1D positions replicated across all three M-RoPE axes -- every image token
# loses its (t, h, w) grid position and is treated as text at its sequence offset. Traced and proven
# on the GH200 box: a correctly-called stock forward matches our interleaved g=1 arm bit-exactly
# (0/8,714,240 elements differing), while the degraded call does not. So the ARM was always right
# and the BASELINE was wrong. These numbers, and every gap or crossing point measured against them,
# have been re-established. RefCOCO and GQA below now carry the corrected values; the RealWorldQA
# ceilings still do not, and neither does anything derived from the old bounds (the keep-rate sweeps
# and the "-1pp crossing at ~58%" conclusion in QWEN25VL_APPCORR_LOG.md), which all need re-deriving
# rather than re-centering -- RefCOCO's gap NARROWED from 10.99pp to 8.51pp because its floor moved
# further than its ceiling (+4.92 against +2.44), so a recovery fraction computed against the old
# bounds is wrong by more than a shift. GQA's bounds barely moved at all (-0.04 / +0.08), which is
# the same insensitivity its churn analysis showed.
#
# Re-measurement in progress on the GH200 box. RefCOCO baseline has landed: 85.75 -> **88.19**
# (mean IoU 0.7620 -> 0.8024), full 8811 split. It moved TOWARD the published Qwen2.5-VL figures,
# which is the independent corroboration that the mechanism is what the trace says it is -- a fix
# that left the number flat, or moved it down, would have meant the story was wrong even though the
# tensors matched. GQA is running. The floor arms came from the same broken function and are being
# re-measured too, so nothing here is updated until BOTH bounds are back: quoting a new ceiling
# against an old floor would invent a gap neither measurement supports.
LITERALS = {
    # V*Bench full split (191), 2026-09-01 campaign, pyr L2, from the oracle Final Summary lines
    # (analysis/results/logs/vstar_mistral_*.log / vstar_ov2_*.log -- --out-json was not passed).
    ("Mistral Small 3.1 (24B)", "V*Bench (Acc.)"): {"floor": 50.26, "ceiling": 52.36,
                                                    "k0.25": 53.40, "k0.50": 54.97,
                                                    "stream": 51.83},
    ("LLaVA-OV2 (8.5B)", "V*Bench (Acc.)"): {"floor": 74.87, "ceiling": 85.86,
                                             "k0.25": 78.01, "k0.50": 83.25,
                                             "stream": 84.29},
    # V*Bench full split via qwen25vl_bench_eval retry (logs vstar_qwen25_*.log, '=== Summary').
    ("Qwen2.5-VL (33.5B)", "V*Bench (Acc.)"): {"floor": 60.73, "ceiling": 79.06,
                                               "k0.25": 70.68, "k0.50": 74.87,
                                               "stream": 78.01},
    # Qwen3.5 accuracy, full RealWorldQA split (765), 2026-08-27, thinking disabled, shared greedy
    # decode across all three arms. Single streaming arm (g=4) sits under the k0.50 columns per the
    # dagger footnote's nominal-placement rule.
    ("Qwen3.5-MoE (35B-A3B)", "RealWorldQA (Acc.)"): {"floor": 74.51, "ceiling": 77.39,
                                                      "k0.25": 77.52, "k0.50": 77.25,
                                                      "stream": 77.25},
    ("Qwen3.5-MoE (35B-A3B)", "ChartQA (Relaxed Acc.)"): {"floor": 60.76, "ceiling": 88.56,
                                                          "k0.25": 83.68, "k0.50": 86.80,
                                                          "stream": 88.32},
    ("Qwen3.5-MoE (35B-A3B)", "VSR zeroshot (Acc.)"): {"floor": 88.46, "ceiling": 89.77,
                                                       "k0.25": 88.63, "k0.50": 88.95},
    # MMVP full 300 (2026-08-28 real-photo sweep): floor / streaming g=4 / ceiling.
    # NOTE: floor/streaming measured with the bicubic-era filter; BOX re-measurement
    # (qwen35_accuracy_box/) supersedes these when it lands.
    ("Qwen3.5-MoE (35B-A3B)", "MMVP (Acc.)"): {"floor": 79.00, "ceiling": 82.00,
                                               "k0.25": 80.67, "k0.50": 80.00,
                                               "stream": 81.67},
    # GH200 campaign (2026-08-28): cvbench ceiling full 2638, zero skips, FINAL.
    # Floor arrives after their chunked-attention fix rerun; do not backfill early values.
    # floor is the completed full-2638 rerun after the chunked-attention fix (the 111
    # recovered 2K-res images scored 62%, harder than average). k0.50/streaming pending.
    # MMVP full 300, five arms, zero skips (GH200). Largest Qwen2.5 gap measured (9.34pp --
    # CLIP-blind discrimination is exactly what a level-2 pyramid destroys). n=300: one sample
    # = 0.33pp, recovery CI ~ +-5pp -- read recovery coarsely, preservation ordering is solid.
    ("Qwen2.5-VL (33.5B)", "MMVP (Acc.)"): {"floor": 65.33, "ceiling": 74.67,
                                            "k0.25": 68.33, "k0.50": 69.00,
                                            "stream": 71.67},
    ("Qwen2.5-VL (33.5B)", "CV-Bench (Acc.)"): {"ceiling": 79.87, "floor": 72.52,
                                                "k0.25": 75.09, "k0.50": 77.52,
                                                "stream": 78.96},
    # full 2638, BOX floor (post-convention-audit), shared greedy decode.
    ("Qwen3.5-MoE (35B-A3B)", "CV-Bench (Acc.)"): {"floor": 84.08, "ceiling": 85.06,
                                                   "k0.25": 84.50, "k0.50": 84.76,
                                                   "stream": 84.87},
    # One-shot rows share the interleaved rows' bounds (same floor/ceiling arms).
    ("OpenCLIP (2.5B)", "COCO Ret. one-shot g=1 (i2t R@1)"): {"floor": 50.14, "ceiling": 67.92},
    ("OpenCLIP (2.5B)", "COCO Ret. one-shot g=1 (t2i R@1)"): {"floor": 40.37, "ceiling": 50.64},
    # Re-measured 2026-08-26 on the M-RoPE-fixed code, full splits, both bounds through the same
    # driver. RefCOCO N=8811, GQA N=12578. These REPLACE the pre-fix values (which were
    # 85.75/74.76, 76.20/65.02, 60.84/55.16) -- see the block comment above.
    # RefCOCO ours: GH200 full-split (8811 images) 2026-08-27, energy x attention, bs=1 vs
    # bs=16 bounds (measured Acc-identical, 0.001 mIoU).
    # k0.50 is the text-split-schedule run at FULL n=8811 coverage (2026-08-28): the original
    # 207-image OOM exclusion turned out to be a driver defect (missing no_grad pinning ~27GB of
    # autograd graph per image -- fixed same day), and the skipped images were completed via the
    # jsonl resume under the identical schedule. Bounds are therefore the plain full-split bounds,
    # no kept-set restriction needed, and the section-mark footnote no longer applies to k0.50.
    # Schedule A/B at full scale: -0.02pp (flips 21:23) vs every-round.
    # k0.25 kept from the every-round run at n=8803 (8 OOM skips from the same driver defect,
    # predating text-split -- resuming them would mix schedules in one file, and the schedule
    # A/B says the number would not move; the footnote still covers this arm).
    # Streaming (k=1.0) full n=8811, 2026-08-28: 89.55/81.50 -- ABOVE ceiling by +1.36pp.
    # Verified real, not mismeasurement: paired flips 278:158 (net +120, ~4sigma) vs ceiling;
    # control (k0.50, same driver/decode structure) nets -78 as expected. Mechanism isolated by
    # elimination: interleaved k=1.0 subset ALSO beats ceiling (+1.9pp) -> not chunked-prefill-
    # specific; the g=1 identity gate is bitwise-exact vs stock -> not the fork decode; what
    # remains is the multi-round vision correction's partial staleness acting as a beneficial
    # perturbation for grounding. Cross-model: Mistral-24B MMVP streaming +1.33pp (no chunked
    # LLM at all) and OV2 GQA/VSR at 100.2% point the same way.
    ("Qwen2.5-VL (33.5B)", "RefCOCO val (Acc.@0.5)"):    {"floor": 79.68, "ceiling": 88.19,
                                                          "k0.25": 86.10, "k0.50": 87.30,
                                                          "stream": 89.55},
    ("Qwen2.5-VL (33.5B)", "RefCOCO val (mIoU)"):        {"floor": 70.36, "ceiling": 80.24,
                                                          "k0.25": 78.22, "k0.50": 79.35,
                                                          "stream": 81.50},
    ("Qwen2.5-VL (33.5B)", "GQA testdev (Exact Match)"): {"floor": 55.24, "ceiling": 60.80},
    # 72B dropped 2026-08-26: not worth the run. It also does not fit -- the GH200 box has ~66 GB
    # free against a ~130 GB pull, so the row could only ever have carried a prose ceiling.
    ("Qwen2.5-VL (33.5B)", "RealWorldQA (Acc.)"):        {"ceiling": 68.89},
    ("SAM 3 (0.85B)", "COCO Tracker (Mask AP)"):  {"floor": 53.74, "ceiling": 60.10},
    ("SAM 3 (0.85B)", "COCO Detector (Mask AP)"): {"floor": 43.32, "ceiling": 50.92},
    ("SAM 3 (0.85B)", "LVIS Detector (Mask AP)"): {"floor": 41.21, "ceiling": 56.38},
    ("SAM 3 (0.85B)", "SA-Co crowded (cgF1)"):    {"floor": 53.15, "ceiling": 58.95},
    ("SAM 3 (0.85B)", "SA-Co sa1b (cgF1)"):       {"floor": 52.78, "ceiling": 53.94},
    ("SAM 3 (0.85B)", "SA-Co attributes (cgF1)"): {"floor": 53.96, "ceiling": 54.21},
    ("DINOv3 (7B)", r"ImageNet-1k (Top-1 $\uparrow$)"): {"floor": 84.50, "ceiling": 88.11},
    ("DINOv3 (7B)", r"COCO Detector (mAP $\uparrow$)"): {"floor": 55.83, "ceiling": 63.14},
    ("DINOv3 (7B)", r"ADE20K m2f (mIoU $\uparrow$)"):   {"floor": 56.01, "ceiling": 62.24},
    # measured here 2026-08-25, full 654-sample split, same driver as the ours arms
    ("DINOv3 (7B)", r"NYUv2 (AbsRel $\downarrow$)"):    {"floor": 0.05302, "ceiling": 0.05013,
                                                        "fmt": "{:.4f}"},
    ("DINOv3 (7B)", r"Co3Dv2 (Rot. deg $\downarrow$)"): {"floor": 5.440, "ceiling": 2.885,
                                                        "fmt": "{:.3f}"},
    # measured here 2026-08-25, full 310 sequences -- these confirmed the literals to 3 decimals,
    # which is what made VGGT's ours-below-floor anomaly real rather than a bad reference
    ("VGGT-Omega (7B)", r"Co3Dv2 (Depth AbsRel $\downarrow$)"): {"floor": 0.04773, "ceiling": 0.04255,
                                                                "fmt": "{:.4f}"},
    ("VGGT-Omega (7B)", r"Co3Dv2 (Rot. deg $\downarrow$)"):     {"floor": 1.552, "ceiling": 1.332,
                                                                "fmt": "{:.3f}"},
    ("VGGT-Omega (7B)", r"Co3Dv2 ($\delta < 1.10$ $\uparrow$)"): {"floor": 92.83, "ceiling": 92.97},
    ("VGGT-Omega (7B)", r"Co3Dv2 (3D Point Err. $\downarrow$)"): {"floor": 0.1464, "ceiling": 0.1572,
                                                                 "fmt": "{:.4f}"},
    ("VGGT-Omega (7B)", r"Co3Dv2 (3D Inlier $<10\%$)"):          {"floor": 66.25, "ceiling": 62.61},
    ("OpenCLIP (2.5B)", "ImageNet-1k (Top-1)"): {"floor": 65.92, "ceiling": 77.14},
    ("OpenCLIP (2.5B)", "ImageNet-1k (Top-5)"): {"floor": 88.20, "ceiling": 94.88},
    # measured here 2026-08-26 through COCOCaptionsLoader, full 5000/25014 -- our own ceiling landed
    # within 0.06 of the literal, which is what validates the new loader against the prior protocol
    ("OpenCLIP (2.5B)", "COCO Ret. val2017 (i2t R@1)"): {"floor": 50.14, "ceiling": 67.92},
    ("OpenCLIP (2.5B)", "COCO Ret. val2017 (t2i R@1)"): {"floor": 40.37, "ceiling": 50.64},
    # LIBERO-Spatial: ceiling re-measured 2026-09-04 on the rebuilt env (numpy 1.26 fix; llvmpipe
    # EGL) with the offload driver's `full` schedule, 500 episodes = 10 tasks x 50 trials, primary
    # evidence AppCorr-openvla/analysis/results/openvla/libero_spatial_{full,approx,interleaved}_t50.jsonl
    # (July's same-harness 82.8/17.2/81.6 were lost with /tmp; paper 84.7 +- 0.9). floor = approx-only
    # schedule 93/500; stream = interleaved schedule, frontiers 32x4, sequential grouping, g=4,
    # 409/500 (2026-09-04) -- ties the 408/500 ceiling. (Gated bit-identical to the chunked causal
    # prefill whose compute the Comp cells report; see the FLOPs comment in SPEC.)
    ("OpenVLA (7B)", "LIBERO-Spatial (Success Rate)"): {"ceiling": 81.60, "floor": 18.60,
                                                        "stream": 81.80},
    # LIBERO-Object/Goal/Long: same campaign, 2026-09-05..07 (offload full/approx/interleaved
    # schedules, 500 episodes each, official max_steps 280/300/520); prior ceiling literals
    # 89/73/54. Object 430/99/424, Goal 375/89/374, Long 259/13/227 (ceiling/floor/stream).
    # Long stream ran on cuda:1 (--device) alongside approx on cuda:0; paired delta vs ceiling
    # -6.4pp, 95% CI [-12.0, -1.2], 85 wins / 117 losses over 500 (task,trial) pairs -- the
    # only suite where stream is below the ceiling beyond noise (520-step horizon).
    ("OpenVLA (7B)", "LIBERO-Object (Success Rate)"):  {"ceiling": 86.00, "floor": 19.80,
                                                        "stream": 84.80},
    ("OpenVLA (7B)", "LIBERO-Goal (Success Rate)"):    {"ceiling": 75.00, "floor": 17.80, "stream": 74.80},
    ("OpenVLA (7B)", "LIBERO-Long (Success Rate)"):    {"ceiling": 51.80, "floor": 2.60, "stream": 45.40},
}


# The VFM "ours" arms, produced by run_vfm_accuracy_campaign.sh / run_vfm_bounds.sh, which write a
# `Final Summary: {...}` line to a log rather than a summary JSON. Keyed by (model label, dataset
# label) -> (log tag prefix, metric key, scale).
#
# Scale is per-row and NOT guessable from the value: ImageNet reports `top1_acc` already in percent,
# COCO reports `mAP` as a fraction, and NYU/VGGT report raw error values that must not be scaled at
# all. Getting one wrong yields a number off by 100x that still looks like a plausible metric for
# some other task, so each is written out rather than inferred.
VFM_OURS = {
    ("DINOv3 (7B)", r"ImageNet-1k (Top-1 $\uparrow$)"): ("dinov3_imagenet", "top1_acc", 1.0),
    ("DINOv3 (7B)", r"COCO Detector (mAP $\uparrow$)"): ("dinov3_coco", "mAP", 100.0),
    ("DINOv3 (7B)", r"ADE20K m2f (mIoU $\uparrow$)"):   ("dinov3_ade20k", "mIoU", 1.0),
    ("DINOv3 (7B)", r"NYUv2 (AbsRel $\downarrow$)"):    ("dinov3_nyu", "abs_rel", 1.0),
    ("VGGT-Omega (7B)", r"Co3Dv2 (Depth AbsRel $\downarrow$)"): ("vggt_co3d", "abs_rel", 1.0),
    ("VGGT-Omega (7B)", r"Co3Dv2 (Rot. deg $\downarrow$)"):     ("vggt_co3d", "rot_deg", 1.0),
    ("VGGT-Omega (7B)", r"Co3Dv2 ($\delta < 1.10$ $\uparrow$)"): ("vggt_co3d", "delta_1.10", 100.0),
    # SAM 3's six rows share one vision encoder -- which is why their Crit. Comp. column repeats the
    # same FLOPs by construction -- but they are different TASKS, so accuracy is measured per task.
    # Metric keys differ by evaluator, but every SAM 3 summary -- mask_AP AND cgF1 -- is a
    # 0..1 FRACTION (verified against the measured logs: crowded k0.25 prints cgF1 0.5594).
    # An earlier version of this block asserted cgF1 was "already in percent" and gave it scale
    # 1.0, which rendered 0.56 in a column of 55-60s and read as a collapsed model. Writing the
    # scale per row stays deliberate for exactly that reason -- a wrong scale still looks like a
    # plausible metric, so each entry is checked against its own log line, not inferred.
    ("SAM 3 (0.85B)", "COCO Tracker (Mask AP)"):  ("sam3_coco", "mask_AP", 100.0),
    ("SAM 3 (0.85B)", "COCO Detector (Mask AP)"): ("sam3_cocodet", "mask_AP", 100.0),
    ("SAM 3 (0.85B)", "LVIS Detector (Mask AP)"): ("sam3_lvis", "mask_AP", 100.0),
    ("SAM 3 (0.85B)", "SA-Co crowded (cgF1)"):    ("sam3_saco_crowded", "cgF1", 100.0),
    ("SAM 3 (0.85B)", "SA-Co sa1b (cgF1)"):       ("sam3_saco_sa1b", "cgF1", 100.0),
    ("SAM 3 (0.85B)", "SA-Co attributes (cgF1)"): ("sam3_saco_attributes", "cgF1", 100.0),
    ("SAM 3 (0.85B)", "SA-Co metaclip (cgF1)"):    ("sam3_saco_metaclip", "cgF1", 100.0),
    ("SAM 3 (0.85B)", "SA-Co fg-sports (cgF1)"):   ("sam3_saco_fg_sports_equipment", "cgF1", 100.0),
    ("SAM 3 (0.85B)", "SA-Co fg-food (cgF1)"):     ("sam3_saco_fg_food", "cgF1", 100.0),
    ("SAM 3 (0.85B)", "SA-Co wiki-common (cgF1)"): ("sam3_saco_wiki_common", "cgF1", 100.0),
    ("OpenCLIP (2.5B)", "ImageNet-1k (Top-1)"): ("openclip_imagenet", "top1_acc", 1.0),
    ("OpenCLIP (2.5B)", "ImageNet-1k (Top-5)"): ("openclip_imagenet", "top5_acc", 1.0),
    ("OpenCLIP (2.5B)", "COCO Ret. val2017 (i2t R@1)"): ("cocoret", "i2t_R@1", 1.0),
    ("OpenCLIP (2.5B)", "COCO Ret. val2017 (t2i R@1)"): ("cocoret", "t2i_R@1", 1.0),
    ("OpenCLIP (2.5B)", "COCO Ret. one-shot g=1 (i2t R@1)"): ("cocoret_g1", "i2t_R@1", 1.0),
    ("OpenCLIP (2.5B)", "COCO Ret. one-shot g=1 (t2i R@1)"): ("cocoret_g1", "t2i_R@1", 1.0),
}

# (base model, dataset label) pairs where the MODEL fails the TASK outright (floor ~= ceiling at
# degenerate accuracy), so the cells say nothing about the streaming axis. Rendered with a
# $\diamond$ after the dataset label per the user's 2026-09-01 directive. MG RefCOCO was cut at
# ceiling 7.26% (6450/8811, decision: skip remaining arms); MG VisDrone Det ceiling is 2.23%.
CAPABILITY_LIMIT = {
    ("Muse Glimmer (29.6B)", "RefCOCO val (Acc.@0.5)"),
    ("Muse Glimmer (29.6B)", "RefCOCO val (mIoU)"),
    ("Muse Glimmer (29.6B)", "VisDrone Det (Acc.@0.5)"),
    ("Muse Glimmer (29.6B)", "VisDrone Det (mIoU)"),
}

VFM_DIR = os.path.join(RESULTS, "vfm_accuracy")

for _sub, _lbl in (("metaclip", "metaclip"), ("fg_sports_equipment", "fg-sports"),
                   ("fg_food", "fg-food"), ("wiki_common", "wiki-common")):
    _fc = {}
    for _arm in ("floor", "ceiling"):
        _p = os.path.join(VFM_DIR, f"sam3_saco_{_sub}_{_arm}.json")
        if os.path.exists(_p):
            try:
                _fc[_arm] = 100.0 * json.load(open(_p))["cgF1"]
            except (KeyError, ValueError):
                pass
    if _fc:
        LITERALS.setdefault(("SAM 3 (0.85B)", f"SA-Co {_lbl} (cgF1)"), {}).update(_fc)


def vfm_accuracy(tag: str, key: str, scale: float) -> Optional[float]:
    """Read one metric out of a `Final Summary: {...}` line.

    A run that died still leaves a log, and several have exited rc=0 while producing nothing
    (a missing dataset loader, an invalid device ordinal). Absence of the summary line is the
    only reliable "this arm did not happen" signal, so it is what this returns None on.

    Both quoting styles appear: the offload drivers print a Python dict (single quotes), the SAM 3
    oracle prints JSON (double quotes).
    """
    # The SAM 3 oracle rewrite (048835e) writes proper result JSONs instead of Final-Summary
    # logs; try {tag}.json first (committed, box-independent), then the legacy {tag}.log
    # (B200-local campaign logs -- dinov3/vggt cells stay "--" on other boxes until those are
    # pushed; they were never committed, which is why this table renders them empty here while
    # B200's local renders showed them).
    jpath = os.path.join(VFM_DIR, f"{tag}.json")
    if os.path.exists(jpath) and os.path.getsize(jpath) > 0:
        try:
            v = json.load(open(jpath)).get(key)
            return None if v is None else v * scale
        except Exception:
            return None
    path = os.path.join(VFM_DIR, f"{tag}.log")
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return None
    try:
        text = open(path, errors="ignore").read()
    except Exception:
        return None
    summaries = re.findall(r"Final Summary: (\{.*?\})", text)
    if not summaries:
        return None
    m = re.search(rf"['\"]{re.escape(key)}['\"]\s*:\s*(-?[0-9.eE+]+)", summaries[-1])
    if not m:
        return None
    try:
        return float(m.group(1)) * scale
    except ValueError:
        return None


def load_accuracy(model: str, dataset: str, tag: str, key: str = "accuracy") -> Optional[float]:
    """key="accuracy" is the headline; key="mean_score" is the graded companion the oracle
    writes alongside it (mIoU for bbox specs, soft score for counting/VQA-soft) -- rows whose
    label carries "(mIoU)" or "(Soft" read it instead."""
    p = os.path.join(RESULTS, f"{model}_{dataset}", f"{tag}.json")
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        return None
    try:
        v = json.load(open(p))["summary"].get(key)
        return None if v is None else v * 100.0
    except Exception:
        return None


_INPROC = None
_INLAT = None


def inproc_flops(model: str, dataset: str, key: str) -> Optional[float]:
    global _INPROC
    if _INPROC is None:
        p = os.path.join(FLOPS_DIR, "inprocess_flops.json")
        _INPROC = json.load(open(p)) if os.path.exists(p) else {}
    return (_INPROC.get(model, {}) or {}).get(dataset, {}).get(key)


def inproc_latency(model: str, dataset: str, key: str) -> Optional[float]:
    """TTFT median (ms) from the latency probe; None when the model has no measured serving path."""
    global _INLAT
    if _INLAT is None:
        p = os.path.join(LAT_DIR, "inprocess_latency.json")
        _INLAT = json.load(open(p)) if os.path.exists(p) else {}
    return (_INLAT.get(model, {}) or {}).get(dataset, {}).get(key)


def get_latency(spec, key: str) -> Optional[float]:
    # Only the in-process specs carry a latency entry (the vLLM-served Qwen3.5 rows); the
    # offload models were never timed end to end, so their cells stay empty.
    if spec is None or spec[0] != "inproc":
        return None
    return inproc_latency(spec[1], spec[2], key)


def fmt_ms(ms: Optional[float], full: Optional[float]) -> str:
    """ms -> a latency cell, with the ratio to the full-res TTFT in parentheses (fmt_tf's twin)."""
    if ms is None:
        return "--"
    if full:
        return f"{ms:.0f}\\,ms ({100 * ms / full:.0f}\\%)"
    return f"{ms:.0f}\\,ms"


def offload_total(base: str, key: str, groups: str = "g4") -> Optional[float]:
    """Per-instruction TOTAL FLOPs of an arm: approximate pass plus every correction round.

    total/full is the compute OVERHEAD the schedule pays. It is a different question from the
    critical share, and the two move in opposite directions -- deferring less past the last byte
    generally costs more work overall.
    """
    p = os.path.join(FLOPS_DIR, f"{base}_{groups}_{key}.json")
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        return None
    try:
        j = json.load(open(p))
        return j["mean_total_gflops"] / max(int(j.get("batch_size", 1) or 1), 1)
    except Exception:
        return None


def get_total(spec, key: str) -> Optional[float]:
    if spec is None:
        return None
    if spec[0] == "inproc":
        return inproc_flops(spec[1], spec[2], f"total_{key}")
    return offload_total(spec[1], key, *spec[2:])


def offload_flops(base: str, key: str, groups: str = "g4") -> Optional[float]:
    """Per-INSTRUCTION FLOPs from a worker-written JSON.

    Divides by the recorded batch size. Arms of one model disagree on it -- NYU's ceiling runs at 1
    while its interleaved arms run at 8 -- so per-request means are not comparable and reading them
    directly made NYU's critical exceed its own ceiling.
    """
    # Optional third spec element selects the arm's group-count infix ("g1" for the one-shot rows);
    # the ceiling has no groups and is shared across them.
    tag = f"{base}_ceiling" if key == "full" else f"{base}_{groups}_{key}"
    p = os.path.join(FLOPS_DIR, f"{tag}.json")
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        return None
    try:
        j = json.load(open(p))
        bs = max(int(j.get("batch_size", 1) or 1), 1)
        field = "mean_total_gflops" if key == "full" else "mean_critical_gflops"
        return j[field] / bs
    except Exception:
        return None


def get_flops(spec, key: str) -> Optional[float]:
    if spec is None:
        return None
    if spec[0] == "inproc":
        return inproc_flops(spec[1], spec[2], key)
    return offload_flops(spec[1], key, *spec[2:])


def fmt_tf(gf: Optional[float], full: Optional[float]) -> str:
    """GFLOPs -> a TF cell, with the share of the full-res critical computation in parentheses."""
    if gf is None:
        return "--"
    tf = gf / 1000.0
    body = f"{tf:.3f}" if tf < 0.1 else f"{tf:.2f}"
    if full:
        return f"{body}\\,TF ({100 * gf / full:.1f}\\%)"
    return f"{body}\\,TF"


def build_rows(keeps, groups: int, latency: bool = False):
    """Rows of the evaluation table. latency=True appends the Lat. / Crit. Lat. cells to
    every block (the combined form used until 2026-09-09); the default is the accuracy +
    compute layout, latency having moved to its own table (emit_latency_latex)."""
    out = []
    for model, rows in SPEC:
        # Footnote marks ($^\dagger$ etc.) are display-only; every lookup table is keyed by the
        # BASE model name. Keying lookups on the decorated name silently blanks the whole block --
        # adding $^\S$ to Qwen2.5 turned its bounds into "--" before this split existed.
        base_model = model.split("$")[0]
        block = []
        for label, acc_key, fl_key in rows:
            lit = LITERALS.get((base_model, label), {})
            if base_model == "Qwen3.5-MoE (35B-A3B)" and label in QWEN35_PYR_ROWS:
                ds_name, metric = QWEN35_PYR_ROWS[label]
                lit = {**lit, **qwen35_pyr_lit(ds_name, metric)}
            if base_model == "Qwen3.5-MoE (35B-A3B)" and label in QWEN35_BOX_ROWS:
                ds_name, metric = QWEN35_BOX_ROWS[label]
                lit = {**lit, **qwen35_pyr_lit(ds_name, metric, dir_=QWEN35_BOX_DIR,
                                               expected=QWEN35_BOX_EXPECTED)}
            if base_model == "Qwen3.5 (4B)" and label in QWEN35_PYR_ROWS:
                ds_name, metric = QWEN35_PYR_ROWS[label]
                lit = {**lit, **qwen35_pyr_lit(ds_name, metric, slug=QWEN35_4B_SLUG)}
            if base_model == "Qwen3.5 (4B)" and label in QWEN35_BOX_ROWS:
                ds_name, metric = QWEN35_BOX_ROWS[label]
                lit = {**lit, **qwen35_pyr_lit(ds_name, metric, slug=QWEN35_4B_SLUG,
                                               dir_=QWEN35_BOX_DIR, expected=QWEN35_BOX_EXPECTED)}
            if base_model.startswith("Qwen3.5-MoE (122B") and label in QWEN35_BOX_ROWS \
                    and label not in ("RealWorldQA (Acc.)", "ChartQA (Relaxed Acc.)"):
                # VSR / MMVP / CV-Bench (added 2026-09-11): the vLLM form ("_c4") or the HF box
                # chain, whichever has landed; the HF value wins where both exist.
                ds_name, metric = QWEN35_BOX_ROWS[label]
                lit = {**lit,
                       **qwen35_pyr_lit(ds_name, metric, slug=QWEN122B_SLUG,
                                        dir_=QWEN122B_VLLM_DIR, expected=QWEN35_BOX_EXPECTED,
                                        suffix=QWEN122B_VLLM_SUFFIX),
                       **qwen35_pyr_lit(ds_name, metric, slug=QWEN122B_SLUG, dir_=QWEN35_BOX_DIR,
                                        expected=QWEN35_BOX_EXPECTED)}
            if base_model.startswith("Qwen3.5-MoE (122B") and label == "RealWorldQA (Acc.)":
                # Full split (765), 2026-09-08 GPU0 chain (q122b_rwqa_*.log): same dir, filter
                # (box) and greedy loop as the 35B RealWorldQA literals, so the cells render
                # normally. Streaming ran with the Triton finegrained-fp8 fallback like the probe.
                # keep<1 arms (2026-09-09, deferred patch score) ran through the vLLM form into
                # qwen_vllm_accuracy/ ("_c4"); the HF chain's bounds and keep=1 arm win where
                # both exist (the vLLM bounds are the engine-band gate copies).
                lit = {**lit,
                       **qwen35_pyr_lit("realworldqa", "ok", slug=QWEN122B_SLUG,
                                        dir_=QWEN122B_VLLM_DIR, expected={"realworldqa": 765},
                                        suffix=QWEN122B_VLLM_SUFFIX),
                       **qwen35_pyr_lit("realworldqa", "ok", slug=QWEN122B_SLUG,
                                        dir_=QWEN35_BOX_DIR, expected={"realworldqa": 765})}
            if base_model.startswith("Qwen3.5-MoE (122B") and label == "ChartQA (Relaxed Acc.)":
                # vLLM form, box filter (the ChartQA convention of every other Qwen3.5 row).
                lit = {**lit, **qwen35_pyr_lit("chartqa", "ok", slug=QWEN122B_SLUG,
                                               dir_=QWEN122B_VLLM_DIR, expected={"chartqa": 2500},
                                               suffix=QWEN122B_VLLM_SUFFIX)}
            if base_model.startswith("Qwen3.5-MoE (122B") and label in QWEN35_PYR_ROWS:
                ds_name, metric = QWEN35_PYR_ROWS[label]
                # HF-chain rows first (VisDrone Count, V*Bench), the vLLM form for the rest
                # (VisDrone Det, TextVQA, RefCOCO); an arm present in both keeps the HF value.
                full = {**qwen35_pyr_lit(ds_name, metric, slug=QWEN122B_SLUG,
                                         dir_=QWEN122B_VLLM_PYR_DIR, expected=QWEN35_PYR_EXPECTED,
                                         suffix=QWEN122B_VLLM_SUFFIX),
                        **qwen35_pyr_lit(ds_name, metric, slug=QWEN122B_SLUG, dir_=QWEN35_PYR_DIR,
                                         expected=QWEN35_PYR_EXPECTED)}
                if label == "V*Bench (Acc.)" or full:
                    # Full-split 122B values (V*Bench n=191 IS the benchmark; VisDrone Count 2350
                    # from the 2026-09-08 chain) render normally; arms still incomplete in the
                    # full run fall through to the parenthesized probe below only when NOTHING
                    # full-split exists for the row.
                    # V*Bench n=191 IS the whole benchmark: a full-split value, rendered
                    # normally. The parenthesized-probe rule covers reduced-n subsets only.
                    lit = {**lit, **full}
                else:
                    # User directive (2026-09-01): 122B probe numbers render PARENTHESIZED, never
                    # shaded, no preservation % -- they must not read as full-split values.
                    lit = {**lit, **qwen35_pyr_lit(ds_name, metric, slug=QWEN122B_SLUG,
                                                   dir_=QWEN122B_DIR,
                                                   expected=QWEN122B_EXPECTED),
                           "probe": True}
            if base_model == "Muse Glimmer (29.6B)" and label in MG_PYR_ROWS:
                ds_name, metric = MG_PYR_ROWS[label]
                lit = {**lit, **qwen35_pyr_lit(ds_name, metric, dir_=MG_PYR_DIR,
                                               expected=MG_PYR_EXPECTED)}
            f = "{:.2f}"
            if "fmt" in lit:
                f = lit["fmt"]
            lower_better = r"\downarrow" in label

            _metric_key = "mean_score" if ("(mIoU)" in label or "(Soft" in label) else "accuracy"

            def acc_raw(tag, lit_key=None):
                v = load_accuracy(*acc_key, tag, _metric_key) if acc_key else None
                if v is None and lit_key and lit_key in lit:
                    v = lit[lit_key]
                return v

            ceiling_v = acc_raw("ceiling", "ceiling")

            def acc_with_pres(tag, lit_key=None, shade_hi=False):
                """Accuracy, with (preservation vs.\\ ceiling %) in parentheses.

                Preservation is this-value-relative-to-ceiling, not the other way round: for a
                lower-is-better metric that means ceiling/value, so a value further from the
                ceiling in the bad direction still reads as a preservation < 100%. Applied to
                Low-res. and Ours -- Full-res. is the reference point itself, always 100%.

                `shade_hi` (Ours cells only) marks preservation >= SHADE_PRES with a light-gray
                \\cellcolor so near-ceiling cells read at a glance; emit_md strips the macro.
                """
                v = acc_raw(tag, lit_key)
                if v is None:
                    return "--"
                if lit.get("probe"):
                    return f"({f.format(v)})"
                s = f.format(v)
                if ceiling_v:
                    pres = 100.0 * ((ceiling_v / v) if lower_better else (v / ceiling_v))
                    s += f" ({pres:.1f}\\%)"
                    if shade_hi and pres >= SHADE_PRES:
                        s = SHADE_MACRO + s
                return s

            vfm = VFM_OURS.get((base_model, label))

            def ours(k):
                """Ours at keep `k`. VLM rows read a summary JSON; VFM rows read a campaign log."""
                if vfm is not None:
                    tag, key, scale = vfm
                    v = vfm_accuracy(f"{tag}_k{k:.2f}", key, scale)
                    if v is None:
                        return "--"
                    out = f.format(v)
                    if ceiling_v:
                        pres = 100.0 * ((ceiling_v / v) if lower_better else (v / ceiling_v))
                        out += f" ({pres:.1f}\\%)"
                        if pres >= SHADE_PRES:
                            out = SHADE_MACRO + out
                    return out
                # Tag preference: canonical progressive arm where re-measured; the upfront
                # interleaved arm's file otherwise (ddagger caveat); the one-shot corrected
                # arm last (gemma4-class models whose interleaved walk is not ported yet).
                for tag in (f"progressive_g{groups}_k{k:.2f}",
                            f"interleaved_g{groups}_k{k:.2f}",
                            f"streaming_g{groups}_k{k:.2f}",     # mistral3_oracle's name for it
                            f"corrected_k{k:.2f}"):
                    v = acc_with_pres(tag, lit_key=f"k{k:.2f}", shade_hi=True)
                    if v != "--":
                        return v
                return "--"

            full_gf = get_flops(fl_key, "full")
            full_ms = get_latency(fl_key, "full")
            cells = [acc_with_pres("floor", "floor")]
            for k in keeps:
                cells.append(ours(k))
                # Comp. = the arm's TOTAL backbone compute (approximate pass + every correction
                # round, overlapped work included), vs Crit. Comp. = only what waits on the last
                # byte. The two move in opposite directions -- deferring less costs more overall --
                # which is why both columns exist side by side.
                cells.append(fmt_tf(get_total(fl_key, f"k{k:.2f}"), full_gf))
                cells.append(fmt_tf(get_flops(fl_key, f"k{k:.2f}"), full_gf))
                # Lat. / Crit. Lat. are the measured twins: TTFT from t0 (inputs on the GPU, the
                # whole schedule serialized; served path: the server's receipt of the open) vs
                # TTFT from the GPU completion of band g-2's vision work (= chunk g-2's departure;
                # the last band's correction, transfer, merge and remaining prefill -- Crit.
                # Comp.'s accounting). Anchors fixed 2026-09-09; keep<1 cells print "--" until
                # re-measured (the probe stashes the old CPU-issue-anchored values).
                if latency:
                    cells.append(fmt_ms(get_latency(fl_key, f"total_k{k:.2f}"), full_ms))
                    cells.append(fmt_ms(get_latency(fl_key, f"k{k:.2f}"), full_ms))
            # Streaming (keep=1.0) block: the causal-LLM category (LLM prefills exactly once,
            # vision corrects everything progressively). Accuracy comes from a "stream" literal
            # (Qwen3.5's jsonl-driven runs) or a streaming_g4.json beside the row's other arms
            # (OV2's oracle); compute from the k1.00 inproc keys. Non-causal models leave all
            # three cells empty -- Gemma 3's image tokens are bidirectional, and the VFMs have no
            # LLM to stream.
            sv = None
            if "stream" in lit:
                sv = lit["stream"]
            elif acc_key:
                sv = load_accuracy(*acc_key, "streaming_g4", _metric_key)
            if sv is None:
                cells.append("--")
            elif lit.get("probe"):
                cells.append(f"({f.format(sv)})")
            else:
                out_s = f.format(sv)
                if ceiling_v:
                    pres = 100.0 * ((ceiling_v / sv) if lower_better else (sv / ceiling_v))
                    out_s += f" ({pres:.1f}\\%)"
                    if pres >= SHADE_PRES:
                        out_s = SHADE_MACRO + out_s
                cells.append(out_s)
            cells.append(fmt_tf(get_total(fl_key, "k1.00"), full_gf))
            cells.append(fmt_tf(get_flops(fl_key, "k1.00"), full_gf))
            if latency:
                cells.append(fmt_ms(get_latency(fl_key, "total_k1.00"), full_ms))
                cells.append(fmt_ms(get_latency(fl_key, "k1.00"), full_ms))
            if ceiling_v is None:
                cells.append("--")
            else:
                cells.append(f"({f.format(ceiling_v)})" if lit.get("probe") else f.format(ceiling_v))
            cells.append(fmt_tf(full_gf, None))
            if latency:
                cells.append(fmt_ms(full_ms, None))
            if (base_model, label) in CAPABILITY_LIMIT:
                label = label + r"\,$\diamond$"
            block.append((label, cells))
        out.append((model, block))
    return out


# Datasets STRUCK THROUGH in the tables (user, 2026-09-12). The rows stay and their measured cells
# stay; only the dataset label is struck, to mark a dataset that is SATURATED -- floor and ceiling
# statistically indistinguishable -- so no keep arm on that row can say anything about the method.
# VSR: 122B floor and ceiling are both 90.9165 on the full 1222 rows, 66 ok-discordant pairs
# splitting exactly 33/33, sign test p=1.00; 35B p=0.268 (66 discordant, 38/28). Measurement of the
# remaining 122B VSR arms was stopped on that basis. Keeping the numbers visible is deliberate: a
# reader should see the saturation rather than wonder why the dataset vanished. NOTE: 4B VSR is
# NOT saturated (+1.72 pp, p=0.035) -- it is the one place VSR still carries signal, and it carries
# it as a zero-recovery row (k=1 lands exactly on floor, 68 discordant 34/34).
# Requires \usepackage[normalem]{ulem} in the preamble.
STRUCK_DATASETS = {"VSR zeroshot (Acc.)", "VSR"}


def struck(label: str) -> str:
    """Wrap a struck dataset label in \sout{}; pass everything else through unchanged."""
    return f"\\sout{{{label}}}" if label in STRUCK_DATASETS else label


def emit_latex(table, keeps, latency: bool = False) -> str:
    # Each Ours / Streaming block: Acc., Comp., Crit. Comp. (+ Lat., Crit. Lat. in the combined
    # form); Full-res.: Acc., Comp. (+ Lat.).
    W, F = (5, 3) if latency else (3, 2)
    heads = " & ".join(f"\\multicolumn{{{W}}}{{c}}{{Ours ({int(k*100)}\\%)}}" for k in keeps)
    heads += f" & \\multicolumn{{{W}}}{{c}}{{Streaming (k$=$1.0)}}"
    cmids, col = [], 4
    for _ in keeps:
        cmids.append(f"\\cmidrule(lr){{{col}-{col+W-1}}}")
        col += W
    cmids.append(f"\\cmidrule(lr){{{col}-{col+W-1}}}")     # streaming block
    col += W
    cmids.append(f"\\cmidrule(lr){{{col}-{col+F-1}}}")
    blk = "Acc. (\\%) & Comp. & Crit. Comp." + (" & Lat. & Crit. Lat." if latency else "")
    sub = " & ".join([blk] * (len(keeps) + 1)
                     + ["Acc. (\\%) & Comp." + (" & Lat." if latency else "")])
    # 2 labels + Low-res. + W per Ours block + W for Streaming + F for Full-res.
    ncol = 2 + 1 + W * len(keeps) + W + F
    lat_sent = (
             r"Lat.\ and Crit.\ Lat.\ are their measured twins, single-request "
             r"time-to-first-token medians (ms, B200, concurrency 1, 36 evenly spaced images): "
             r"Lat.\ with the whole image on the GPU at $t{=}0$ (vision and prefill serialized, no "
             r"transmission credit; clock starts at the server's receipt of the request), Crit.\ "
             r"Lat.\ from the moment the previous band's vision work completes on the GPU (the "
             r"departure of the second-to-last chunk), i.e.\ the last band's vision correction, "
             r"its transfer, merge and remaining prefill, the same accounting as Crit.\ Comp.; "
             r"parentheses are the ratio to the Full-res.\ TTFT. Both Qwen3.5 models are timed on "
             r"their served path (vLLM 0.28 engine, FP8 for 122B, fed by the in-process vision "
             r"tower, one GPU shared by both; keep$<$1 latency cells await re-measurement under "
             r"these anchors and show --); the other VLMs on the in-process HF eager path. "
             ) if latency else (
             r"Measured latency is reported separately in Table~\ref{tab:latency_results}. ")
    L = []
    L.append(r"% requires \usepackage[table]{xcolor} in the preamble (for \cellcolor)")
    L.append(r"% requires \usepackage[normalem]{ulem} (for \sout on saturated dataset labels)")
    L.append(r"\begin{table*}[t]")
    L.append(r"\vspace{-0.1in}")
    # The long caption moved to docs/memo/eval_table_notes.md (2026-09-11): the table is a working
    # view; the caption keeps the column definitions and a one-line legend of the cell markers.
    L.append(r"\caption{Evaluation results. Crit.\ Comp.: backbone prefill FLOPs per "
             r"instruction that can only start once the whole image has arrived (decode "
             r"excluded); Comp.: total backbone compute incl.\ work overlapped with transmission; "
             r"parentheses: ratio to Full-res. Ours = interleaved $g{=}4$ (progressive per-round "
             r"selection); Streaming ($k{=}1$) = causal-LLM chunked prefill with progressive "
             r"vision correction. Shaded: $\geq$98\% of Full-res.\ accuracy. Markers: "
             r"$^\dagger$ Qwen3.5 Ours = keep-limited streaming arm; $^\ddagger$ compute from "
             r"the progressive arm, accuracy partly from the earlier upfront arm; $^\S$ batch-1 "
             r"vs.\ batch-16 bounds; \textsuperscript{\P} 122B-FP8 (parenthesized = 240-row "
             r"probe); $\diamond$ model not capable of the task; $^\ast$ OpenVLA cumulative "
             r"correction. Every Qwen3.5-35B row has all its arms from one box, code version and "
             r"degrade filter EXCEPT RefCOCO, whose Low-/Full-res.\ and Streaming cells predate "
             r"the 2026-09-11 re-measurement (same pyr filter, different box); its Ours cells "
             r"should be read against each other rather than against those bounds. "
             r"Notes: docs/memo/eval\_table\_notes.md.}")
    L.append(r"\label{tab:evaluation_results}")
    L.append(r"\begin{center}\begin{small}\begin{sc}")
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\begin{tabular}{ll" + "c" * (ncol - 2) + "}")
    L.append(r"\toprule")
    L.append(r"\multirow{2}{*}{Model} & \multirow{2}{*}{Dataset (Metric)} & "
             r"\multicolumn{1}{c}{Low-res.} & " + heads +
             f" & \\multicolumn{{{F}}}{{c}}{{Full-res.}} \\\\")
    L.append(r"\cmidrule(lr){3-3} " + " ".join(cmids))
    L.append(r"& & Acc. (\%) & " + sub + r" \\")
    L.append(r"\midrule")
    # Section boundaries get a DOUBLE rule (user 2026-09-01): before Gemma 3 (VFM->VLM),
    # before LLaVA-OV2 (Gemma family -> the rest), and before OpenVLA (VLM -> VLA).
    DOUBLE_RULE_BEFORE = ("Gemma 3", "LLaVA-OV2", "OpenVLA")
    for i, (model, rows) in enumerate(table):
        if i:
            L.append(r"\midrule\midrule" if model.startswith(DOUBLE_RULE_BEFORE)
                     else r"\midrule")
        # "Name (size)" wraps to two lines inside the multirow cell -- the size (and any footnote
        # marks after it) drops to the second line, keeping the model column narrow.
        m = re.match(r"^(.*?) (\(.*)$", model)
        cell = f"\\shortstack[l]{{{m.group(1)}\\\\{m.group(2)}}}" if m else model
        L.append(f"\\multirow{{{len(rows)}}}{{*}}{{{cell}}}")
        for label, cells in rows:
            L.append(f"& {struck(label)} & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{sc}\end{small}\end{center}")
    L.append(r"\vspace{-0.22in}")
    L.append(r"\end{table*}")
    return "\n".join(L)


def emit_md(table, keeps, latency: bool = False) -> str:
    lat = ["lat", "crit-lat"] if latency else []
    hdr = ["model", "dataset", "low-res"]
    for k in keeps:
        hdr += [f"ours{int(k*100)} {c}" for c in ["acc", "comp", "crit"] + lat]
    hdr += [f"stream {c}" for c in ["acc", "comp", "crit"] + lat]
    hdr += ["full acc", "full comp"] + (["full lat"] if latency else [])
    L = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for model, rows in table:
        for label, cells in rows:
            L.append("| " + " | ".join([model, label] +
                                       [c.replace(SHADE_MACRO, "").replace("\\,", " ")
                                         .replace("\\%", "%")
                                        for c in cells]) + " |")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------
# Latency table (split out of the evaluation table 2026-09-09): the served Qwen3.5 rows only.
# Part (a) is the single-request breakdown behind the former Lat. / Crit. Lat. cells, straight
# from inprocess_latency.json's `detail` block (per-column medians over the probe's 36 images --
# the columns of one row therefore do not sum); part (b) the concurrency sweeps
# (conc_sweep_20260909.json, folded by vllm_stream/conc_sweep_20260909/conc_table_json.py).
# Keys with the `_cg1024` suffix are the 2026-09-09 re-measurement with the engine's CUDA-graph
# capture limit raised to 1024 tokens (server --max-cudagraph-capture-size 1024; vLLM's default
# is 2 x max_num_seqs = 128 / 64, so every prefill step above it -- most chunk steps and every
# short full-res prompt -- ran eager at ~+20 ms). The bare keys keep the default-capture numbers.
LAT_MODELS = [("qwen35_4b_cg1024", "Qwen3.5 (4B)"),
              ("qwen35_moe_cg1024", "Qwen3.5-MoE (35B-A3B)"),
              ("qwen35_122b_cg1024", "Qwen3.5-MoE (122B-A10B FP8)")]
LAT_DATASETS = [("chartqa", "ChartQA"), ("realworldqa", "RealWorldQA"), ("vsr", "VSR"),
                ("mmvp", "MMVP"), ("cvbench", "CV-Bench"), ("refcoco", "RefCOCO val"),
                ("textvqa", "TextVQA"), ("visdrone_count", "VisDrone Count"),
                ("visdrone_det", "VisDrone Det"), ("vstar", "V*Bench")]
CONC_DATASETS = [("realworldqa", "RealWorldQA"), ("visdrone_det", "VisDrone Det")]


def _lat_entry(model: str, dataset: str) -> dict:
    inproc_latency(model, dataset, "full")          # loads _INLAT
    return (_INLAT.get(model, {}) or {}).get(dataset) or {}


def _ms(v: Optional[float], full: Optional[float] = None, ratio: bool = False) -> str:
    if v is None:
        return "--"
    if ratio and full:
        return f"{v:.0f} ({100 * v / full:.0f}\\%)"
    return f"{v:.0f}"


def emit_latency_latex(keeps) -> str:
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\vspace{-0.1in}")
    L.append(r"% requires \usepackage[normalem]{ulem} (for \sout on saturated dataset labels)")
    L.append(r"\caption{Measured latency of the served Qwen3.5 models (vLLM 0.28 engine on one "
             r"B200, FP8 for 122B, fed by the in-process vision tower; CUDA-graph capture limit raised "
             r"to 1024 tokens from vLLM's default of $2\times$max-num-seqs, so prefill steps up to "
             r"1024 tokens replay a captured graph for both paths -- the default left every chunk "
             r"step above 128 (35B) / 64 (122B) tokens and every short full-resolution prompt "
             r"eager, ${\approx}20$\,ms of launch overhead per step). A \emph{band} is one of "
             r"$g{=}4$ contiguous raster-order strips of the image's vision tokens (the unit that "
             r"arrives, is corrected and is pushed to the engine as one \emph{chunk}); the prompt "
             r"is [leading text][vision tokens][trailing text], so chunk 0 carries the leading text "
             r"with band 0 and chunk $g{-}1$ the trailing text with band $g{-}1$. "
             r"\textbf{(a)} Single-request time-to-first-token (TTFT) breakdown, medians over 36 "
             r"evenly spaced images per dataset, concurrency 1; every column is its own median, so "
             r"the columns of a row do not sum. Tokens: prompt length. Full-res.: Vision is the "
             r"one-shot tower, TTFT the clock from inputs-on-GPU (the server's receipt of the "
             r"request) to the first token, i.e.\ Lat.\ with the whole image at $t{=}0$ and no "
             r"transmission credit. Streaming (k$=$1.0): Vision is the approximate pass plus all "
             r"four correction rounds and chunk pushes (the approximate pass itself is never sent "
             r"to the engine); First / Last: when chunk 0 (band 0 after its correction, with the "
             r"leading text) and the last chunk left the driver; First$\to$FT / "
             r"Last$\to$FT: the engine's own span from receiving the first / the last chunk to "
             r"the first token; Lat.: TTFT from $t{=}0$ with the bands back-to-back (every pass "
             r"serialized, the analogue of Comp.); Crit.\ Lat.: TTFT from the arrival of the last "
             r"band's pixels, measured with the bands spaced 150\,ms apart as on a slow link (so "
             r"no chunk queues behind the previous chunk's prefill step and the tower does not "
             r"overlap an engine step), i.e.\ the last band's correction, its transfer and its "
             r"prefill step plus the first decode -- the accounting of Crit.\ Comp.\ in "
             r"Table~\ref{tab:evaluation_results}; the other streaming columns come from the "
             r"back-to-back run. Both Full-res.\ TTFT and Crit.\ Lat.\ are clocks that start at the "
             r"last byte of the image, so on a link with transfer time $T_{tx}$ the user-visible TTFT "
             r"is $T_{tx}$ plus either column for either path: the absolute saving is independent of "
             r"the link, the relative one shrinks as the link slows, and Lat.\ is the infinite-bandwidth "
             r"limit ($T_{tx}{=}0$) where the serialized streaming passes lose. Parentheses "
             r"give the ratio to the Full-res.\ TTFT. Ours (keep-limited streaming) columns are the "
             r"same two clocks for the keep 0.50 / 0.25 arms under the deferred patch score (band 0 "
             r"ranked on energy alone, the received-attention term computed right after chunk 0 is "
             r"pushed): that score pass sits on the serialized path, so their Lat.\ exceeds "
             r"k$=$1.0's, while their Crit.\ Lat.\ matches it (band $g{-}1$ corrects fewer rows "
             r"but its prefill step is the same). "
             r"\textbf{(b)} Closed-loop concurrency sweep on the 35B model at the engine's default "
             r"capture limit (128; measured before the limit was raised) (RealWorldQA 765 / "
             r"VisDrone Det 448 splits, 160 or 400 evenly spaced images per cell): $N$ driver "
             r"processes (one vision tower each, same GPU) $\times$ $c$ requests in flight per "
             r"driver against one engine. Med./p90 in ms; req/s and engine busy (share of wall "
             r"time inside an engine step) over the active phase of each run. Ratio is streaming "
             r"Crit.\ Lat.\ over the Full-res.\ TTFT of the same configuration. Streaming's "
             r"capacity is about half the Full-res.\ path's because each request costs the engine "
             r"four chunk-sized prefill steps near the MoE step floor plus four vision passes that "
             r"share the GPU; at equal throughput the Full-res.\ path is the lower-latency one.}")
    L.append(r"\label{tab:latency_results}")
    L.append(r"\begin{center}\begin{small}\begin{sc}")
    # ---- (a) single-request breakdown ----
    ours = [(k, f"Ours ({int(k*100)}\\%)") for k in sorted(keeps, reverse=True)]
    ncol = 2 + 1 + 2 + 7 + 2 * len(ours)
    L.append(r"\textbf{(a) Single-request TTFT breakdown (ms)}\\[2pt]")
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\begin{tabular}{ll" + "r" * (ncol - 2) + "}")
    L.append(r"\toprule")
    hdr = (r"\multirow{2}{*}{Model} & \multirow{2}{*}{Dataset} & \multirow{2}{*}{Tokens} & "
           r"\multicolumn{2}{c}{Full-res.} & \multicolumn{7}{c}{Streaming (k$=$1.0)}")
    for _, name in ours:
        hdr += f" & \\multicolumn{{2}}{{c}}{{{name}}}"
    L.append(hdr + r" \\")
    cm = [r"\cmidrule(lr){4-5}", r"\cmidrule(lr){6-12}"]
    col = 13
    for _ in ours:
        cm.append(f"\\cmidrule(lr){{{col}-{col+1}}}")
        col += 2
    L.append(" ".join(cm))
    L.append(r"& & & Vision & TTFT & Vision & First & Last & First$\to$FT & Last$\to$FT & Lat. & "
             r"Crit. Lat." + " & Lat. & Crit. Lat." * len(ours) + r" \\")
    L.append(r"\midrule")
    for mi, (mkey, mname) in enumerate(LAT_MODELS):
        rows = [(d, dn) for d, dn in LAT_DATASETS if _lat_entry(mkey, d).get("full") is not None]
        if not rows:
            continue
        if mi:
            L.append(r"\midrule")
        m = re.match(r"^(.*?) (\(.*)$", mname)
        cell = f"\\shortstack[l]{{{m.group(1)}\\\\{m.group(2)}}}" if m else mname
        L.append(f"\\multirow{{{len(rows)}}}{{*}}{{{cell}}}")
        for d, dn in rows:
            e = _lat_entry(mkey, d)
            full = e.get("full")
            dc = (e.get("detail") or {}).get("ceiling") or {}
            ds = (e.get("detail") or {}).get("streaming_k1.00") or {}
            c = [f"{dc['prompt_tokens']:.0f}" if dc.get("prompt_tokens") else "--",
                 _ms(dc.get("t_vision_ms")), _ms(full),
                 _ms(ds.get("t_vision_ms")), _ms(ds.get("t_open_ms")), _ms(ds.get("t_last_push_ms")),
                 _ms(ds.get("ttft_open_ms")), _ms(ds.get("ttft_last_chunk_ms")),
                 _ms(e.get("total_k1.00"), full, True), _ms(e.get("k1.00"), full, True)]
            for k, _ in ours:
                c += [_ms(e.get(f"total_k{k:.2f}"), full, True), _ms(e.get(f"k{k:.2f}"), full, True)]
            L.append(f"& {struck(dn)} & " + " & ".join(c) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    # ---- (b) concurrency sweep ----
    p = os.path.join(LAT_DIR, "conc_sweep_20260909.json")
    conc = json.load(open(p))["rows"] if os.path.exists(p) else []
    if conc:
        L.append(r"\\[6pt]")
        L.append(r"\textbf{(b) Concurrency sweep, Qwen3.5-MoE (35B-A3B), keep$=$1.0}\\[2pt]")
        L.append(r"\resizebox{\textwidth}{!}{%")
        L.append(r"\begin{tabular}{llrrrrrrrrrrrr}")
        L.append(r"\toprule")
        L.append(r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Config} & \multicolumn{4}{c}{Full-res.} "
                 r"& \multicolumn{6}{c}{Streaming (k$=$1.0)} & \multirow{2}{*}{Ratio} \\")
        L.append(r"\cmidrule(lr){3-6} \cmidrule(lr){7-12}")
        L.append(r"& & TTFT & p90 & req/s & busy & Lat. & p90 & Crit. Lat. & p90 & req/s & busy & \\")
        L.append(r"\midrule")
        first = True
        for d, dn in CONC_DATASETS:
            rs = [r for r in conc if r["ds"] == d]
            cfgs = sorted({(r["N"], r["c"]) for r in rs})
            if not cfgs:
                continue
            if not first:
                L.append(r"\midrule")
            first = False
            L.append(f"\\multirow{{{len(cfgs)}}}{{*}}{{{dn}}}")
            for N, cc in cfgs:
                ce = next((r for r in rs if r["N"] == N and r["c"] == cc and r["arm"] == "ceiling"), None)
                st = next((r for r in rs if r["N"] == N and r["c"] == cc and r["arm"] == "streaming"), None)
                if ce is None or st is None:
                    continue
                cfg = f"$N{{=}}{N}$, $c{{=}}{cc}$" if N > 1 else f"1 driver, $c{{=}}{cc}$"
                ratio = st["crit_med"] / ce["tot_med"] if ce.get("tot_med") else None
                cells = [_ms(ce.get("tot_med")), _ms(ce.get("tot_p90")), f"{ce['req_s']:.1f}",
                         f"{ce['busy']:.0f}\\%",
                         _ms(st.get("tot_med")), _ms(st.get("tot_p90")),
                         _ms(st.get("crit_med")), _ms(st.get("crit_p90")), f"{st['req_s']:.1f}",
                         f"{st['busy']:.0f}\\%",
                         "--" if ratio is None else f"{ratio:.2f}"]
                L.append(f"& {cfg} & " + " & ".join(cells) + r" \\")
        L.append(r"\bottomrule")
        L.append(r"\end{tabular}}")
    L.append(r"\end{sc}\end{small}\end{center}")
    L.append(r"\vspace{-0.22in}")
    L.append(r"\end{table*}")
    return "\n".join(L)



# ---------------------------------------------------------------------------------------------
# Interleaved LLM correction on Qwen3.5 (develop/vllm-interleaved-engine, 2026-09-10): the LLM
# decoder re-corrects the rows of each vision band inside the vLLM engine instead of prefilling
# once after the vision stream (docs/memo/vllm_interleaved_design.md). Its own table
# (--table interleaved) -- streaming vs interleaved at every keep, both arms on ONE served
# engine (paired rows, same driver, same vision tower) -- until the user decides how the main
# table carries it. The rows/FLOPs/latency live in the interleaved-engine worktree until the
# branch is squash-merged; IL_ROOT falls back to that worktree when the local tree lacks them.
IL_ROOT = next((r for r in (RESULTS,
                            "/NHNHOME/share/cjpark/AppCorr-il-engine/analysis/results")
                if os.path.isdir(os.path.join(r, "qwen_vllm_accuracy_il_pyr"))), RESULTS)
IL_MODELS = [  # (display, slug, file suffix, expected n per dataset, probe?, flops json per ds,
               #  interleaved latency key, streaming latency key, inproc flops key)
    ("Qwen3.5-MoE (35B-A3B)", "_qwen3.5-35b-a3b", "_c4",
     {"vstar": 191, "realworldqa": 765, "textvqa": 5000, "infovqa": 2801,
      "visdrone_count": 2350, "visdrone_det": 448,
      "chartqa": 2500, "cvbench": 2638, "mmvp": 300, "refcoco": 8811}, False,
     {"vstar": "qwen35_flops_il.json", "realworldqa": "qwen35_flops_il_rwqa.json",
      "textvqa": "qwen35_flops_il_ext.json", "infovqa": "qwen35_flops_il_ext.json",
      "visdrone_count": "qwen35_flops_il_ext.json", "visdrone_det": "qwen35_flops_il_ext.json",
      "chartqa": "qwen35_flops_il_ext2.json", "cvbench": "qwen35_flops_il_ext2.json",
      "mmvp": "qwen35_flops_il_ext2.json", "refcoco": "qwen35_flops_il_ext2.json"},
     "qwen35_35b_il", "qwen35_moe_cg1024", "qwen35_moe"),
    # 122B: 40-row V* probe on the interleaved server (c2: the DeltaNet side buffers sit outside
    # the KV budget), parenthesized per the 2026-09-01 probe rule; FLOPs json is the 2026-09-10
    # hooked re-measure (FP8Experts fix; qwen35_122b_flops_fixed.json, --il-rows), so its vision
    # half is the 122B run's own streaming arm (the older decoder-only fold that borrowed the 35B
    # tower is kept as qwen35_122b_flops_il_decoderonly.json).
    # The 2026-09-10 extension (TextVQA / InfoVQA / VisDrone, interleaved + staged arms only,
    # user go) runs 122B on 240-row strided subsets (--samples 240), parenthesized like the V*
    # probe; their FLOPs fold (--il-only) into the fixed json's own hooked rows, InfoVQA being
    # a separate hooked run (it was not in the 2026-09-10 re-measure).
    ("Qwen3.5-MoE (122B-A10B FP8)", "_qwen3.5-122b-a10b-fp8", "_c2",
     {"vstar": 40, "realworldqa": 240, "textvqa": 240, "infovqa": 240, "visdrone_count": 240,
      "visdrone_det": 240, "chartqa": 240, "cvbench": 240, "mmvp": 240, "refcoco": 240},
     True,
     {"vstar": "qwen35_122b_flops_il.json", "textvqa": "qwen35_122b_flops_fixed.json",
      "visdrone_count": "qwen35_122b_flops_fixed.json",
      "visdrone_det": "qwen35_122b_flops_fixed.json",
      "infovqa": "qwen35_122b_flops_infovqa.json",
      "realworldqa": "qwen35_122b_flops_fixed.json",
      "chartqa": "qwen35_122b_flops_fixed.json", "cvbench": "qwen35_122b_flops_ext2.json",
      "mmvp": "qwen35_122b_flops_ext2.json", "refcoco": "qwen35_122b_flops_fixed.json"},
     "qwen35_122b_il", "qwen35_122b_cg1024", "qwen35_122b"),
    # GLM-4.6V (user go 2026-09-12 20:40). The request was GLM-4.7-Flash (30B-A3B), which is a
    # text-only model (Glm4MoeLiteForCausalLM, no vision_config) and so has no image rows to
    # stream or correct; the user chose the GLM vision MoE sibling instead: 106B-A12B, 46 softmax
    # GQA layers (no DeltaNet), 128 routed experts top-8 + 1 shared, 24-layer full-attention ViT,
    # FP8 checkpoint (zai-org/GLM-4.6V-FP8, compressed-tensors) on one B200 like the 122B row.
    # Slug follows the driver: args.model.split("/")[-1].lower(). Every cell renders "--" until the
    # port lands (docs/memo/glm46v_port_plan.md); concurrency 2 like the 122B row.
    ("GLM-4.6V (106B-A12B FP8)", "_glm-4.6v-fp8", "_c2",
     {"vstar": 191, "realworldqa": 765, "textvqa": 5000, "infovqa": 2801,
      "visdrone_count": 2350, "visdrone_det": 448,
      "chartqa": 2500, "cvbench": 2638, "mmvp": 300, "refcoco": 8811}, False,
     {ds: "glm46v_flops_il.json" for ds in ("vstar", "realworldqa", "textvqa", "infovqa",
                                            "visdrone_count", "visdrone_det", "chartqa",
                                            "cvbench", "mmvp", "refcoco")},
     "glm46v_il", "glm46v_cg1024", "glm46v"),
]
IL_DATASETS = [("vstar", "V*Bench (Acc.)", "ok", "qwen_vllm_accuracy_il_pyr"),
               ("realworldqa", "RealWorldQA (Acc.)", "ok", "qwen_vllm_accuracy_il"),
               # 2026-09-10 extension: pyr filter on all four (the chain-2b convention for
               # TextVQA / VisDrone; InfoVQA has no Qwen precedent and follows them). Metric
               # keys as in the main table: TextVQA = mean(val) (VQA soft score), InfoVQA =
               # mean(val) (ANLS), VisDrone Count = exact match, Det = Acc.@IoU0.5.
               ("textvqa", "TextVQA (VQA Acc.)", "val", "qwen_vllm_accuracy_il_pyr"),
               ("infovqa", "InfoVQA (ANLS)", "val", "qwen_vllm_accuracy_il_pyr"),
               ("visdrone_count", "VisDrone Count (Exact Acc.)", "ok", "qwen_vllm_accuracy_il_pyr"),
               ("visdrone_det", "VisDrone Det (Acc.@0.5)", "ok", "qwen_vllm_accuracy_il_pyr"),
               # 2026-09-12 extension (user go): the four main-table datasets the interleaved
               # table lacked. VSR is deliberately NOT added -- it is saturated on both models
               # (floor vs ceiling p=0.268 on 35B, 0.403 on 122B), so it carries no signal about
               # the schedule. Filter follows the campaign convention: box -> _il, pyr -> _il_pyr.
               ("chartqa", "ChartQA (Relaxed Acc.)", "ok", "qwen_vllm_accuracy_il"),
               ("cvbench", "CV-Bench (Acc.)", "ok", "qwen_vllm_accuracy_il"),
               ("mmvp", "MMVP (Acc.)", "ok", "qwen_vllm_accuracy_il"),
               ("refcoco", "RefCOCO val (Acc.@0.5)", "ok", "qwen_vllm_accuracy_il_pyr")]
IL_KEEPS = [1.0, 0.5, 0.25]
# Adaptive-k thresholds per (model slug, dataset): the `--keep auto` theta that realises a target
# mean k, calibrated per dataset by threshold_sim.py on a 36-image pscore dump (the score is in
# raw pixel units, so theta is NOT portable across datasets). The interleaved table's rightmost
# group reads the unified_staged auto arms through this registry.
ADAPTIVE_THETA_JSON = "adaptive_theta.json"


def adaptive_thetas(slug: str, dataset: str) -> Dict[float, float]:
    """{target k: theta} for one (model, dataset), from analysis/results/adaptive_theta.json
    (local tree first, then IL_ROOT); {} when uncalibrated."""
    want = slug.lstrip("_").lower()
    for root in (RESULTS, IL_ROOT):
        p = os.path.join(root, ADAPTIVE_THETA_JSON)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        # registry shape (B200-8, 2026-09-13): {"_meta": {"model": ...}, "entries": [{dataset,
        # filter, target_k, theta, realised_mean_k, measured_mean_k, ...}]}; an entry may carry
        # its own "model" when the file holds several models.
        default_model = (d.get("_meta") or {}).get("model", "")
        out = {}
        for e in d.get("entries", []):
            m = str(e.get("model", default_model)).split("/")[-1].lower()
            if m == want and e.get("dataset") == dataset:
                out[float(e["target_k"])] = float(e["theta"])
        if out:
            return out
    return {}
# The progressive arm's LLM schedules, as (table key, row-file tag). The adaptive arm exists for
# each of them; its file tag is `{schedule}_g{groups}_auto{theta:g}` (qwen_vllm_accuracy.
# keep_suffix) -- a THRESHOLD, not a k, because k is per sample there and lives in the rows.
IL_SCHEDULES = [("stream", "streaming"), ("il", "interleaved"),
                ("ils", "interleaved_staged"), ("ilu", "interleaved_unified")]
# Full split per dataset. A row whose arms carry fewer unique `i` than this is a reduced-n subset
# and renders parenthesized, unshaded, with no preservation % (the 2026-09-01 probe rule).
IL_FULL_N = {"vstar": 191, "realworldqa": 765, "textvqa": 5000, "infovqa": 2801,
             "visdrone_count": 2350, "visdrone_det": 448,
             "chartqa": 2500, "cvbench": 2638, "mmvp": 300, "refcoco": 8811}
# Latency keys whose hooked FLOPs reference is known-understated: the FLOPs cells render with a
# dagger until the reference is re-measured. Empty since the 2026-09-10 122B re-measure (the
# FP8Experts hook fix; gate F on the ceiling: closed form within 0.05% of full - tower).
IL_FLOPS_PENDING: set = set()


def _pctl(v, q):
    v = sorted(v)
    if not v:
        return None
    i = (len(v) - 1) * q / 100.0
    lo = int(i)
    return v[lo] if lo >= len(v) - 1 else v[lo] + (v[lo + 1] - v[lo]) * (i - lo)


def il_lit(dataset: str, metric: str, slug: str, dir_: str, expected: int,
           suffix: str, thetas=()) -> Dict[str, float]:
    """{floor, ceiling, stream_k*, il_k*, ils_k*, ilu_k*} from one paired served run; complete
    arms only.

    `thetas`: also load the ADAPTIVE arms `{schedule}_g4_auto{theta:g}` under the keys
    `auto_{schedule-key}_{theta:g}`. They go through the same completeness filter and the same
    common-subset intersection as every other arm -- an arm that skipped different rows must not
    be averaged over its own survivors (2026-09-12) -- and their realised k (mean / p95 / min /
    max of the rows' `keep_realised`, over the SAME common subset) lands in `out["_k"][key]`."""
    out: Dict[str, float] = {}
    # `_n`: the smallest unique-i count among the arms that loaded. The 122B rows were first
    # measured on strided subsets and are being re-run at full split dataset by dataset, so
    # whether a row is a probe is a property of the FILES, not of the model -- the emitter
    # parenthesises on this rather than on a per-model flag (2026-09-12).
    out: Dict[str, float] = {}
    tags = [("floor", "floor"), ("ceiling", "ceiling")]
    for k in IL_KEEPS:
        kk = "" if k == 1.0 else f"_k{k:.2f}"
        tags += [(f"stream_k{k:.2f}", f"streaming_g4{kk}"), (f"il_k{k:.2f}", f"interleaved_g4{kk}"),
                 (f"ils_k{k:.2f}", f"interleaved_staged_g4{kk}"),
                 # `ilu`: the unified vision+decoder axis (memo §7.12), 35B V*/RWQA so far
                 (f"ilu_k{k:.2f}", f"interleaved_unified_g4{kk}")]
    for th in thetas:
        tags += [(f"auto_{key}_{th:g}", f"{sched}_g4_auto{th:g}")
                 for key, sched in IL_SCHEDULES]
    # Pass 1: load every arm's scored rows. Arms can skip DIFFERENT rows -- `prompt_too_long` is
    # the same set for all arms of a model, but `oom` is not: on 122B InfoVQA the keep<1
    # interleaved arms OOM on 194-236 long documents that the bounds arms scored. Averaging each
    # arm over its own survivors would then compare different row sets, and since the dropped
    # rows are the long (harder) ones it would flatter exactly the arms that dropped them.
    # Pass 2 therefore scores every arm on the INTERSECTION of all arms' scored rows.
    per_arm: Dict[str, Dict[int, float]] = {}
    per_k: Dict[str, Dict[int, float]] = {}
    for key, tag in tags:
        p = os.path.join(dir_, f"{dataset}{slug}_{tag}{suffix}.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        if len({r["i"] for r in rows}) < expected:
            continue
        sc = {int(r["i"]): float(r[metric]) for r in rows if "skip" not in r}
        if sc:
            per_arm[key] = sc
            n_i = len({r["i"] for r in rows})
            out["_n"] = min(out.get("_n", n_i), n_i)
            kk_ = {int(r["i"]): float(r["keep_realised"]) for r in rows
                   if "skip" not in r and r.get("keep_realised") is not None}
            if kk_:
                per_k[key] = kk_
    if per_arm:
        common = set.intersection(*(set(v) for v in per_arm.values()))
        for key, sc in per_arm.items():
            use = {i: sc[i] for i in common} if common else sc
            out[key] = 100.0 * sum(use.values()) / len(use)
        out["_n_scored"] = len(common)
        ks: Dict[str, Dict[str, float]] = {}
        for key, kk_ in per_k.items():
            v = [kk_[i] for i in (common & set(kk_))] or list(kk_.values())
            ks[key] = {"mean": sum(v) / len(v), "p95": _pctl(v, 95),
                       "min": min(v), "max": max(v), "n": len(v)}
        if ks:
            out["_k"] = ks
    return out


def il_flops(model_row, dataset: str) -> Dict[str, float]:
    """{full, stream_total_k*, stream_crit_k*, il_total_k*, il_crit_k*} in GFLOPs."""
    _, _, _, _, _, fjson, _, _, inproc_key = model_row
    out: Dict[str, float] = {}
    name = fjson.get(dataset)
    p = os.path.join(IL_ROOT, "flops", name) if name else None
    j = json.load(open(p)).get(dataset, {}) if p and os.path.exists(p) else {}
    # streaming half: the same measurement run when it carried the streaming arms (35B),
    # the campaign's inprocess_flops entry otherwise (122B, --il-only).
    if "full" in j:
        out["full"] = j["full"]
        for k in IL_KEEPS:
            kk = "" if k == 1.0 else f"_k{k:.2f}"
            if f"crit_g4{kk}" in j:
                out[f"stream_crit_k{k:.2f}"] = j[f"crit_g4{kk}"]
                out[f"stream_total_k{k:.2f}"] = j[f"total_g4{kk}"]
    else:
        v = inproc_flops(inproc_key, dataset, "full")
        if v is not None:
            out["full"] = v
        for k in IL_KEEPS:
            c, t = inproc_flops(inproc_key, dataset, f"k{k:.2f}"), \
                inproc_flops(inproc_key, dataset, f"total_k{k:.2f}")
            if c is not None:
                out[f"stream_crit_k{k:.2f}"], out[f"stream_total_k{k:.2f}"] = c, t
    # vision half for a decoder-only (--il-only) json: the 35B run's split at the same keep.
    p35 = os.path.join(IL_ROOT, "flops", IL_MODELS[0][5].get(dataset, ""))
    j35 = json.load(open(p35)).get(dataset, {}) if os.path.exists(p35) else {}
    for tag in ("il", "ils", "ilu"):   # unstaged / depth-staged / unified-axis interleaved arm
        for k in IL_KEEPS:
            kk = "" if k == 1.0 else f"_k{k:.2f}"
            d = j.get(f"_{tag}_g4{kk}")
            if not d:
                continue
            vt, vc = d["vision_total"], d["vision_crit"]
            if vt == 0 and j35.get(f"_il_g4{kk}"):
                vt, vc = j35[f"_il_g4{kk}"]["vision_total"], j35[f"_il_g4{kk}"]["vision_crit"]
            out[f"{tag}_total_k{k:.2f}"] = d["decoder_total"] + vt
            out[f"{tag}_crit_k{k:.2f}"] = d["decoder_crit"] + vc
    # unified_staged + adaptive k: the fold keys the arm by its theta tag (`_ilu_g4_auto<theta>`)
    for k, th in adaptive_thetas(model_row[1], dataset).items():
        d = j.get(f"_ilu_g4_auto{th:g}")
        if d:
            out[f"ilu_auto_total_k{k:.2f}"] = d["decoder_total"] + d["vision_total"]
            out[f"ilu_auto_crit_k{k:.2f}"] = d["decoder_crit"] + d["vision_crit"]
    return out


def il_latency(model_row, dataset: str) -> Dict[str, float]:
    """{full, stream_k*, il_k*} Crit. Lat. medians (ms, d=150 anchor = last band's pixels)."""
    _, _, _, _, _, _, il_key, st_key, _ = model_row
    lat = {}
    # the local file wins per model key (the worktree copy of the streaming entries is stale);
    # the interleaved keys exist only in the worktree copy until the branch is merged.
    for root in (IL_ROOT, RESULTS):
        p = os.path.join(root, "latency", "inprocess_latency.json")
        if os.path.exists(p):
            lat.update(json.load(open(p)))
    st = (lat.get(st_key) or {}).get(dataset) or {}
    il = (lat.get(il_key) or {}).get(dataset) or {}
    ils = (lat.get(il_key + "_staged") or {}).get(dataset) or {}
    ilu = (lat.get(il_key + "_unified") or {}).get(dataset) or {}
    out: Dict[str, float] = {}
    if "full" in st:
        out["full"] = st["full"]
    for k in IL_KEEPS:
        if f"k{k:.2f}" in st:
            out[f"stream_k{k:.2f}"] = st[f"k{k:.2f}"]
        if f"k{k:.2f}" in il:
            out[f"il_k{k:.2f}"] = il[f"k{k:.2f}"]
        if f"k{k:.2f}" in ils:
            out[f"ils_k{k:.2f}"] = ils[f"k{k:.2f}"]
        if f"k{k:.2f}" in ilu:
            out[f"ilu_k{k:.2f}"] = ilu[f"k{k:.2f}"]
    for k, th in adaptive_thetas(model_row[1], dataset).items():
        if f"auto{th:g}" in ilu:                       # latency_probe.py --keeps auto:<theta>
            out[f"ilu_auto_k{k:.2f}"] = ilu[f"auto{th:g}"]
            p95 = ilu.get(f"auto{th:g}_ttft_last_band_p95_ms")
            if p95 is not None:
                out[f"ilu_auto_p95_k{k:.2f}"] = p95
    return out


def emit_interleaved_latex() -> str:
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\vspace{-0.1in}")
    # The long measurement caption moved to docs/memo/interleaved_table_notes.md (2026-09-11):
    # this table is a working view, so the caption only names the arms and the columns.
    L.append(r"\caption{Streaming vs.\ interleaved LLM correction, served Qwen3.5 (vLLM, one "
             r"B200; FP8 for 122B). Streaming: LLM prefills each band's final rows once. "
             r"Interleaved: LLM prefills the base-resolution prompt, then re-runs each band's "
             r"corrected rows (full depth). Depth-staged: round $r$ corrects over the first $b_r$ "
             r"decoder layers and carries every row through $[b_r, b_{r+1})$. Unified: one depth "
             r"axis of tower $+$ decoder stages split by equal cost; $k{<}1$ selects by the "
             r"prefix layer-mean attention (progressive). Acc.\ (preservation vs.\ Full-res.); "
             r"Comp.\ / Crit.\ Comp.: total / last-round FLOPs (share of the full-resolution "
             r"pass); Crit.\ Lat.: TTFT from the last band's arrival, bands 150\,ms apart, "
             r"median of 36 (share of the full-res TTFT). All arms of a model from one served "
             r"engine, paired rows. Every cell is the full split unless PARENTHESIZED, which "
             r"marks a reduced-$n$ strided subset (rendered unshaded and without a preservation "
             r"\%; the per-row flag compares each row's own $n$ against its full split). VSR is "
             r"deliberately absent: its floor and ceiling are indistinguishable on both models "
             r"(paired $p = 0.27$ / $0.40$), so it carries no signal about the schedule. "
             r"Unified $+$ adaptive: the unified schedule with a per-band pscore THRESHOLD "
             r"$\theta$ instead of a fixed budget (score $=$ RMS residual in raw pixel units "
             r"$\times$ $N\cdot$received attention; the count is ceilinged onto 1/8 buckets), "
             r"$\theta$ calibrated per dataset so the mean realised $k$ matches the row's $k$; "
             r"the realised mean $\bar k$ is printed under the accuracy. "
             r"Notes: docs/memo/interleaved\_table\_notes.md.}")
    L.append(r"\label{tab:interleaved_results}")
    L.append(r"\vspace{0.05in}")
    L.append(r"\centering")
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\begin{tabular}{l c | c c c c | c c c c | c c | c c c | c c c c}")
    L.append(r"\toprule")
    L.append(r" & & \multicolumn{4}{c|}{Streaming (LLM prefills once)} & "
             r"\multicolumn{4}{c|}{Interleaved (LLM re-corrects per band)} & "
             r"\multicolumn{2}{c|}{Interleaved, depth-staged} & "
             r"\multicolumn{3}{c|}{Unified (tower $+$ decoder)} & "
             r"\multicolumn{4}{c}{Unified $+$ adaptive $k$ (threshold $\theta$)} \\")
    L.append(r"Dataset & $k$ & Acc.\ (\%) & Comp. & Crit.\ Comp. & Crit.\ Lat. & "
             r"Acc.\ (\%) & Comp. & Crit.\ Comp. & Crit.\ Lat. & Acc.\ (\%) & Comp. & "
             r"Acc.\ (\%) & Comp. & Crit.\ Lat. & "
             r"Acc.\ (\%) & Comp. & Crit.\ Comp. & Crit.\ Lat. \\")
    for model_row in IL_MODELS:
        disp, slug, suffix, expected, probe_model, _, lat_key, _, _ = model_row
        probe = probe_model
        dag = r"$^\dagger$" if lat_key in IL_FLOPS_PENDING else ""
        L.append(r"\midrule")
        L.append(r"\multicolumn{19}{l}{\emph{" + disp + r"}} \\")
        first_ds = True
        for ds, label, metric, sub in IL_DATASETS:
            if ds not in expected:
                continue
            if not first_ds:          # a light rule between datasets (the model rows use \midrule)
                L.append(r"\cmidrule(lr){1-19}")
            first_ds = False
            thetas = adaptive_thetas(slug, ds)
            lit = il_lit(ds, metric, slug, os.path.join(IL_ROOT, sub), expected[ds], suffix,
                         thetas=tuple(thetas.values()))
            ks_auto = lit.pop("_k", {})
            # per-ROW probe flag: reduced n renders parenthesized even when the model's other
            # rows are full-split (and vice versa), which is what the 122B re-run needs
            # Parenthesise on SCORED coverage, not on the file's row count: a row can hold every
            # index and still be a biased subsample of the split if arms skipped rows
            # (122B InfoVQA scores 2227 of 2801 -- 338 over-length + up to 236 OOM -- and the
            # dropped rows are the longest, so floor moves -1.44 pp when they go).
            n_file = lit.pop("_n", 0); n_scored = lit.pop("_n_scored", n_file)
            probe = min(n_file, n_scored) < 0.95 * IL_FULL_N.get(ds, 0) if lit else probe_model
            fl = il_flops(model_row, ds)
            lat = il_latency(model_row, ds)
            ceil = lit.get("ceiling")
            full_gf, full_ms = fl.get("full"), lat.get("full")

            def acc(v):
                if v is None:
                    return "--"
                if probe:
                    return f"({v:.2f})"
                s = f"{v:.2f}"
                if ceil:
                    pres = 100.0 * v / ceil
                    s += f" ({pres:.1f}\\%)"
                    if pres >= SHADE_PRES:
                        s = SHADE_MACRO + s
                return s
            bounds = []
            if "floor" in lit or ceil is not None:
                fv = lit.get("floor")
                bounds.append((f"({fv:.2f})" if probe else f"{fv:.2f}") if fv is not None else "--")
                bounds.append((f"({ceil:.2f})" if probe else f"{ceil:.2f}") if ceil is not None else "--")
            # dataset cell: the name on the k=1 row, the references the percentages are taken
            # against on the two rows below it (floor / ceiling accuracy; full-res FLOPs + TTFT)
            sub = []
            if bounds:
                sub.append(r"{\footnotesize Low-res.\ " + bounds[0] + r" / Full-res.\ " + bounds[1] + "}")
            ref = [x for x in ((fmt_tf(full_gf, None) + dag) if full_gf else "",
                               fmt_ms(full_ms, None) if full_ms else "") if x]
            if ref:
                sub.append(r"{\footnotesize Full-res.\ " + ", ".join(ref) + "}")
            heads = [label] + sub + [""] * len(IL_KEEPS)
            for i, k in enumerate(IL_KEEPS):
                kk = f"k{k:.2f}"
                cells = [heads[i], f"{k:.2f}",
                         acc(lit.get(f"stream_{kk}")),
                         fmt_tf(fl.get(f"stream_total_{kk}"), full_gf) + dag,
                         fmt_tf(fl.get(f"stream_crit_{kk}"), full_gf) + dag,
                         fmt_ms(lat.get(f"stream_{kk}"), full_ms),
                         acc(lit.get(f"il_{kk}")),
                         fmt_tf(fl.get(f"il_total_{kk}"), full_gf) + dag,
                         fmt_tf(fl.get(f"il_crit_{kk}"), full_gf) + dag,
                         fmt_ms(lat.get(f"il_{kk}"), full_ms),
                         acc(lit.get(f"ils_{kk}")),
                         fmt_tf(fl.get(f"ils_total_{kk}"), full_gf) + dag,
                         acc(lit.get(f"ilu_{kk}")),
                         fmt_tf(fl.get(f"ilu_total_{kk}"), full_gf) + dag,
                         fmt_ms(lat.get(f"ilu_{kk}"), full_ms)]
                # unified + adaptive: the k=1 row has no threshold (auto at theta->0 IS k=1)
                th = thetas.get(k)
                if k < 1.0 and th is not None:
                    key = f"auto_ilu_{th:g}"
                    a_cell = acc(lit.get(key))
                    kr = ks_auto.get(key)
                    if kr and lit.get(key) is not None:
                        a_cell += r" {\footnotesize ($\bar k{=}" + f"{kr['mean']:.2f}" + r"$)}"
                    cells += [a_cell,
                              fmt_tf(fl.get(f"ilu_auto_total_{kk}"), full_gf) + dag,
                              fmt_tf(fl.get(f"ilu_auto_crit_{kk}"), full_gf) + dag,
                              fmt_ms(lat.get(f"ilu_auto_{kk}"), full_ms)]
                else:
                    cells += ["--"] * 4
                L.append(" & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{table*}")
    return "\n".join(L)


def emit_adaptive_md(thetas, groups: int) -> str:
    """Adaptive-k ("--keep auto") rows next to the fixed-k arms they are meant to replace.

    Its own table, in markdown: an adaptive arm has no fixed k to sit under, and what it has
    instead -- the realised-k distribution -- is two columns the LaTeX table has no room for.
    Accuracies are on the common subset of every arm that loaded for that dataset (il_lit)."""
    L = [f"| model | dataset | arm | theta | n | Acc. (%) | mean k | p95 k | min | max |",
         "|" + "---|" * 10]
    for model_row in IL_MODELS:
        disp, slug, suffix, expected, _, _, _, _, _ = model_row
        for ds, label, metric, sub in IL_DATASETS:
            if ds not in expected:
                continue
            lit = il_lit(ds, metric, slug, os.path.join(IL_ROOT, sub), expected[ds], suffix,
                         thetas=thetas)
            ks = lit.get("_k", {})
            ref = [("floor", "low-res.", None), ("ceiling", "full-res.", None)] + \
                  [(f"{k_}_k{kp:.2f}", f"{sch} k={kp:.2f}", kp)
                   for k_, sch in IL_SCHEDULES for kp in IL_KEEPS]
            rows = []
            for key, name, kp in ref:
                if key in lit:
                    rows.append((name, "--", lit[key], kp, kp, kp, kp))
            for th in thetas:
                for key, sched in IL_SCHEDULES:
                    k_ = f"auto_{key}_{th:g}"
                    if k_ not in lit:
                        continue
                    d = ks.get(k_, {})
                    rows.append((f"{sched} auto", f"{th:g}", lit[k_], d.get("mean"),
                                 d.get("p95"), d.get("min"), d.get("max")))
            if not any(r[1] != "--" for r in rows):
                continue                      # no adaptive row for this cell yet
            n_s = lit.get("_n_scored", lit.get("_n", 0))
            for name, th_, acc_, m_, p_, lo_, hi_ in rows:
                f = lambda v: "--" if v is None else f"{v:.3f}"
                L.append(f"| {disp} | {label} | {name} | {th_} | {n_s} | "
                         f"{'--' if acc_ is None else f'{acc_:.2f}'} | "
                         f"{f(m_)} | {f(p_)} | {f(lo_)} | {f(hi_)} |")
    if len(L) == 2:
        L.append("| _no adaptive rows found_ | | | | | | | | | |")
    return "\n".join(L)


def emit_status(keeps, groups: int) -> str:
    L, missing, total = [], 0, 0
    for model, rows in SPEC:
        for label, acc_key, fl_key in rows:
            if acc_key is None:
                continue
            for tag in ["floor", "ceiling"] + [f"interleaved_g{groups}_k{k:.2f}" for k in keeps]:
                total += 1
                if load_accuracy(*acc_key, tag) is None:
                    missing += 1
                    L.append(f"  MISSING  {acc_key[0]}/{acc_key[1]}/{tag}")
    L.append(f"\n  accuracy cells: {total - missing}/{total} present")
    return "\n".join(L)


def emit_overhead(keeps) -> str:
    hdr = ["model", "dataset", "full (TF)"]
    for k in keeps:
        hdr += [f"total{int(k*100)} (TF)", f"x full"]
    L = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for model, rows in SPEC:
        for label, _, fl_key in rows:
            full = get_flops(fl_key, "full")
            cells = [model, label, f"{full/1000:.2f}" if full else "--"]
            for k in keeps:
                t = get_total(fl_key, f"k{k:.2f}")
                cells += [f"{t/1000:.2f}" if t else "--",
                          f"{t/full:.2f}x" if (t and full) else "--"]
            L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keeps", type=float, nargs="+", default=[0.25, 0.50])
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--format", choices=["latex", "md"], default="latex")
    ap.add_argument("--thetas", type=float, nargs="*", default=[],
                    help="--table adaptive: the `--keep auto` thresholds to look for "
                         "(row files `..._{schedule}_g{groups}_auto{theta:g}.jsonl`)")
    ap.add_argument("--table", choices=["eval", "latency", "interleaved", "adaptive"],
                    default="eval",
                    help="eval: accuracy + compute (eval_table_*.tex); latency: the served-path "
                         "latency table (latency_table_*.tex); interleaved: streaming vs "
                         "interleaved LLM correction on Qwen3.5 (interleaved_table_*.tex)")
    ap.add_argument("--with-latency", action="store_true",
                    help="eval table in the combined form (Lat. / Crit. Lat. columns inline)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--overhead", action="store_true",
                    help="report TOTAL compute (approx + all corrections) against the ceiling, "
                         "i.e. what the schedule costs rather than what it defers")
    a = ap.parse_args()
    if a.status:
        print(emit_status(a.keeps, a.groups))
        return
    if a.overhead:
        print(emit_overhead(a.keeps))
        return
    if a.table == "latency":
        print(emit_latency_latex(a.keeps))
        return
    if a.table == "interleaved":
        print(emit_interleaved_latex())
        return
    if a.table == "adaptive":
        print(emit_adaptive_md(a.thetas, a.groups))
        return
    table = build_rows(a.keeps, a.groups, latency=a.with_latency)
    print(emit_latex(table, a.keeps, a.with_latency) if a.format == "latex"
          else emit_md(table, a.keeps, a.with_latency))


if __name__ == "__main__":
    main()
