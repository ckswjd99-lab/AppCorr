# Main eval table -- measurement notes (moved out of the tex caption 2026-09-11)

The tex caption of `eval_table_*.tex` now carries only the column definitions and a one-line legend of the cell markers; the full text it used to carry is below, verbatim. Update this file when a convention changes.

## Long caption (verbatim)

\caption{Evaluation Results across Different Configurations. Crit.\ Comp.\ is backbone prefill FLOPs per instruction that can only begin once the whole image has arrived (decode excluded); Comp.\ is the arm's total backbone compute including work overlapped with transmission. Parentheses give the ratio to the Full-res.\ computation. Ours uses interleaved $g{=}4$; shaded Ours and Streaming accuracy cells retain $\geq$98\% of the Full-res.\ accuracy. The Streaming (k$=$1.0) block is the causal-LLM category: the LLM prefills exactly once in arrival-order chunks while the vision encoder corrects everything progressively -- total $\approx$ full + one vision pass, critical $\approx 1/g$. Only causal models qualify (Gemma 3's image tokens are bidirectional; VFMs have no LLM). $^\dagger$Qwen3.5's Ours columns are keep-limited STREAMING arms (band-wise top-$k$ selection), not interleaved correction. $^\ddagger$Gemma 3, LLaVA-OV2 and Mistral compute figures are from the progressive per-round selection arm (2026-08-26); accuracy cells are the progressive arm's where re-measured (2026-08-28 sweep; Mistral: the 50\% cells of RefCOCO, TextVQA and VisDrone) and the earlier upfront (Mistral: one-shot corrected) arm's otherwise. $^\S$Qwen2.5 ours ran at batch size 1 against batch-16 bounds (measured equivalent); the 25\% arm excludes 8/8811 images (0.09\%, a since-fixed driver defect) with bounds restricted to the same kept set -- the 50\% arm has full coverage. \textsuperscript{\P}122B-FP8 runs under the Triton finegrained-fp8 fallback (the DeepGEMM path mis-generates on sm\_100). Its PARENTHESIZED, unshaded accuracy cells are a reduced-scale probe (n$=$240 per arm) and must not be compared 1:1 against full-split cells; unparenthesized 122B cells are full-split values (RealWorldQA, VisDrone Count, V*Bench in-process; VisDrone Det, TextVQA, RefCOCO and ChartQA through a vLLM 0.28 FP8 engine fed by the same vision tower, whose prediction-level agreement with the in-process path is 96--99\% on all arms alike). Compute figures are shape-determined and kernel-independent. $\diamond$: the model is not capable of the task itself (floor $\approx$ ceiling at degenerate accuracy), so these rows carry no signal about the method. $^\ast$OpenVLA Streaming corrects cumulatively (every patch received so far is re-corrected each round). A cheaper new-only variant (each round corrects only the patches that just arrived; total 113\%, critical 32\%) ties the ceiling on Object/Goal (85.6/74.6) but loses on Spatial (77.2, $-$4.6pp) and Long (41.0, $-$10.8pp vs.\ Full-res.; 500 episodes each, paired 95\% CIs exclude 0), so the cumulative form is the one reported.}

## Which tree generates the table

See `interleaved_table_notes.md` ("Which tree generates which table"): the main table is generated from AppCorr-qwen35-eval.

## 2026-09-11: degrade-filter convention fixed, 35B bounds re-measured

The 35B bounds (floor / ceiling / streaming k=1) in `qwen35_accuracy/` were measured 2026-08-31 on
B200-6 with the **bicubic** degrade filter, while every arm from 2026-09-09 on -- including all the
keep<1 "Ours" cells -- used **box**. Rows therefore compared a bicubic floor against box Ours cells.

Verified rather than inferred: re-running chartqa floor under bicubic on the new box gives **60.80**
against the archived **60.76** (105 discordant rows, 53/52, p = 1.00), while box gives **46.64**. So
the 08-31 files were bicubic and the ChartQA floor difference (-14.16 pp) is entirely the filter.
Controls: `ceiling` never consumes the degraded base and moved +0.16 pp (1.1% discordance);
TextVQA, which used pyr on BOTH sides, moved +0.24 pp (2.0%) -- an 11-fold difference in floor
discordance between the dataset where the filter was held fixed and the one where it was not.
Charts break because the answer IS the small text in the image; natural-image sets move <1 pp.

**Convention (user decision, 2026-09-11):** box for realworldqa / chartqa / mmvp / vsr / cvbench;
pyr for textvqa / vstar / visdrone_count / visdrone_det / refcoco / infovqa. Per-dataset-family by
decision, not by accident. Never pair arms measured under different filters.

All 18 bounds arms were re-measured 2026-09-11 on B200-8 with current code (box for the five, pyr
for textvqa) so every 35B row now shares one box, one code version and one filter. The superseded
files are archived at `/NHNHOME/share/cjpark/backup/qwen35_bounds_pre_rebounds_20260911/` with a
provenance README. Significance changes: MMVP k0.50 vs floor p 0.332 -> 0.031; MMVP k0.25 no longer
below floor (that anomaly was the artefact); RealWorldQA k0.50 / k0.25 p 0.098 / 0.070 -> 0.033 /
0.005; CV-Bench k0.50 p 0.037 -> 0.013; VSR unchanged and still null in both directions -- which is
what shows the procedure removed an error rather than manufactured significance.

**Required clause whenever ChartQA's gain is quoted** (+24.44 -> +38.56 pp vs floor at k=0.50): the
baseline's filter changed bicubic -> box, ceiling and streaming are unchanged, and the method did
nothing different.

### RefCOCO 35B is the one row that is not internally consistent

Its Ours cells (k0.50 Acc@0.5 91.6468 / mIoU 83.4310; k0.25 91.0907 / 82.6642) were measured
2026-09-12 on B200-8; its floor / ceiling / streaming k=1 are 2026-08-31 files from B200-6 and were
NOT part of the bounds re-measurement. The filter matches on both sides (pyr), so this row escapes
the bicubic/box problem entirely; what it carries is the cross-box term, which on the pyr precedent
(TextVQA, pyr on both sides) was +0.24 pp on floor at 2.0% discordance, p=0.111.

Deliberately not fixed: a bounds triplet is 8811 x 3 arms at the measured 2.86 h per arm, ~4.3 h,
for a row whose span is 2.21 pp on Acc@0.5 -- i.e. likely to show nothing even once matched. The
two Ours cells ARE mutually comparable (same box, same filter, same code) and separate cleanly:
k0.25 vs k0.50 is -0.556 pp on Acc@0.5, 131/180 over 311 discordant, p = 0.0064. Read them against
each other, not against the 08-31 bounds.

## Scorer and budget defects found 2026-09-13 (apply after the GLM-4.6V campaign; re-score from stored preds)
1. MCQ scorers take the first A-D letter (V*Bench/SOU), `\b([ab])\b` (MMVP) or `\b([A-Za-z])\b`
   (CV-Bench/RWQA) -- all award points to English prose. Replaced by an anchored extractor
   (`b200-8_logs/mcq_extract.py`: sentinel / whole-string letter / letter+delimiter+text /
   "answer is X" / last-line letter / "(X)"; else no_answer -> 0). Regression over all 11 trees:
   358 of 160,304 rows change (0.22%), all 1 -> 0 on rows with no answer; 262 of them are the
   4B V*Bench row below.
2. CV-Bench loader strips gold to A-D; 86 rows (Count task, 5-6 options, answers (E)/(F)) were
   scored as gold "" -> every model's CV-Bench accuracy is ~2.0 pp low (ceilings +1.97..+2.05
   once fixed; paired contrasts unchanged). Fix: choices "ABCDEF" in gold extraction and scorer.
3. **4B V*Bench row is invalid**: 58-78% no_answer (prose cut at the 24-token in-process cap),
   arm-dependent, so the contrast is contaminated; re-run at a measured budget on B200-8.
The Qwen served rows are otherwise safe from truncation: `finish_reason == "length"` rate is
<= 0.16% everywhere except 35B V*Bench 1.68% (letter already emitted).
