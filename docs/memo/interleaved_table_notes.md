# Interleaved / unified table -- measurement notes (moved out of the tex caption 2026-09-11)

The tex caption of `interleaved_table_*.tex` is deliberately short (the table is a working
view, not paper text). Everything the long caption used to say lives here, verbatim as of the
2026-09-11 11:30 state, so the numbers quoted are those of that regeneration; the live cells are
in the tex. Update this file when a convention changes, not the caption.

## Long caption (verbatim)

\caption{Streaming vs.\ interleaved LLM correction on the served Qwen3.5 models (vLLM 0.28 engine, one B200, FP8 for 122B). \emph{Streaming}: the vision tower is corrected progressively and the LLM prefills each band's final rows exactly once (chunked causal prefill; the main-table Qwen3.5 rows). \emph{Interleaved}: the LLM prefills the whole base-resolution prompt once (approximate pass) and, per band, re-runs only the rows the tower corrected in that band -- softmax layers as 1-token pseudo-sequences over the request's KV blocks, Gated DeltaNet layers by a side-buffer re-scan from the band's checkpoint -- plus the trailing text in the last round; every round runs at full depth, so Comp.\,$= 1 + k\,f_{\mathrm{img}} + f_{\mathrm{text}}$ of a full-resolution pass while Crit.\ Comp.\ is the last round alone ($\approx k f_{\mathrm{img}}/g + f_{\mathrm{text}}$). \emph{Depth-staged} (\S3.3): round $r$ corrects its rows over the first $b_r = L(r{+}1)/g$ decoder layers only and every image row is then carried through layers $[b_r, b_{r+1})$ with the corrected context -- the approximate pass is spread over the rounds, so Comp.\,$= 1 + \bar b\,k\,f_{\mathrm{img}} + f_{\mathrm{text}}$ with $\bar b = 0.625$ at $g{=}4$, and the not-yet-corrected rows see the corrected rows of earlier bands in their deep layers (its $k{<}1$ state therefore differs from the unstaged form -- KV rel.\ L2 0.43--0.47 between the two on the gate images -- but its measured accuracy does not: paired against the unstaged arm, $-0.5$/$-1.6$/$-1.0$\,pp on V*Bench and $+0.3$/$-0.1$/$0.0$\,pp on RealWorldQA at $k{=}1$/$0.5$/$0.25$, every bootstrap CI covering 0; at $k{=}1$ the two are identical up to run-to-run noise). Its Crit.\ Comp.\ is that of the unstaged form (the last round is full depth in both) and its measured Crit.\ Lat.\ matches the unstaged rows within 1.3\,ms on every cell (62/49/36\,ms V*Bench, 34/28/25\,ms RealWorldQA vs.\ 62/50/36 and 34/28/24; the last round also carries the previous band's rows through the last layer stage, which costs nothing visible), so the unstaged column stands for both; the accuracy column is measured on the served engine running the staged schedule (its rounds walk the frontier over the stock full-depth approximate prefill, so the measured work is not the ideal Comp., the state is) and Comp.\ is the closed form of the ideal schedule replayed over the same rows. Both arms, their bounds and the low-/full-resolution references come from ONE served engine per model (paired rows, same driver, same vision tower, greedy; concurrency 4 on 35B, 2 on 122B); the vision tower's correction is identical between the two arms (bitwise on the pushed embeddings) so every difference is the LLM schedule. Acc.: accuracy (preservation vs.\ Full-res.); Comp.\ / Crit.\ Comp.: total / last-round FLOPs per instruction (share of the full-resolution pass). Comp.\ is vision tower $+$ decoder: the vision half -- approximate pass on the full token grid plus the $k$ correction, $1 + 0.95k$ of the tower, 65\% (V*Bench) / 46\% (RealWorldQA) of the full pass -- is common to all three arms and is what puts Streaming at 130\% on V*Bench at $k{=}0.5$; the decoder half is $1\times$ the stock prefill for Streaming and the closed forms above for the interleaved arms ($1.49\times$ / $1.31\times$ at $k{=}0.5$ on V*Bench, unstaged / staged; $1.51$ / $1.34$ on RealWorldQA). Both halves are per-instruction means over the same 12 hooked samples: the vision half and the stock prefill are hooked on the streaming arm, and the interleaved decoder is that hooked prefill scaled by the closed-form ratio of \texttt{flops\_analytic.Qwen35Decoder} replayed over every accuracy row's actual correction windows (the ratio moves by $<0.003$ between the 12 and the full row set); Crit.\ Lat.: single-request TTFT from the last band's pixel arrival with the bands spaced 150\,ms (as in Table~\ref{tab:latency_results}; share of the full-resolution TTFT), medians over 36 images. The interleaved correct step replays vLLM's piecewise CUDA graphs ($|P|$ padded to the capture size; gated bitwise-equal to the padded eager step on 4B and 35B) with its DeltaNet re-scan free of host syncs (a constant $-11$ to $-13$\,ms per step on every cell, 32/32 paired samples), and costs 15--25\,ms per round on 35B (14.7\,ms at $|P|{\approx}60$ to 25.2\,ms at $|P|{=}835$): the step is GPU-bound with a ${\approx}9$\,ms floor from streaming the full MoE expert weights once per layer (every expert is touched at $|P| \ge 100$), so Crit.\ Lat.\ falls with $k$ more slowly than Crit.\ Comp. The last round is fused with the first decode step (row $N{-}1$ is computed inside the correct step and the sampler reads its hidden state directly, no separate hold-back forward; gated in the same band; $-1.5$ to $-4.5$\,ms on 35B, $-3.3$ to $-5.8$\,ms on 122B, paired), which leaves ${\approx}4$\,ms of engine step and ${\approx}2$\,ms of reply hop around it. 122B: 40-row V*Bench probe, parenthesized; its interleaved server caps the correct step at 512 pseudo-sequences (1024 exceeds the Mamba cache blocks at this memory budget; the 35B one: 1024), so a round is $\lceil |P|/512 \rceil$ sub-batches of 26--37\,ms and at $k{=}1$ on V*Bench the 835-row last band costs two (76\,ms with the fused step; the earlier 256-cap eager server measured 742\,ms, 398 of it queueing). \emph{Unified} (\S3.3, memo \S7.12): one depth axis of $27$ vision-tower stages $+$ $L$ decoder layers, cut into $g$ rounds of equal cumulative cost (35B on V*Bench: tower depths 11/22/27/27 and decoder depths 0/0/13/40 at the four rounds), so the tower's approximate pass is itself spread over the rounds and the LLM opens only when the frontier crosses the projector; $k{<}1$ selects each band by the layer-mean attention of the stages walked so far (progressive), not by the full-tower score of the other arms, so its $k{<}1$ cells are not a schedule-only comparison. Its Crit.\ Comp.\ equals the depth-staged one by construction (the last round walks the full remaining depth of both halves) and is not repeated; paired against the depth-staged arm its accuracy is $-0.5$/$-1.0$/$0.0$\,pp on V*Bench and $+0.5$/$+0.3$/$+0.9$\,pp on RealWorldQA at $k{=}1$/$0.5$/$0.25$ (every bootstrap CI covering 0; the six cells were served three times -- twice on the engine before the changes below, bit-identical per sample, and once after them, 1--4 samples per cell moving each way). For the unified arm the served engine skips its stock full-depth prefill: the opening push carries $b_0$ and the engine walks the prompt rows through decoder layers $[0, b_0)$ from the pushed embeddings as soon as the request is scheduled (the same catch-up walk the first staged round used to run inside itself), so the served work is the closed form's and the first LLM round carries no full prefill; every contiguous-row step of the engine (frontier walks, keep=1 band corrections) runs as one prefill sequence over the request's KV blocks instead of one-token pseudo-sequences (V*Bench: the 3.3\,k-row walk 132 to 43\,ms), which also serves the other two interleaved arms (their Crit.\ Lat.\ was re-probed on this engine; the change is $-1$ to $-4$\,ms). With that its Crit.\ Lat., anchored at the last band's arrival like the others, equals the staged arm's on every cell (V*Bench 62/49/36 vs.\ 62/49/36\,ms; RealWorldQA 34/28/24 vs.\ 34/28/25) -- equal critical compute by construction, so it cannot fall below it -- while end-to-end the unified request finishes 100--210\,ms earlier on V*Bench and 140--160\,ms on RealWorldQA (the tower's approximate work overlaps the band gaps). Under each dataset: the Low-res.\ / Full-res.\ accuracies and the full-resolution FLOPs and TTFT every percentage in its rows is taken against.}

## Pointers

- Schedules: memo `vllm_interleaved_design.md` §7.11 (staged), §7.12 (unified), §7.13 (measured
  unified results + the three engine changes: post-crossing `correct_rows`, `open_walk`,
  `CONTIG_ROWS`).
- Rows: il-engine `analysis/results/qwen_vllm_accuracy_il{,_pyr}/`; earlier unified runs in
  `ilu_run1/`, `ilu_run2/`.
- FLOPs: `flops/qwen35_flops_il.json` (V*), `_il_rwqa.json`, `_il_ext.json` (TextVQA / InfoVQA /
  VisDrone), 122B: `qwen35_122b_flops_il.json`, `_fixed.json`, `_infovqa.json`.
- Latency: `latency/inprocess_latency.json` keys `qwen35_35b_il{,_staged,_unified}` (canonical =
  the open-walk + contiguous-row engine; `*_v1_pseudo` = before), 122B `qwen35_122b_il{,_unified}`.

## Which tree generates which table (SUPERSEDED 2026-09-14)

**The interleaved table is now generated from AppCorr-adaptive**, not AppCorr-il-engine. That
worktree is the consolidated branch (develop/adaptive-keep): it holds the GLM engine + ports, the
adaptive columns, and the only `latency/inprocess_latency.json` carrying every GLM-5.3 key
(_tp2graph, _tp2graph_fixed = the table's source, _tp2graph_lat, _tp2graph_rerun5871, _tp4graph).
Every `interleaved_table_20260914_*.tex` in docs/tables came from it.

Keep the two trees' `analysis/results/` in step by hand when a run writes to only one of them:
they are separate copies, not links. This bit us on 2026-09-14 -- the GLM-5.3 VisDrone Det floor
was re-scored for the pixel-frame fix in the adaptive tree while il-engine kept the 0.22
bad-frame copy, and the note below would have sent a reader to the stale one. Both now carry the
re-scored file with the original parked under `_badframe_20260914/`.

## Which tree generates which table (2026-09-11)

- `interleaved_table_*.tex`: generate from **AppCorr-il-engine** (its `latency/inprocess_latency.json`
  is the live one with the streaming k<1 probes; the il-unified copy is stale for streaming keys).
- `eval_table_*.tex` (main table): generate from **AppCorr-qwen35-eval** -- it holds the 122B
  full-split in-process rows (`qwen35_accuracy/realworldqa_*122b*` 765 rows, `qwen35_accuracy_pyr/
  visdrone_count_*122b*` 2350 rows) that AppCorr-vllm only has as 50-row probes. The 35B progressive
  keep<1 rows (chain 2b, 2026-09-09/10) were produced in AppCorr-vllm's `qwen35_accuracy_pyr/` and
  copied into AppCorr-qwen35-eval on 2026-09-11 (7 files); AppCorr-vllm's old eager-pscore rows sit
  in `_eager_pscore_20260909/`. TextVQA k0.50 is incomplete (2900/5000, chain 2b killed) -> `--`.
- `qwen35_pyr_lit` falls back to the no-suffix file name when the "_c4" one is absent (122B
  RealWorldQA bounds / k=1 were written before the suffix convention).

## Extension to 10 datasets (2026-09-12)

The interleaved table originally carried the six datasets the schedule work was developed on
(V*Bench, RealWorldQA, TextVQA, InfoVQA, VisDrone Det/Count). Four main-table datasets were
missing; all four were added at full split on 35B, 14 arms each (bounds + 4 schedules x 3 keeps),
one served engine per dataset, `APPCORR_CONTIG_ROWS=1` + open-walk engine, box filter except
RefCOCO (pyr), per the standing filter convention.

Verified on disk before tabulating (row count == split, `len(unique i) == rows`, accuracy
recomputed from the jsonl): 56/56 arms complete.

Bounds and the schedule verdict, common subset, sign test on `ok`:

| dataset | n | floor | ceiling | span | span p | significant cells |
|---|---:|---:|---:|---:|---|---|
| MMVP | 300 | 76.00 | 81.67 | +5.67 | 0.0015 | 0 of 9 |
| ChartQA | 2500 | 47.52 | 89.08 | +41.56 | 1.2e-271 | 1 (unified k=0.50, **+0.96 pp**, 58/34, p=0.016) |
| CV-Bench | 2638 | 83.93 | 85.22 | +1.29 | 0.0051 | 1 (unified k=0.25, **-0.42 pp**, n=23, p=0.0347) |
| RefCOCO | 8811 | 91.19 | 93.24 | +2.04 | 5.0e-16 | 0 of 9, despite 154-197 discordant pairs |

The two significant cells have OPPOSITE sign and sit on different keeps, which is what a null with
81 cells looks like. RefCOCO is the strongest single piece of evidence for schedule equivalence:
the largest split in the campaign, a real floor-ceiling span, hundreds of discordant pairs per
cell, and not one cell separating from streaming.

PR-2 running total after the extension: **81 cells, 5 significant, 1 of them unstaged** -- exact
binomial p=1.00 against the pre-registered null of 1/3. PR-1 closed as no effect.

### Gaps that are deliberate, not missing work

- **122B RefCOCO is a 240-row subset**, not the 8811-row split (a full 122B RefCOCO sweep is
  14 arms x 8811 rows on the slowest model in the campaign). The per-row probe flag fires on it
  automatically (`min(n_file, n_scored) < 0.95 * IL_FULL_N[ds]`), so the cell renders parenthesized.
- **122B CV-Bench and MMVP FLOPs cells are EMPTY.** No hooked FLOPs base exists for those two
  datasets in any of the eight 122B FLOPs files, and `flops_report_qwen35.py --il-only` folds
  decoder-only: it would emit `vision_total 0.0` and exit 0, i.e. a silently wrong number rather
  than a missing one. Decision 2026-09-12: leave empty until a hooked run exists (queued on B200-8
  behind the 4B block), then fold and check the vision half against the other 122B datasets at a
  comparable token count before writing it in.

### Two fold bugs worth not repeating

1. A hooked FLOPs run needs a real GPU. `CUDA_VISIBLE_DEVICES=""` gives `No CUDA GPUs are
   available`; only `--il-only` re-folds are CPU-only.
2. `--il-rows` must point at the directory the arm was actually written to. Pointing all four
   extension datasets at the box dir made RefCOCO (which lives in `_il_pyr`) silently produce
   3 arms and no il/ils/ilu keys, exit code 0. Fold RefCOCO separately.

## Why VSR is not in this table (re-verified 2026-09-12)

The interleaved table covers 10 datasets; the main table's 10 differ by one on each side (this
table has InfoVQA, the main table has VSR). VSR is excluded because it is **saturated on both
served models**: 35B floor 88.5434 vs ceiling 89.3617, 66 discordant pairs splitting 38/28, sign
test p=0.268 (recomputed from `qwen35_accuracy/vsr_{floor,ceiling}.jsonl` before the decision was
re-taken); 122B p=0.403. Every schedule arm would sit between two statistically identical bounds,
so the row cannot separate schedules under any outcome.

This is a property of VSR *at that model size*, not of VSR: on Qwen3.5-4B the span is real
(+1.72 pp, p=0.035) and streaming at k=1 recovers none of it (68 discordant, 34/34 -- the only
k=1 row in any model that lands exactly on floor). So VSR saturates somewhere between 4B and 35B,
and the 4B row belongs in the MAIN table as a genuine zero-recovery result.

## Paired counts go on the scored field, never on `pred`

4B MMVP, k=1 vs ceiling: 17 discordant on `ok`, **42 on `pred`**. The extra 25 are verbosity
differences ('(b)' vs '(b) Away from the camera') that MMVP's letter-extracting scorer never sees.
Compute every discordant count on the field the metric reads -- `ok` for exact match / multiple
choice / IoU threshold, `val` for TextVQA (VQA soft score) and InfoVQA (ANLS). `pred` is for
eyeballing what changed after the counts exist.

## Defect found in `qwen35_122b_cg1024` (2026-09-12) -- two server eras in one key

The streaming latency key the 122B rows take their `full` TTFT and Crit. Lat. from contains entries
measured under two different servers:

| entries | measured | server |
|---|---|---|
| chartqa, refcoco, textvqa, visdrone_count, visdrone_det, vstar (13 keys each) | 2026-09-09 | `--gpu-mem 0.85 --max-model-len 8192 --max-num-seqs 32 --max-cudagraph-capture-size 1024`, two passes (d0 with ceiling -> `full`/`total_k*`, then d150 skip-ceiling -> `k*`) |
| infovqa, realworldqa (10 keys, no `total_k*`) | 2026-09-12 | the mns512 **interleaved** server, NO cudagraph flag, one d150 pass with the ceiling arm |

A 10-key entry is the tell: no `total_k*` means only the d150 pass ran. Whether the two eras are
comparable is an open empirical question -- plausible that they are, because streaming's steps are
large chunked prefills that never replay a captured graph, so the capture-size flag should not
bite; but that is an argument, not a measurement. The decisive control is one re-probe of ChartQA
on the new server against the stored 2026-09-09 values (`full` 53.1, `k1.00` 34.7, `total_k1.00`
96.5). Commissioned on B200-8 alongside the cvbench/mmvp pass.

Second, smaller defect in the same key: the 2026-09-12 pass probed `realworldqa:pyr`, overwriting a
box-measured entry, so 122B RealWorldQA's latency now carries `filter=pyr` while its accuracy rows
and the standing convention say box (the 35B entry is still box). The degrade filter changes pixel
content but not the token grid, so the TTFT should be unaffected and the entry is mislabeled rather
than wrong -- but it should be re-probed as box when a server pass is running anyway.

**Rule this yields:** a latency key is only as consistent as its worst entry, and the per-entry key
COUNT is a free consistency check. Before adding datasets to an existing key, diff the key count of
the existing entries and re-probe one of them as a control.

### Two config traps in the cg1024 recipe

The 2026-09-09 line is `--gpu-mem 0.85 --max-model-len 8192 --max-num-seqs 32
--max-cudagraph-capture-size 1024` on a server WITHOUT `--interleaved`, probed at `--samples 36
--warmup 4` in two passes.

1. `--max-cudagraph-capture-size 1024` is the whole point of the key and is easy to omit, because
   the flag lives on the server while the key name lives in the probe. Steps above the largest
   captured graph run eager at 35-40 ms instead of replaying at ~15 ms.
2. The `32` is `--max-num-seqs`, a SERVER flag. It reads like a sample count. Every entry in every
   key in this campaign is `--samples 36 --warmup 4`, and each key's own `_note` records it.
3. The stored `n` field is an OUTPUT, not the input that produced it: **36 samples minus 4 warmup
   = 32 measured points**. Verified across the file -- every cg1024 entry stores `n` 32 (122B
   InfoVQA stores 30, two rows over-length) while its note says `--samples 36`, and the il /
   il_unified / 35b_il keys store no `n` at all under the same note. So `n` is recorded by some
   eras and not others and is never a probe argument. Extending a key from its stored entries
   rather than from the recipe is exactly how someone probes at `--samples 32`, gets 28 measured
   points and a different sample set, and makes the new cells non-comparable for a reason that has
   nothing to do with the schedule.

The generalisable part: two numbers in the same key disagreed (`n: 32` in the entry, `--samples 36`
in its note) and the trap was reading both and not reconciling them. Reading the file is necessary
and not sufficient; the check is reconciling the readings against each other.

### Guards fail in both directions (2026-09-12)

Two guard bugs on the same day, opposite signs, both caught only by reading the file instead of
the verdict:
- A FLOPs sanity check read `vision_total` at the top level of each dataset dict; the folder writes
  it inside each ARM sub-dict (`_il_g4`, `_ilu_g4_k0.25`, ...). It got `None` and printed "REFUSE"
  on two perfectly good 122B datasets.
- A peer's arm-completion report flagged all 14 TextVQA arms because its criterion was
  `skips == 0`; the right criterion is skip DIVERGENCE between arms, since identical skips are
  perfectly paired and need no correction. Rewritten to pre-scan every arm's scored-row set and
  print IDENTICAL or DIVERGENT; TextVQA flips to clean, V*Bench flips to genuinely needing the
  common subset.

The reusable lesson is the peer's: **a guard you have learned to wave through is worse than no
guard**, because it trains you to ignore it on the day it is right. Verify a guard's verdict
against the file the first time it fires, in either direction.

## The streaming baseline was stale, and it had reversed 17 of 30 cells (2026-09-12)

The table's streaming column came from `qwen35_moe_cg1024` measured 2026-09-09; its interleaved and
unified columns were measured today, across three engine changes (post-crossing `correct_rows`,
`open_walk`, `CONTIG_ROWS`). Re-probing all 11 cg1024 datasets on today's engine, same server
config, same `--samples 36 --warmup 4`, two passes:

| quantity | n | mean | median | sign |
|---|---:|---:|---:|---|
| `full` (ceiling TTFT, the ratio denominator) | 10 | -3.2% | -4.9% | 8/10 negative |
| `k*` (Crit. Lat., the number in the cell) | 33 | **-9.1%** | -9.6% | **32/33 negative**, p~4e-9 |

The engine changes sped up the STREAMING path too, by about 9% on the critical path, and nobody had
re-measured it. Effect on the 35B block, interleaved vs streaming per cell:

| | interleaved faster | streaming faster | tied within 0.5 ms |
|---|---:|---:|---:|
| stale streaming baseline | 29 | 1 | 0 |
| one-session baseline | 12 | 14 | 4 |

So the apparent Crit. Lat. win for the interleaved schedule was a baseline artefact. Measured on one
engine the two are a wash on 35B. Seventeen cells changed verdict, every one against us.

**What survives, unchanged.** Crit. Comp. is a closed form over the same rows and is untouched:
the interleaved critical compute really is k/g of the image rows, 2-4x below streaming. The latency
column now says the compute advantage does not translate at batch 1 on a B200, because the correct
step is bound by streaming the MoE expert weights once per layer rather than by token count. The
long caption already predicted exactly that, so the table now demonstrates the caption instead of
contradicting it.

**The methodological trap, which is not the column rule.** The alarm and its retraction were BOTH
computed on `full` while the conclusion was about `k*`. Same data, opposite verdicts. Computing more
cells of the wrong quantity does not help -- ten `full` values would have produced a more confident
wrong retraction. Name the exact field the conclusion will be stated about, then apply the column
rule to that field.

**Hard rule.** Every cell in a row comes from ONE session. A stale baseline leaves no marker in the
cell, no provenance field, nothing to catch. Third instance of "the cells look fine individually and
the comparison between them is wrong", after the bicubic/box filter mismatch and the mixed-box
bounds.

Known gap while the 122B half re-probes: 35B VSR has `k*` from today but no `full`/`total_k*`,
because its d0 pass died on a missing `_vsr_images` cache in this worktree (`len(ds)==0` ->
ZeroDivisionError). Symlinked from AppCorr-vllm; VSR needs a d0-only re-run. VSR is not in this
table, only in the latency table.

## 122B InfoVQA stays a probe: OOM recovery refused by its own gate (2026-09-12 20:13)

The row is full split (2801 rows in all 14 arms) but the interleaved keep<1 arms lose 44-236 rows
to side-buffer `oom` on the longest prompts (unified 4-5x fewer), so the common subset is 79.5%.
Recovering those rows needed a server with more free memory; the only config that fit
(`--gpu-mem 0.78 --max-num-seqs 256`) was gated against the canonical mns512 rows first and
**failed: 1 pred flip in 25 scored rows** (i=2139, '3' vs '4', identical chunks). Batching is not
numerically neutral on 122B FP8 -- same phenomenon as the chunked-prefill first-token flips. The six
files were restored byte-for-byte from backup. Details and the rule in the memory file
`reference_122b_interleaved_infovqa_oom_and_batching_gate`.

## What the 2026-09-13 caption trim moved here

The tex caption was cut to three typeset lines because at 19 columns and ~120 body rows the
float no longer fit a page (user, 2026-09-13). The sentences it dropped, in full:

**Schedule definitions.** *Streaming*: the vision tower is corrected progressively and the LLM
prefills each band's final rows exactly once (chunked causal prefill). *Interleaved*: the LLM
prefills the whole base-resolution prompt once, then per band re-runs only the rows the tower
corrected in that band, plus the trailing text in the last round; every round runs at full depth,
so Comp. = 1 + k·f_img + f_text of a full-resolution pass while Crit. Comp. is the last round
alone (≈ k·f_img/g + f_text). *Depth-staged*: round r corrects its rows over the first
b_r = L(r+1)/g decoder layers only and every image row is then carried through [b_r, b_{r+1}).
*Unified*: one depth axis of tower + decoder stages cut into g rounds of equal cumulative cost;
k<1 selects each band by the layer-mean attention of the stages walked so far (progressive), not
by the full-tower score of the other arms, so its k<1 cells are not a schedule-only comparison.

**Adaptive k.** The unified schedule with a per-band pscore THRESHOLD θ instead of a fixed
budget. Score = RMS residual in raw pixel units × N·received attention; the per-band count is
ceilinged onto 1/8 buckets. θ is calibrated per (model, dataset, schedule) so the mean realised k
matches the row's k -- it is NOT portable across datasets (the score is in raw pixel units) and
not across schedules (the unified arm ranks on prefix attention). Registry:
`analysis/results/adaptive_theta.json`. The realised mean k̄ is printed under the accuracy.

**Pairing and coverage.** All arms of a model come from one served engine with paired rows, same
driver, same vision tower, greedy. Every cell is the full split unless PARENTHESISED, which marks
a reduced-n strided subset (rendered unshaded and without a preservation %); the flag is per ROW,
comparing each row's own n against its full split, not per model.

**VSR is deliberately absent**: its floor and ceiling are indistinguishable on both Qwen models
(paired p = 0.27 / 0.40), so it carries no signal about the schedule.

**GLM rows are closed form.** The FLOPs hooks never ran on GLM-4.6V or GLM-5.3, so both halves of
every GLM Comp. cell AND its full-resolution reference come from `flops_analytic`
(Glm46VDecoder / Glm53Decoder + GLM46V_VISION / GLM53_VISION) replayed from each row's `chunks`
records, not from a hooked measurement. The Qwen3.5 rows keep their hooked basis. Each entry in
the GLM flops jsons records which basis it used (`vision_basis`, `decoder_basis`, `_full_basis`).
Regenerate with `scratchpad/glm_flops_fold.sh` (CPU only): `--il-closed-vision` is what turns the
closed forms on, and without it the table falls back to the Qwen3.5-35B tower for the vision half.

**GLM-5.3-Flash box-metric cells are withheld** (RefCOCO, VisDrone Det). The model emits box
coordinates in a frame the scorer does not share: on the same VisDrone row, GLM-4.6V / 122B / 35B
agree within a few pixels (616, 274-280, 737-741, 347-350) while GLM-5.3 returns
(1200.0, 303.5, 1430.4, 376.9) -- x ≈ 1.95×, y ≈ 1.09× -- and the arm scores 0.22. Running those
datasets measures the mismatch, not the schedule, so they stay empty until the frame is fixed.
The 448-row `visdrone_det_glm-5.3-flash_floor.jsonl` is kept as the reproducer.

**GLM-5.3 file layout.** Its single chain wrote the fixed unified pair AND the two auto arms into
`*_adaptive/`, and tagged some arms `_c2` and some not. The table therefore gives that model a
tuple of candidate suffixes and lets its fixed arms fall back to the `_adaptive` session; every
other model keeps one suffix and a main session that always wins.

## How to read this table (2026-09-14, user's standing rule)

**The objective is preservation against the ceiling, not a ranking of our schedules.** User,
2026-09-14: "내 목표는 무슨 기법이던 ceiling 에 가깝게 붙으면 장땡이야. 기법들 사이의 비교에서 교훈을
얻을 생각은 추호도 없다." So the number that matters in every cell is `best arm / ceiling` at the
operating point, and which of streaming / interleaved / staged / unified / adaptive produced it is
bookkeeping. Do not queue arms whose only purpose is to separate two of our own schedules, and do
not write analysis about the separation.

### Best preservation per cell, at the two cheap operating points (2026-09-14 00:30)

At k=0.50 every cell except VisDrone is at or above 93.6% of ceiling, and 26 of 34 are above 97%.
The cells that are NOT near ceiling, and so are the only ones where more work can pay:

  VisDrone Det     35B 90.1 / 122B 88.7 / GLM-4.6V 95.2 %   (k=0.50);  77.8-90.9 % at k=0.25
  VisDrone Count   35B 86.0 / 122B 97.0 / GLM-4.6V 98.7 %   (k=0.50);  78.1-94.7 % at k=0.25
  InfoVQA          96.9-97.6 % at k=0.50;  88.2-90.7 % at k=0.25   -- the steepest k-response
  ChartQA          97.6-100.3 % at k=0.50;  93.9-98.0 % at k=0.25
  TextVQA          97.9-98.9 % at k=0.50;  95.3-96.2 % at k=0.25
  122B MMVP        93.6 % at k=0.50  (n=300, underpowered -- see the 122B MMVP memo)

Everything else (V*, RealWorldQA, CV-Bench, RefCOCO on every model; GLM-5.3 on all four of its
finished datasets) sits at 97-101 % of ceiling at k=0.50 and 95-101 % at k=0.25. Those cells are
DONE: no schedule can be meaningfully better than the ceiling it already matches.

Note VisDrone Count's ceiling is 15.8-18.9 absolute, so its percentages move on a few questions
and should be read with the raw numbers next to them.

## Bookkeeping: which cells have a matched-budget fixed arm

Kept only so nobody later reads a blank cell as a failed run. Per the rule above, the
adaptive-vs-uniform contrast is NOT the point of the table; it is recorded here and nowhere else.

**Where the uniform-vs-adaptive contrast exists.** Qwen3.5-35B, Qwen3.5-122B and GLM-4.6V: all
ten datasets. GLM-5.3-Flash: vstar, mmvp, realworldqa, cvbench ONLY -- those four ran before the
2026-09-14 00:05 instruction ("unified 는 굳이 안 해도 된다. auto 만 해"), after which the
remaining GLM-5.3 datasets (textvqa, infovqa, chartqa, visdrone_count) run floor / ceiling /
auto(θ50) / auto(θ25) and no fixed pair. Those four cells render BLANK in the fixed Unified
column BY DESIGN, not because a run failed, and they support "adaptive at θ reaches X against a
floor/ceiling span of Y-Z" -- nothing about uniform vs adaptive.

**The finding, stated per cell.** Adaptive allocation helps where the model still loses accuracy
under a uniform budget of the same size, and is indistinguishable from uniform where the fixed
arm already sits near the ceiling. Compute and latency are unchanged either way.

  moves:  ChartQA and VisDrone-count on 35B / 122B / GLM-4.6V (+1.1 to +2.2 pt and +0.6 to
          +1.5 pt); CV-Bench on GLM-5.3 at the 0.50 target only -- paired on the common 2634
          rows, 84.9658 -> 86.1807, +1.21 pt, discordant 64/32, McNemar p = 0.0014.  At the 0.25
          target the same cell is null (+0.27 pt, 51/44, p = 0.54): the budget is then too small
          for reallocation to buy anything back.
  null:   RealWorldQA, RefCOCO, TextVQA, InfoVQA, and CV-Bench on the Qwen rows and GLM-4.6V.

The axis is per (MODEL, DATASET) cell, not per dataset -- CV-Bench is null on three models and
significant on the fourth. The predictor is headroom under the fixed arm, which the reader can
check directly from the Low-res. / Full-res. bounds already printed under each dataset label:
GLM-5.3's CV-Bench fixed k0.50 sits at 84.97 against a 84.31 / 86.39 span (near the floor, room
to reallocate), while the 122B and GLM-4.6V fixed arms on that dataset already sit near their
ceilings. State it that way rather than asserting a "steep k-response" we did not measure.

## The 122B V* adaptive Crit. Lat. cell is withheld (cap artefact, 2026-09-14)

Measured at the server's default pseudo-sequence cap, the 122B V*Bench adaptive arm at the 0.50
target read 101.1 ms against the fixed k=0.50 arm's 65.5 ms -- adaptive apparently 55% SLOWER,
against every other cell in the table where it is equal or faster. It is an engine artefact.

**Mechanism.** Adaptive holds the MEAN k (0.501 measured against a 0.50 target) but reallocates,
which raises the per-band MAXIMUM: max |P_r| median 589 against the fixed arm's 457. The server
runs a correct round as at most `max_num_seqs` 1-token pseudo-sequences, so a band above the cap
splits into two scheduler passes and pays a second round of prefill overhead. Matching the mean
says nothing about the max, and the cap bites on the max.

**Ladder, same 36 images, same thetas, rows still crossing the cap in brackets:**

    cap 512 [25/36]   adaptive  101.1 ms   fixed  65.5 ms    gap +35.6
    cap 602 [18/36]   adaptive   72.7 ms   fixed  66.5 ms    gap  +6.2
    cap 788 [ 3/36]   adaptive   69.9 ms   fixed  69.0 ms    gap  +0.9   <- cap-free: EQUAL

Only the arm that crossed the cap ever moved. The three arms that never crossed (fixed 0.50,
fixed 0.25, adaptive 0.25) are flat across all three caps, which is what makes this a controlled
result rather than a story. 788 is the engine's own ceiling at gpu-mem 0.90 (vLLM refuses a cap
above the available Mamba cache blocks and names the number: 602 at 0.85, 788 at 0.90).

**Why the cell is withheld rather than corrected in place.** The table's whole 122B latency column
is measured at cap 512 / gpu-mem 0.85, the engine that produced every other 122B latency cell.
Printing the 788 value in one cell would mix two engines inside one row. Printing the 512 value
would print an artefact, and a reader takes a printed number, not its footnote. So the cell is
empty and this section is the record: at a cap where it does not bind, the arm equals the fixed one.

**Carry this as a deployment note, next to the cap and not in the method's latency column**:
wherever `max_num_seqs` is near the per-band size, adaptive allocation pays sub-batches a uniform
budget of the same mean size avoids. It is invisible on ChartQA / TextVQA / VisDrone-count, whose
bands are far below any of these caps.

## GLM-5.3's latency row is EAGER-only (2026-09-14)

Marked with $^\ddagger$ in the table. GLM-5.3-Flash must run `--enforce-eager` at TP=2, so it is
the one model whose Crit. Lat. does NOT carry the CUDA-graph correct step -- the single largest
latency optimisation in this system (on 35B that step went 45.9 -> 24.5 ms, and served Crit. Lat.
fell by a constant 11-13 ms per round on every cell).

**Evidence that it is the eager step and not the model or the images.** `last_correct_step_ms` is
~93 ms on EVERY dataset and EVERY arm, moving only ~15% across a 7x range of prompt length:

    chartqa         620 tok    91.1 - 93.9 ms
    textvqa        1067 tok    92.2 - 94.7
    visdrone_count 1401 tok    92.8 - 94.6
    vstar          4425 tok   102.2 - 110.7

That is the fixed CPU-launch cost measured at 45-55 ms on 35B, roughly doubled: 45 KDA + 11
sparse-MLA layers launch far more kernels per round than a dense decoder. It is the numerator of
every cell, which is why three of the four sit at 102-107 ms whatever the image is, and why
ChartQA reads 104-106% of its own full-resolution pass. V* is the only cell whose ratio looks
normal (30-36%) and it does so because its DENOMINATOR is large (full 386.5 ms), not because its
tail is smaller.

**The graph attempt failed, once, as agreed.** `--max-model-len 8192` without `--enforce-eager` at
gpu-mem 0.90: `torch.OutOfMemoryError: Tried to allocate 24.13 GiB` during capture. Halving the
context does not free enough on top of 328 GB of weights across the pair, and 0.88 already yields
negative KV, so the window is narrow -- this needs a third card for KV or a smaller capture set,
not a flag change. Not retried.

**What the row still supports, and what it does not.** WITHIN the row both arms pay the same
numerator, so the adaptive-vs-fixed comparison is sound and is the answer this column exists for:
at matched mean k the two are within 3.0 ms on all four datasets (chartqa 106.5/105.4 and
102.2/104.2; textvqa 103.9/106.9 and 105.2/103.6; visdrone 106.3/105.0 and 105.6/105.4; vstar
135.7/137.8 and 120.8/117.8), and `ttft_last_band_p95` stays near the median, so no band crossed
max-num-seqs 1024 -- none of the 122B V* cap artefact here. ACROSS rows it is not comparable: the
other three models carry graph numbers.

Measurement: B200-8 GPU0+1 (TP=2) with the driver's vision tower on GPU7, gpu-mem 0.90,
max-model-len 16384, max-num-seqs 1024, d150 anchor, 36 samples / 4 warmup dropped, concurrency 1,
n=32 per cell. Two passes: unified arms on one server, the ceiling on a second fresh one (the
GLM-5.3 server dies on its first unified request after any non-unified one -- six occurrences,
always `Inplace update to inference tensor`). `VLLM_GDN_DECODE_KERNEL=triton` was NOT set for this
run, unlike the accuracy chains. Numbers here were re-aggregated from the shared probe rows under
b200-8_logs/probe_glm53_ilu_adaptive/, not transcribed from a log; they match the run's report.

## GLM-5.3 grounding cells: as-scored, with the capped count stated (user 2026-09-15)

The 512-token cap ends GLM-5.3 mid-thought on sharp grounding frames, and it hits the arms
unequally -- the sharper the image, the longer it reasons. VisDrone Det, 448 rows, under
c7320f9 (as scored / excluding finish_reason=length rows / capped count):

    floor     21.21 / 21.64 (n=439) / 9 capped
    ceiling   30.13 / 44.48 (n=299) / 149 capped      <- 33 % of the arm
    auto50    28.79 / 29.59 (n=436) / 12 capped
    auto25    25.67 / 26.02 (n=442) / 6 capped

**Decision: print the as-scored column** and state the capped count. Excluding capped rows would
leave each arm on a different row set (299 vs 436), the exact trap that has already cost this
campaign twice ([[feedback_common_subset_and_power]]), and the 512 cap is the standing protocol.

**But the as-scored column flatters us and must carry the footnote.** It puts preservation at
28.79/30.13 = 95.5 % with ceiling-vs-auto50 a wash (p = 0.53); on the 289 rows where every arm
answered inside the budget the same data reads 38.75/45.33 = 85.5 % with the ceiling ahead at
p = 0.005. The ordering there is the ordinary one and every gap but auto50-vs-auto25 is
significant: floor 29.41 < auto25 35.64 < auto50 38.75 < ceiling 45.33. Quote the 95.5 % only
next to the capped counts; never alone.
