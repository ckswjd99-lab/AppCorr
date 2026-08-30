# HANDOVER: GH200 becomes the main session (B200 maintenance 2026-08-27 24:00)

The B200 stops at midnight (processes die, /NHNHOME and /home survive). GH200 is
unaffected and takes over as the MAIN working session. This document is the complete
context transfer; the user talks to GH200 directly from here on.

## Who the user is, how to work with them
- Korean for conversation; English for code/comments/memos. Terse, direct, expects
  reasoning to be shown. "(SQ)" prefix = answer from context, no tool work.
- Standing rules (all learned the hard way, all in AppCorr-flops/CLAUDE.md — READ IT):
  preservation-first reporting; floor+ceiling with every sweep; measure before claiming
  a mechanism (three retractions in one session taught this); subsets locate, never
  size; numbers come from logs, never notifications; no silent fallbacks — crash;
  launch checklist (start proof + full-coverage monitor + no output-path reuse);
  one task per GPU; effect size + interval, never bare numbers.
- The user decides presentation (e.g. left near-zero-gap rows as-is deliberately);
  don't "fix" table cosmetics unasked. They accept known-caveat numbers when the
  provenance is stated (OpenVLA July numbers entered with lost-primary caveat).

## The deliverable: the evaluation table
- Source of truth: `analysis/experiments/make_eval_table.py` on
  `origin/develop/qwen35-appcorr` (HEAD b9e4c71+). Regenerate:
  `python3 analysis/experiments/make_eval_table.py --format latex --keeps 0.25 0.50`
- Structure: Low-res | Ours(25%) Acc/Comp/Crit | Ours(50%) A/C/C |
  Streaming(k=1.0) A/C/C | Full-res A/C. Model names wrap Name\\(size).
  Model order fixed by user: Gemma3, OV2, Qwen2.5, Qwen3.5-35B, 122B, OpenVLA,
  DINOv3, SAM3, OpenCLIP, VGGT.
- Footnotes: ddagger(Gemma3/OV2: accuracy still upfront-arm), dagger(Qwen3.5 Ours =
  keep-limited streaming), section(Qwen2.5 bs/OOM caveats), pilcrow(122B FP8 broken
  kernels). Lookups key on BASE model name (decoration stripped) — don't regress.
- Data flows: file-based cells fill on regeneration; literals carry GH200/July
  numbers with provenance comments inline.

## State at handover (see also RESUME_20260827_maintenance.md, now superseded by this)
- Qwen2.5-VL: CLOSED at every scale (your own work; b9e4c71 merged it).
- Qwen3.5-35B: closed except VSR streaming k1.0 cell (not measured; low value).
- 122B-FP8: compute cells filled (shape-valid); accuracy blocked on sm_100 kernels.
  Recorded options: 2-GPU bf16 device_map, or FP8Linear dequant-to-bf16 shim
  (recommended, awaiting user go).
- Streaming category (user's intended architecture): causal LLM streams, vision
  corrects. Filled: Qwen3.5 (RWQA 77.25/99.8%, ChartQA 88.32/99.7%), OV2
  (chartqa 86.40/99.6%, textvqa 76.01, infovqa 73.72, realworldqa 69.41, pope 87.83
  + gqa/vsr/docvqa landing tonight — check analysis/results/ov2_*/streaming_g4.json
  after the window and regenerate), OpenVLA (LIBERO-Spatial 81.6/98.6%, July numbers,
  primary lost — re-measurement queued as VLA-track task).
- Tonight's B200 queue (dies at midnight if unfinished; all resumable):
  OV2 gqa→vsr streaming, then docvqa retry (guard bound widened to 1.9 after a
  false positive). Resume command per arm is in scripts/streaming_category_chain.sh.

## Gemma 4 31B (added 2026-08-28, user decision)
The bigger-model extension is `google/gemma-4-31B-it` (OV2 has no larger sibling).
Weights cached on /NHNHOME (survive maintenance). Level-1 axis DONE and gated
bitwise (cab2f9e): appcorr/models/gemma4/unified.py + gemma4_axis_gate.py.
READ docs/memo/gemma4_port_scoping.md before continuing — the mask finding
(image tokens causal on full layers, block-bidir on sliding only) makes this a
third point on the causal<->bidirectional axis, and steps 2-6 of the port plan
are laid out there. B200 work (31B needs ~65GB GPU).

## 2026-08-28 EOD state (this supersedes the priority list below where they conflict)

USER PRIORITY SHIFT (evening): MMVP/CV-Bench proved resolution-INSENSITIVE for
modern models (gaps 1-2.5pp; only Qwen2.5 shows 7-9pp). Prioritize
resolution-sensitive, real-understanding tasks: RefCOCO-class grounding,
MME-RealWorld (spec ready, 23.6k, deferred for scale), and the L3 axis --
the Mistral L3 sweep showed L2's 1.67pp gap opening to 12pp at 1/8, with full
arm structure recovering (36/69/72%+). Existing rows stay.

Landed today, all pushed:
- Gemma 4 31B: levels 1-3 (axis, vision fork, one-shot oracle) gated bitwise;
  mmvp/cvbench 4-arm rows + FLOPs (crit 91-94% = one-shot structure argument
  for the interleaved/streaming ports).
- Mistral Small 3.1: FULL port (axis + Pixtral fork, G1/V2 bitwise) +
  qsel arms. Query-aware selection: +1.33pp on corrected k0.50 at 2x total
  (extra approx-prefill, overlappable) -- worth retrying on MoE (qwen35) where
  the extra prefill is ~10% not 100%.
- Streaming-above-ceiling MECHANISM (GH200 elimination + cross-model): partial
  vision staleness as beneficial perturbation for grounding; 4 observations,
  one 4-sigma. Approx-LLM pass STRIPPED from streaming executor (user: keep
  the stripped variant for the paper); Qwen2.5 RefCOCO streaming cell:
  89.55 (101.5%) | 34.04TF (107.1%) | 10.35TF (32.6%).
- RefCOCO capability probes (50): gemma4 48% / mistral 44% (0-1 FRACTIONS!) /
  MG 16% / vs Qwen ~88-93. THREE different coordinate conventions -- match
  convention before judging capability (qwen3.5 = 0-1000, rescale in driver).
- qwen35 RefCOCO ceiling: running acc ~93.1%, jsonl at analysis/results/
  qwen35_accuracy/refcoco_ceiling.jsonl, RESUME by rerunning the same command
  (--dataset refcoco --arms ceiling). Floor/streaming/keep arms not started.
- MG mmvp bounds: 70.0/74.33 (channel-fixed). MG harness needs
  reasoning_strength=low + forced ' to=user<|message|>' generation prefix
  (auto by model id in vlm_bounds_oracle).

DECIDED (user, 2026-08-28, via GH200): degradation-filter standard = **OPTION B**
-- pyr going forward, past bicubic numbers stand with a footnote (pyr==bicubic
in the mean licenses this), and ONLY the BOX-measured arms get re-run ("Box만
다시 하자"). Probe chain that grounded it: bicubic -4flips- box -6flips-
pyr(archetype); box was the outlier (+4pp floor). Action for the B200 on
return: flip qwen35_accuracy default box->pyr, re-run only the arms that were
measured under box, resume the halted qwen35 re-measurement in that form.

## Open items, priority order (the backlog GH200 now owns)
1. Gemma3/OV2 ddagger: progressive-arm accuracy re-evals (the ONLY remaining ddagger).
   Wide-gap datasets first (chartqa/textvqa/docvqa/infovqa). B200 GPU work — queue for
   when the box returns, or coordinate with the user.
2. Qwen2.5 streaming executor mode (joins the streaming category; your box, your code).
3. VGGT ours-below-floor anomaly (padding-independence confirmed; cause unknown).
4. OpenVLA-OFT LIBERO re-measurement (pipeline committed 25e1339, /tmp results lost;
   T1 rerun instructions in analysis/experiments/OPENVLA_OFT.md; B200 EGL-2 only).
5. 122B dequant shim if the user green-lights.
6. Parked (keep parked): k0.50 energy×attn rerun, Jaccard, bs=1 bounds, avg-attn
   ImageNet 2 arms, OpenCLIP one-shot FLOPs cells.

## Key technical conclusions this week (with where they're recorded)
- Progressive per-round selection is canonical; upfront double-pass removed on
  gemma3/ov2/sam3 (memos + commit messages; totals converged to 114-143%).
- VGGT padded-width correction was NOT idempotent (selection-set dump proved it);
  de-padded arm is accuracy-neutral, 4x cheaper.
- Scoring null (energy vs energy×attn) is a three-point CONCLUSION; keep energy×attn
  for cross-model consistency.
- Text-split schedule: -0.02pp at full scale, 33pp compute saving (your result).
- FP8 chunking claim RETRACTED (broken kernels); only harness==stock survives.
  bf16-dense noise-floor point stands (yours).
- VSR: degradation-insensitive across model sizes (1pp gaps at 4B/8.5B/35B) —
  the boundary-condition row for where AppCorr pays.
- Streaming category identity: total ≈ full + one vision pass, critical ≈ 1/g.

## Practical gotchas for the B200 (when it returns)
- datasets .filter caches by lambda fingerprint — filesystem-dependent filters need
  load_from_cache_file=False (VSR bit this).
- Qwen3.5 evals need enable_thinking=False (truncated-<think> scored 18%).
- kernels==0.15.2 + one ONLINE run to cache deep-gemm (then offline OK) — but FP8
  outputs are garbage on sm_100 regardless.
- PUSH the vfm_accuracy campaign logs (B200 return item, found 2026-08-30):
  analysis/results/vfm_accuracy/ holds only the SAM3 JSONs in git; the dinov3_*
  and de-padded vggt_* logs live ONLY on the B200 disk, so every other box
  renders those ours-accuracy cells as "--". The table loader now reads
  {tag}.json first, {tag}.log second -- committing the logs (or re-emitting
  them as JSONs) fills the cells everywhere.
- The orphan hazard: two multi-day orphan campaigns were found writing into live
  result files. On any fresh session: `ps -eo pid,ppid | awk '$2==1'` scan first.
- FORK-PORT CHECKLIST (two scale-exposed incidents on 2026-08-28, both drops of
  reference-implementation guards): (1) worker-provided invariants -- no_grad --
  do not port themselves; (2) scale-guarded loops -- query-chunked attention
  stats (gemma3's _incoming_attention chunking) -- do not port themselves. A
  port that "works" on RefCOCO-scale images can hide a 57GB materialization
  that only 2K-resolution CV-Bench triggers. Audit both before trusting a port.
- Degradation convention (audited 2026-08-28): canonical = cv2.pyrDown chain in
  native coords + sampled-resolution cap (offload/policies/transmission/
  laplacian.py; pyramid_degradation_native_vs_canvas.md). Oracle drivers use
  BOX-down + BICUBIC-up as the sanctioned approximation. BICUBIC-down is NOT
  neutral: paired probe flipped 4/50 RWQA floor answers (3:1 toward box).
  qwen35 switched to box (re-measurements in qwen35_accuracy_box/); check any
  new driver's degrade() for BOX + cap before trusting its floors.
- no_grad is the WORKER's job, so any in-process driver that calls
  approx_forward/correct_forward directly must supply it itself. The failure mode
  is a silent ~1.4x activation-memory tax that presents as a plausible-looking
  "large-image OOM ceiling" (GH200 found this 2026-08-28: every OOM-skip in the
  Qwen2.5 record — 8, 207, 143 — was this, not image size; 95GB -> 68GB peak once
  wrapped). B200 drivers audited clean (ov2/qwen35/sam3/gemma3 oracles all
  decorated). Check the guard FIRST when an eval "hits a memory ceiling".
- Never chain a merge-resolution script && git commit in one shell call: a
  resolution script that dies BEFORE writing leaves the conflicted file in the
  tree, and the chained commit ships the conflict markers anyway (GH200 did
  exactly this 2026-08-28, caught one verification step later). Resolution,
  parse-verification, and staging are three separate steps, in that order.
- Fork-port checklist (two same-day instances, both fixes present in the
  reference all along): (1) worker-provided invariants -- no_grad, see above;
  (2) scale-guarded loops -- gemma3's `_incoming_attention` chunks its fp32
  attention-statistic accumulation and documents why; the Qwen2.5 port dropped
  the chunking and OOM'd at CV-Bench resolution (57GB for one segment's matrix
  at ~30K patches). When porting, grep the reference for chunk/loop guards and
  carry the docstring's WHY, not just the math.
- A REAL capacity ceiling for the fork path does exist, distinct from the two
  false ones: ~24K-token LLM contexts (WildVision 4032x3024 at the 12.8M-px cap)
  carry fork caches (vision KV + per-layer blocks_out_sum/KV) that structurally
  exceed the ~28GB headroom beside the resident 32B model. Signature of REAL vs
  false: identical, deterministic skip sets across schedules, stock ceiling
  unaffected. 54/500 on WildVision; judge comparisons restrict to the 446 common.
- Category-identity audit: when an arm joins a CATEGORY (streaming, one-shot,
  interleaved), diff its measured cost structure against the category identity
  FIRST -- the identity is a free auditor. The Qwen2.5 streaming arm ran at
  200.4% of full against the category's ~1.25x for a full day before the FLOPs
  cell forced the comparison; the excess was a semantically-dead approx-LLM
  pass inherited from the chassis (stripped, bitwise-gated, -> 107.1%).
- BATCH THE ORACLE DRIVERS (user directive 2026-08-28, measured on mistral3):
  every accuracy oracle runs bs=1 by default and most of its wall-clock is
  batchable. Two patterns, both validated acc-neutral on 50-sample bs1-vs-bsN
  A/Bs before first use (accuracy ties, +-0.01-0.03 fraction-coordinate churn,
  the familiar bf16 batching noise): (1) bound arms (ceiling/floor) are plain
  stock generate -- left-padded chat-template batches, 0.89->0.20 s/ex (4.4x)
  at bs=8; (2) fork arms keep the correction bs=1 (the vision forks carry no
  batch dim by design) but queue the per-sample inputs_embeds and flush one
  left-padded batched generate -- 0.83->0.21 s/ex (4x) at bs=16, zero-embed
  padding rows masked out. Also add per-sample incremental jsonl while there
  (end-only out-json loses a whole arm to one crash). Port both patterns to
  qwen35_accuracy / gemma4 / MG / vlm_bounds_oracle drivers when each is next
  touched; reference implementation: mistral3_oracle.py (--bs).
  CAUTION (2026-08-29, found the hard way): validate the batched-BOUND path
  PER OUTPUT-FORMAT CLASS, not once. It broke EOS stopping on TextVQA-style
  free-text prompts (answer + hallucinated self-QA continuation, scored wrong
  by VQA normalization) while RefCOCO passed -- bbox regex reads the string
  front, so the refcoco-only 50-sample validation could not see it. Caught by
  a corrected arm landing 6pp ABOVE the depressed ceiling. The inputs_embeds
  batched path (fork arms) stops cleanly. Until the bound path is re-validated
  per class, bound arms on exact-normalized free-text datasets run bs=1.
  LIMIT (2026-08-30, gemma4 interleaved): pattern (2) does NOT apply to
  INTERLEAVED arms -- their decode must ride the fork's per-layer K/V (a
  `generate(inputs_embeds=...)` flush would re-prefill the LLM and erase the
  walk), so the only batchable piece is stacking the manual decode loop across
  samples (variable-length KV left-pad + hybrid sliding/full masks; est. ~15%
  wall-clock for short answers -- not worth mid-campaign). Batching interleaved
  properly means batch-capable forks (vision + LLM both assert B=1 today);
  design that in at the NEXT model port, don't retrofit.

## Final state at shutdown (23:15 sweep, 2026-08-27)
Everything is committed and pushed (HEAD 3c64def at sweep time). Landed tonight:
OV2 streaming full splits POPE 87.83 / GQA 63.09 / VSR 78.97 (both GQA and VSR at
100.2% of ceiling — narrow-gap noise, recorded as-is). The ONLY job killed by the
maintenance window: OV2 DocVQA streaming retry (was ~sample 1500/5349 at midnight;
guard bound already fixed at 1.9, commit 570d307). First B200 job after the window:
  python analysis/experiments/ov2_oracle.py --dataset docvqa --full --level 2 \
    --arm streaming --groups 4 --out-json analysis/results/ov2_docvqa/streaming_g4.json
Then regenerate the table and the OV2 streaming row is complete.

## Storage constraint on GH200 (user directive, 2026-08-27)
GH200 storage is tight — plan around models ALREADY downloaded there; anything new
must be minimal. Consequences for the backlog above:
- Feasible now, zero download: Qwen2.5-VL streaming executor mode (the checkpoint is
  already local to you) — this is your top GPU item. Also all non-GPU work: table,
  analysis, memos, paper prose.
- Feasible only if space allows (check `df -h` + your HF cache inventory first, and
  report to the user before downloading anything): Gemma3-4B (~9 GB) and OV2-8.5B
  (~18 GB) for the ddagger accuracy re-evals.
- NOT feasible on GH200 (stay queued for the B200 post-maintenance): Qwen3.5-35B
  (~70 GB), 122B (~130+ GB), the VFM tracks (DINOv3/SAM3/OpenCLIP/VGGT — the
  datasets are the heavy part and live on the B200), OpenVLA/OFT (sim env + EGL).
GH200's role as main is therefore: run what fits locally, and act as coordinator /
queue-planner for the B200 work until a session returns there.

## Appendix: distilled standing knowledge (B200 session memory, re-stated)
User: conversation in Korean, code/comments/memos in English. "(SQ)" prefix = quick
answer from context, no tool work. Working-style rules are in the main body above.

Project facts worth carrying:
- ProgVFM paper (MLSys'27 draft) is the theoretical base: pscore = contrib_i
  (Eq 6, residual x per-head attention, fitted weights); interleaved-is-cheaper (3.3).
- Progressive per-round selection is canonical (user decision 2026-08-26); upfront
  selection was a botched implementation that double-ran the vision tower.
- Interleaved correction contract: docs/memo/interleaved_correction_contract.md —
  round r corrects group r only; corrected increments are NOT written back into
  blocks_out_sum (known defect class); read it before touching any multi-round path.
- Streaming beats approx-then-correct wherever the LLM is causal (OV2 verified:
  ChartQA 85.0 stream vs 81.0 interleaved, g-independent). Gemma3 cannot stream
  (bidirectional image tokens); pi0-FAST prefix is bidirectional (+12% CE / -4pp);
  lerobot+transformers-4.57.1 double-scales Gemma embeds (neutralize sqrt scaling).
- Qwen3.5: GatedDeltaNet linear attention (30/40 layers) makes LLM correction
  ill-defined, so streaming is the only LLM arm. MoE FLOPs are count-based
  (n_tok x top_k), charged via class-name special hooks (raw-Parameter experts are
  invisible to module hooks). M-RoPE: get_rope_index returns (3,B,T), pass as-is;
  decode needs explicit positions (rope_deltas caching trap); enable_thinking=False
  for evals (truncated think-mode output scored 18%).
- 122B-FP8: deep-gemm targets sm_90; on the B200 (sm_100) every arm returns garbage,
  so the earlier "chunking is lossy under FP8" claim is RETRACTED (measured on a
  broken model). What survives: single-chunk==stock at TV 0.0000 (harness exact).
  Spectrum framing stands as hypothesis only: a context-dependent quantizer feeding
  a discontinuous amplifier (MoE routing) makes chunk-boundary numerics a cliff
  rather than noise; pre-flight any quantized streaming with single-chunk==stock
  then a boundary sweep. Accuracy paths if pursued: 2-GPU bf16 device_map, or a
  dequant-to-bf16 shim over FP8Linear (recommended), or Blackwell kernel support.
- Qwen2.5-VL: correction is cuda:0-only on the B200 (Triton dies on cuda:1);
  upfront selection is FREE there (its vision runs full-depth at arrival 0), so it
  keeps upfront while gemma3/ov2/sam3 moved to progressive.
- SAM 3: 55% recompute recovers ~90% (COCO/LVIS) at 0.60x compute; object size is
  the only variable predicting damage; presence token is task-dependent. SA-Co:
  cgF1 + 3-annotator oracle merge are official SAM3 code; we own prompt-sourcing
  and image download only; start with Gold.
- VGGT-Omega: read docs/memo/vggt_omega_status.md first; serving-path numbers only;
  DINOv3 blocks now shared; ours-below-floor anomaly open (padding-independent).
- Gemma3 facts: image tokens bidirectional (no streaming); pre_global does NOT
  transfer (sliding window never bites at 277 tokens); at 896px vision is 72% of
  the full-forward axis (this superseded the old "prefill is 1.8x vision" note).
- OpenVLA July LIBERO-Spatial (82.8 ceiling / 81.6 stream / 17.2 floor): setting
  authenticated via commits (initial-state fix predates measurement, 500 ep,
  parity-gated) but the primary jsonl was lost to /tmp; user accepted the numbers
  for the table with a provenance comment. OFT: implementation + gates committed,
  results lost, re-measurement pending (B200).
- 2D locality is the unifying research axis: process/transmit/correct in 2D-local
  blocks; 2D-local causal order is lossless (raster not required); overlap recovers
  grounding overhead. 7B grounding generation is degenerate — use VQA for
  model-size scaling. nr=400-style subsets are sanity-only; close comparisons need
  full splits. VSR is degradation-insensitive (low-frequency spatial relations) —
  the boundary-condition row for where AppCorr pays.
- Environment: the HF token env var must be UNSET, not empty (empty string makes a
  broken auth header); Triton needs _configure_compile_environment() in every
  executor (and ~/.triton/cache is empty by redirect, not disuse); py-spy blocked
  (no ptrace) so use a SIGUSR1 faulthandler; /tmp is noexec on the B200; datasets
  .filter caches by lambda fingerprint so filesystem-dependent filters need
  load_from_cache_file=False; B200 rendering: no Vulkan, EGL device 2; instance
  reset procedure in /NHNHOME/share/cjpark/backup/RESTORE.md.
- Methodology rules (user feedback, standing): floor + ceiling with every sweep;
  preservation before recovery; effect size + interval (bootstrap the gap, paired
  counts when skewed); never claim a mechanism unmeasured — label hypotheses and
  check with a monotone quantity; the harness must never share the variable under
  test between arms; check running jobs every few minutes; no train/eval index
  overlap for fitted pscores; one task per GPU; apply validated techniques
  proactively when extending to analogous signals.

## 2026-08-30 addendum (GH200): two sharpness-harm findings + filter non-neutrality, both measured

1. Mistral-TextVQA: the corrected>ceiling anomaly root-caused to REAL model behavior, not any
   harness confound -- self-QA tails track input SHARPNESS (ceiling 31.2% of preds, pyr floor
   19.6%, degraded-base fork arms ~0), unchanged under decode-mechanism unification (Option-C
   rerun: embeds ceiling 61.52 vs ids 61.08). Degradation improves Mistral's output discipline;
   whole-string VQA scoring legitimately penalizes the rambling. All arms now decode-unified.
2. Gemma4-RefCOCO: floor>ceiling inversion (52.12 vs 47.07, dose-response over keep) is
   PYR-FILTER-SPECIFIC: BOX floor at n=1000 = 47.70 ~= ceiling. Two framings recorded: (a) pyrUp
   low-frequency overshoot as inadvertent enhancement -- filter effects are MODEL x TASK specific
   (box was the outlier on qwen35-RWQA, pyr is the outlier here; neither is universally neutral);
   (b) protocol-faithful: the real Laplacian transmission reconstructs bases WITH pyrUp, so under
   the deployed system gemma4 genuinely sees (and scores 52 on) that input -- the row then says
   "no gap for AppCorr to close on this model/task", which is a valid boundary-condition finding.
   Either way the row is non-discriminating; batching exonerated on both arms (bs=1 spots match).

## 2026-08-31 addendum (GH200): greedy generate is not run-stable on Gemma4-31B

Probed after the interleaved identity gate "regressed" without a code change:
the same `model.generate(**enc, do_sample=False)` call, same process, same
weights, flips between adjacent boxes on near-tie RefCOCO samples (e.g.
'402,444,541,935' vs '403,443,553,936'; also 306 vs 308 on another sample).
bf16 reduction nondeterminism -- greedy does NOT imply reproducible.

Consequences, applied:
- Text-equality gates against a single ceiling draw are invalid on this
  model. The interleaved gate now samples ceiling 3x and checks MEMBERSHIP
  in the flicker set. (The fork walk itself reproduced bitwise across
  processes -- the manual path is more deterministic than generate.)
- A ~0.1pp-scale accuracy wobble between reruns of the same bound arm is
  expected noise here, not a measurement problem (anomaly-heuristics rider).
