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
- The orphan hazard: two multi-day orphan campaigns were found writing into live
  result files. On any fresh session: `ps -eo pid,ppid | awk '$2==1'` scan first.
- no_grad is the WORKER's job, so any in-process driver that calls
  approx_forward/correct_forward directly must supply it itself. The failure mode
  is a silent ~1.4x activation-memory tax that presents as a plausible-looking
  "large-image OOM ceiling" (GH200 found this 2026-08-28: every OOM-skip in the
  Qwen2.5 record — 8, 207, 143 — was this, not image size; 95GB -> 68GB peak once
  wrapped). B200 drivers audited clean (ov2/qwen35/sam3/gemma3 oracles all
  decorated). Check the guard FIRST when an eval "hits a memory ceiling".

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
