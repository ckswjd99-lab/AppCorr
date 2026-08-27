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
