# Handover from GH200 session — 2026-09-01

The GH200 Claude session is being repurposed for new-technique development. All
code/results are on `origin/develop/critical-flops-accounting` (HEAD 457894d,
clean tree, nothing unpushed). This memo records the items that lived only in
that session's conversation.

## Open work items (now unowned — NHN session or a future GH200 session picks up)

- **Mistral GQA campaign — never launched, user go still pending.** Estimated
  ~7h on one GH200: 50-sample bs1-vs-bs8 per-class validation gate first (GQA is
  a new output-format class) → FLOPs-first n=8 pass → ceiling/floor bs=8
  (~42 min each) → corrected k0.25/k0.50 (~1h each) → streaming k1.0 (~3.5h).
  A Mistral V* FLOPs pass (n=12, minutes) can piggyback on the same load.
- ~~Remaining V* compute gaps: Mistral, Qwen2.5~~ **CLOSED on B200 2026-09-01**
  (cea9022: mistral24b/vstar + qwen25vl_32b/vstar in inprocess_flops.json,
  table cells wired). Do not redo.
- Deferred cleanups: `make_eval_table.py` md-format row-length bug;
  `--group-by-image` (shared image-prefix KV) gate candidate for gemma-family
  VisDrone multi-question (rejected on qwen35: hybrid linear-attention state).

## Parked user-decision items

- WildVision judge model choice (docs/memo/wildvision_judge_design.md).
- VGGT paper-draft discrepancy: draft cites 0.0600/0.0540 (quarantined
  padded_arm); canonical is 0.0632/0.0544 — the ours-below-floor anomaly is
  wider than the draft shows. Flagged to user, no response yet.
- PAT rotation: both boxes held plaintext git credentials (NHN origin URL +
  GH200 ~/.git-credentials); rotate together.
- 122B-chartqa FLOPs stays pre-attention-fix by user deprioritization
  (annotated in inprocess_flops.json).

## GH200 box-local state

- Weights on disk: Mistral-Small-3 24B (Gemma4 deleted 2026-09-01 per user).
- Raw run logs in `analysis/results/logs/` (gitignored by design); every
  distilled JSON/JSONL they produced is committed.
- TinyTeX at ~/.TinyTeX (multirow/booktabs/graphics/colortbl/xcolor) for table
  compile checks.
- venv ~/appcorr-env; known env quirks in project memory (torchao cpp-ext skip,
  realesrgan import break).

## Closed since the 20260827 handover (do not redo)

- Gemma4 port fully closed: is_causal fork fix (bitwise gate 3/3), budgeted
  interleaved option-b (crit 11-27% / total 131-159%), 12/12 accuracy campaign,
  uncapped arm quarantined as *.UNCAPPED. Ceiling-generate flicker gate
  documented in HANDOVER_to_GH200_20260827 addenda.
- Attention-term FLOPs omission fixed everywhere (patch_attention); all
  affected cells re-measured except 122B-chartqa (above).
