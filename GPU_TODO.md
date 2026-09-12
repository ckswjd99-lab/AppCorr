# GPU commands owed by the driver side -- ALL DONE 2026-09-10 (main session, GPU0)

Agent B's hand-off list (G5 on the real tower, first served interleaved sample, latency probe
arm, FLOPs fold-in) was run end to end on 35B; see `docs/memo/vllm_interleaved_design.md` §7
items 6-7 for the numbers and the two things the list did not anticipate:

1. G5 real tower: PASS (k=1 and 0.5 bitwise), `analysis/results/vllm_stream/interleaved_axis_gate.json`.
2. Served rows: `analysis/results/qwen_vllm_accuracy_il_pyr/` (V*, 191 rows, all arms paired on
   one interleaved server) and `analysis/results/qwen_vllm_accuracy_il/` (RealWorldQA, 765);
   `_probe40/` keeps the first 40-row probe. Rounds above `--max-num-seqs` are sub-batched.
3. Latency: `inprocess_latency.json` keys `qwen35_35b_il` / `qwen35_122b_il`. The anchor in
   item 3's note was WRONG (t_recv was stamped after the correct step); the interleaved arm now
   uses the streaming anchor (last band's pixel arrival) with a transport / drain+step
   (`last_correct_step_ms`) / hold-back decomposition.
4. FLOPs: `analysis/results/flops/qwen35_flops_il.json` (V*), `qwen35_flops_il_rwqa.json`,
   `qwen35_122b_flops_il.json` (decoder-only, vision half from the 35B run in the table).

Table: `analysis/experiments/make_eval_table.py --table interleaved` (streaming vs interleaved,
its own table until the main-table presentation is decided). Delete this file at merge.

## Added 2026-09-10 05:15 (after il_next_chain finished; GPU0 idle) -- need the user's go
- 122B hooked FLOPs re-measure with the `FP8Experts` hook fix (every hooked 122B number is
  short by the routed-expert term; memo item 10). Prepared: `$S/il_followups_NOT_LAUNCHED.sh flops`.
- 122B interleaved latency re-probe with `--max-num-seqs 1024` (256 cap => 58 ms x ceil(|P|/256)
  per round and queueing at k=1). Prepared: `$S/il_followups_NOT_LAUNCHED.sh probe1024`.
- Optional: 122B streaming k0.5/0.25 on the 40 probe rows (~1 min each) to fill the paired cells.
Delete this file at merge.

## Added 2026-09-10 (depth staging, user go) -- running: `$S/il35_staged_chain.sh`
- `--llm-schedule interleaved_staged` implemented (memo §7.11): 4B g6 (k=1 identity) / g6k (k<1
  differs from the MVP, as derived) PASSED. Chain: 35B g6/g6k gates -> served V*+RWQA staged arms
  k=1/.5/.25 -> d150 probe (`qwen35_35b_il_staged`) -> `--il-only` FLOPs fold -> table
  (`ils_*` columns). Comp. for the staged arm is the closed form of the ideal schedule.
- Next on GPU0 (user go, after the chain): PIECEWISE CUDA-graph correct step (gate vs eager,
  re-probe), then `$S/il_followups_NOT_LAUNCHED.sh probe1024`.

## Update 2026-09-10 14:02
- Staged chain DONE 13:51: 35B staged arms (V*/RWQA × k1/.5/.25) landed, no measurable
  staged-vs-unstaged accuracy difference (paired, all CIs cover 0); staged d150 probe folded;
  `interleaved_table_20260910.tex` v3 sent + mirrored (memo §7.11).
- CUDA-graph correct step: gate g7 PASS (graph == padded-eager bitwise; memo §7.12), default
  `APPCORR_CORRECT_CUDAGRAPH=1`. 35B step 46 -> 25 ms.
- RUNNING on GPU0: `$S/cg_chain2.sh` -> 35B il + ils graph re-probes (d150) -> 122B server
  mns1024 + il re-probe -> table. Eager latency kept under `*_eager` keys.
- After CG_CHAIN_DONE: regenerate tex (caption: eager sentence -> graph step ms; 122B 256-cap
  sentence -> 1024 result), mirror, send; memo §7.12 latency numbers.
- Still needs a go: 122B hooked FLOPs re-measure (FP8Experts fix); chain 2b tail; squash-merge.

## Update 2026-09-10 14:33 -- optimisation chain DONE
- CG_CHAIN_DONE 14:31: 35B il/ils + 122B (mns512; 1024 refused: 604 Mamba blocks) re-probed
  with graphs; table v4 sent + mirrored; memo §7.12 has the numbers. GPU0 idle.
- Nothing running. Next needs a go: 122B hooked FLOPs re-measure (FP8Experts fix, `$S/
  il_followups_NOT_LAUNCHED.sh flops`); chain 2b tail (TextVQA k0.50 2900/5000 -> ...);
  squash-merge of develop/vllm-interleaved-engine (tag first, drop this file).

## Update 2026-09-10 16:02 -- go "최적화 더 할 거 있냐? 있으면 계속 해봐라"
- DONE: rescan host-sync hoist (memo §7.13; correct step -11..-13 ms on every cell, 32/32
  paired) -> served re-probe folded (`*_prehoist` keys kept), table v5 sent + mirrored.
- DONE: fused hold-back (memo §7.14; `APPCORR_FUSE_HOLDBACK`/`APPCORR_DEFER_FINAL`, default on):
  4B + 35B g8 gates PASS (g2d bitwise == g2, g2f in band), profile final wall+hold-back
  24.6 -> 22.6 ms in-process.
- RUNNING on GPU0: `$S/fuse_reprobe.sh` (bash 12255; `$S/fuse_reprobe.out`): 35B il + ils
  -> 122B il served re-probes (d150) into `probe_*_fuse_d150`; previous keys kept as
  `*_hoistonly`. Then: fold, memo §7.14 RESULTS_PENDING, table v6, mirror, send.
- Still need a go: 122B hooked FLOPs re-measure; chain 2b tail; squash-merge.

## Update 2026-09-10 16:24 -- fused hold-back re-probe DONE
- FUSE_REPROBE_DONE 16:20: 35B il/ils + 122B il re-probed (`probe_*_fuse_d150`; previous keys
  `*_hoistonly`); memo §7.14 filled; table v6 sent + mirrored. GPU0 idle. Nothing running.
- LLM-side critical path is at the model floor (MoE weight stream ~9 ms + kernel floor ~7 ms
  + engine step ~4 ms + reply ~2 ms). Still need a go: 122B hooked FLOPs; chain 2b tail;
  squash-merge.

## Update 2026-09-10 17:05 -- 122B hooked FLOPs re-measure DONE (user go)
- `flops/qwen35_122b_flops_fixed.json` (all 7 datasets, one FP8 load, 8 min; needs ONLINE +
  `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1`, see memo §7.10). Gate F on the ceiling: closed form
  within 0.05 % of full - tower. Folded into `inprocess_flops.json["qwen35_122b"]` (old entry
  -> `qwen35_122b_prefix`) + `qwen35_122b_flops_il.json`; `IL_FLOPS_PENDING` cleared; table v7
  sent + mirrored (script/json/tex). GPU0 idle. Nothing running.
- OPEN: main table (`--table eval`) 122B compute cells changed but tex NOT regenerated: the
  three trees render different 35B k<1 accuracy cells (chain 2b rows) -- reconcile first.
- Still need a go: chain 2b tail; squash-merge.

## Update 2026-09-10 17:27 -- chain 2b DROPPED (user), interleaved-table extension RUNNING

- User decision: the 35B streaming k<1 accuracy campaign (chain 2b: TextVQA k0.50 at 2900/5000,
  then TextVQA k.25, RWQA, MMVP, CV-Bench, VSR, ChartQA) is abandoned, not resumed. Its landed
  rows stay in `analysis/results/qwen35_accuracy_pyr/` (Det/Count/V* k.50/.25 complete).
- RUNNING on GPU0: `$S/il_ext_chain.sh` (detached, markers in `$S/il_ext_chain.out`,
  logs `logs/vllm_stream/ilext_*.log`). Extends the interleaved table (`--table interleaved`)
  with TextVQA / InfoVQA / VisDrone Count / VisDrone Det, pyr filter, out dir
  `qwen_vllm_accuracy_il_pyr/`:
  A) 35B interleaved server (mns1024, c4, full datasets): floor+ceiling, interleaved and
     interleaved_staged at k=1/.5/.25 (streaming arms NOT run -- user: "Interleaved,
     depth-staged 만"), then d150 probes (keys qwen35_35b_il / _staged, new datasets merged
     per dataset into inprocess_latency.json).
  B) 35B hooked FLOPs with --il-rows -> `flops/qwen35_flops_il_ext.json`.
  C) 122B interleaved server (mns512, c2, `--samples 240` strided per dataset; over-length skip
     at max_model_len 8192 counted per arm), same arms + probes (keys qwen35_122b_il / _staged).
  D) 122B FLOPs: `--il-only` fold of textvqa/visdrone_* into `qwen35_122b_flops_fixed.json`
     (vision half from the stored `_split`, patched today) + hooked InfoVQA run (online,
     TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1) -> `qwen35_122b_flops_infovqa.json`.
  Expected ~8 h (35B) + ~1 h (FLOPs) + ~4 h (122B). `make_eval_table.py` already extended
  (IL_DATASETS + expected n + flops json map); regenerate `interleaved_table_20260910.tex`
  when the markers `IL_EXT_35B_DONE` / `IL_EXT_CHAIN_DONE` appear.
- Still OPEN: main eval table regeneration (35B k<1 rows differ per tree; now moot for the
  dropped 2b arms -- reconcile which tree's rows are canonical before regenerating);
  squash-merge of develop/vllm-interleaved-engine (tag first, delete this file).
