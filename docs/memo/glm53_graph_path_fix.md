# GLM-5.3-Flash correct-step CUDA-graph path: gate evidence and the two fixes (B200-8, 2026-09-14)

Working copy: /home/nxclab/glm53_lat (rsync of AppCorr-glm53 @ 5bfd557 + the edits below).
Deliverable diffs -- the shared tree's correct.py / glm53_indexer.py @ 5bfd557 with ONLY the two
fixes re-applied (correct.py +34 lines: persistent slot-mapping buffers; glm53_indexer.py +12: n =
min(n_batch, |P|) + tripwire); server.py untouched.  The gate/profile/probe runs above ran on the
08c97ce-based working copy whose capture_model wrap is equivalent to 5bfd557's, plus the same two
fixes; final sanity runs of the rebased files (graph arm, E6 ladder, MMVP 4 + chartqa 4): with the
`_SIDE` guard -> 0/4 + 0/4, 11.6 ms (defect 1 reproduced); guard removed -> MMVP 4/4 (41.1 ms),
chartqa 4/4 (49.4 ms); with the observability line/raise added -> MMVP 4/4 (42.7 ms), chartqa 4/4 (51.3 ms), server log
"[appcorr] capture_model: recurrent seam=kda indexer ops hooked=11" (glm53_final_sanity/)..
  b200-8_logs/glm53_graph_path_fix.correct.py.diff, glm53_graph_path_fix.glm53_indexer.py.diff.
The gate/profile/probe runs above ran with env-gated diagnostics present in the same files
(probes, sync points, out_stats, profiler hook -- all inert when their env vars are unset, which
they were for the gate and the probe); those blocks are removed from the diff; the pre-strip files
are kept as b200-8_logs/correct.py.with_diagnostics / glm53_indexer.py.with_diagnostics.
Env: /home/nxclab/appcorr-vllm-main-patched (vllm 0.1.1.dev65+g658c8131c, python3.11); the only env
patch in force is the indexer inference_mode fix (indexer_inference_mode_fix.diff). The MNNVL
two-shot all-reduce patch tried at 14:41 was NOT the cause and is REVERTED -- do not apply it.
Server config (E6): TP=2, gpu-mem 0.92, max-num-seqs 512, --cudagraph-capture-sizes 8 16 32 64 128
256 384 512, --cudagraph-mode PIECEWISE, tower on a third card. Flag: APPCORR_GLM53_CUDAGRAPH=1.

## What was wrong (three defects, found in this order)

1. Seams recorded after capture (fixed upstream in 5bfd557: capture_model wrap): the KDA patch and
   the kpool-indexer hook were installed lazily on the first execute_model, after the worker's
   CUDA-graph capture, so the captured eager-break callables were the stock functions. Replay ran
   the stock decode path on the corrected rows: 11 ms full-depth round, '!' x512 (token 0).
1b. (found at hand-over, 16:52) 5bfd557's `_capture_model` guards the install with `if _SIDE:`;
   `_SIDE` is a correcting request's side buffer, which cannot exist when the worker captures at
   init, so the committed wrap was a no-op and the rebased tree reproduced defect 1 ('!' rows,
   11.6 ms round).  The day's runs only worked because the private wrap installed unconditionally.
   Fix: install unconditionally; log "[appcorr] capture_model: recurrent seam=<flavour> indexer
   ops hooked=<n>" and raise if the decoder has the layers but nothing was installed.
2. Captured KV-cache write uses the runner's persistent slot buffer, not the forward context.
   `unified_mla_kv_cache_update` (and `unified_kv_cache_update` on the flash path) are registered
   custom ops WITHOUT @eager_break_during_capture, so they are inside the captured segments; the
   kernel keeps the slot-mapping pointer it saw at capture -- `input_batch.block_table[gid]
   .slot_mapping.gpu[:n_pad]` (gpu_model_runner._get_slot_mappings) -- and at replay writes ALL
   n_pad rows to whatever that buffer holds (the last stock step's slots).  The fresh per-layer
   tensors the correct step put in the forward context were ignored.  Symptom: right after the
   first sparse-MLA layer the request's own KV held exactly n_pad-P NaN rows (23/21/20 for
   P=41/43/44 at n_pad 64; 0 before the op), the NaN pad rows of the padded batch landing on the
   request's positions P..63.  At n_pad 128 (chartqa) the stale buffer tail was -1-padded from a
   small last step, which is why 128 "worked" and 64 did not (ladder 128+ -> MMVP real answers,
   ladder 64+ -> '!'; stock decode at ~48 concurrent, i.e. the size-64 graph, is fine: 82.00).
   Fix (correct.py, graph branch, unconditional): copy this round's slots into the persistent
   buffer for every KV group, fill [P:n_pad] with PAD_SLOT_ID (-1, skipped by the cache kernels),
   and hand those views to the forward context.  Same convention the runner uses; positions and
   inputs_embeds already followed it.
3. Indexer hook rewrote pad rows: glm53_indexer's correct branch used n = hidden_states.shape[0]
   (= n_pad under the graph), so pad rows (position 0, garbage) were rewritten into the pooled and
   tail caches every sparse layer.  Fix: n = min(n_batch, |P|), assert n_batch >= n, and
   rows_real/rows_batch recorded per layer in the round's stats (tripwire).

Why only GLM-5.3 could hit defect 2 (by construction, not luck): the families capture by different
mechanisms.  Qwen3.5 / GLM-4.6V servers run CompilationMode.VLLM_COMPILE with cudagraph_mode
FULL_AND_PIECEWISE and splitting_ops that include vllm::unified_kv_cache_update and
vllm::unified_mla_kv_cache_update (lat_adaptive_122b/server.log, lat_adaptive_glm46v/server.log): the
fx graph is split there, so the KV write runs eagerly BETWEEN the captured pieces and reads the live
forward context (our fresh per-layer slot mapping).  The GLM-5.3 graph servers run
CompilationMode.NONE with breakable cudagraphs (graph_gate_graph/server.log: 'mode':
<CompilationMode.NONE: 0>, splitting_ops: []), where only @eager_break_during_capture functions
break; the undecorated KV write stayed inside the captured segment with its capture-time pointer.
So the Qwen-family graph-mode rows are unaffected by construction (g7's bitwise g2cg == g2pe is
the confirmation), and no re-validation is needed.  The condition to check on any future server is
the compilation mode / splitting ops (and the rope_kvcache fusion variants
fused_rope_and_unified_kv_cache_update, fused_qk_norm_rope_and_unified_kv_cache_update,
fused_rope_unified_mla_kv_cache_update, which are the same class if those passes are enabled --
they are not on our servers: "Enabled custom fusions: norm_quant, act_quant, allreduce_rms"), not
the model name; the unconditional persistent-buffer fix covers both.  Scan of every registered
custom op in this nightly: the KV write is the only captured consumer of runner state on the models
we serve; attention metadata is read only by the decorated attention ops.

Dead ends, kept so nobody repeats them: stream-sync bisect (S0 control PASSED -> not a race);
MNNVL one-shot/two-shot, trtllm backend, VLLM_ALLREDUCE_USE_FLASHINFER=0 (all still NaN);
APPCORR_CONTIG_ROWS=0 (still NaN); workspace zeroing (still NaN).  The one-shot kernel at n_pad 64
was a correlate of the size, not the cause.

## Gate (final code; MMVP unified_staged k0.50 300 rows, chartqa k0.50 36 samples; A' band ref)
| arm (E6 server, same code, flag off/on) | MMVP unified k0.50 (300) | chartqa k0.50 (36) | correct t_ms by depth 0-10 / 0-22 / 0-34 / 0-45 (MMVP) | full-depth, chartqa (n_pad 128) |
|---|---|---|---|---|
| eager reference (flag off)   | 81.00 | 97.22 | 19.5 / 41.6 / 64.3 / **88.4** | 90.7 |
| graph (APPCORR_GLM53_CUDAGRAPH=1) | 79.67 | 97.22 | 19.6 / 41.6 / 64.1 / **41.3** | 50.6 |

Eager-arm spread of this MMVP arm across the day's four eager runs: 80.00 / 80.00 / 81.00 / 81.00
(and 82.33 in the table run at mns 64); pad-eager 78.67 was accepted as in-band earlier.  Only the
full-depth (final) round dispatches a graph; the partial-depth rounds stay eager (2 ms/layer).
Answers are real and terminate normally (3-token MMVP answers, chartqa EOS at 2-4 tokens); the
pre-fix "runs to the token cap" residual is gone with the stale-slot writes.
Rows: graph_gate_eager/rows, graph_gate_graph/rows (16:02-16:14).

## Working-step profile (worker-side, final full-depth round)
Worker-side torch.profiler inside the TP workers (APPCORR_CORRECT_PROFILE_ROUND), final full-depth
round, chartqa k0.50 unified_staged, 4 requests, probes/debug OFF (timings include profiler overhead;
the un-profiled step is the gate's t_ms above).

| arm | P / n_pad | wall (profiled) | kernel launches | TP all-reduce (twoshot x90) | MoE | KDA re-scan | elementwise launches |
|---|---|---|---|---|---|---|---|
| eager (13:0x)            | 100-110 / -   | 115-140 ms | 7,350-8,680 | 26.3 ms (peer wait)  | 13.2 ms | 362 launches | 4,063 |
| graph, fixed, n_pad 128  | 101 / 128     | 62 ms      | 6,594       | 0.7-4.6 ms           | 19.3 ms | 362 launches | 3,974 |
| graph, fixed, n_pad 64   | 44 / 64       | 57 ms      | 6,717       | 0.5-3.2 ms (one-shot x90) | 11.9 ms | 362 launches | 4,044 |

Reading: the captured segments (MoE, dense GEMMs, mHC, norms; ~2.2k launches in the old-seams
run where nothing else ran) replay as graphs; the remaining launches are the eager-break hooks
(KDA re-scan 34 layers, indexer rewrite 11 layers, their index/scatter elementwise work) -- the
next optimisation target.  chartqa 4/4 at n_pad 128; MMVP 4/4 ("(a)"/"(b)", 3 tokens, stop) at
n_pad 64 -- the size that was broken -- with finite output (out_stats nan=False), the one-shot
all-reduce kernel and the MultiCtasKv sparse-attention variant both present and harmless.
Diagnostic signature worth remembering: with NaN inputs the fused MoE collapsed to 2.4 ms per
round (routing on NaN selects almost nothing); with the fix it does real work again (11.9 ms at
P=44, 19.3 ms at P=101).

## Four-dataset latency probe (flag on)
Comparison rule: the graph server runs max-num-seqs 512 (the E6 memory budget) while the eager row
was measured at 1024.  chartqa (~101 rows), textvqa (~108) and visdrone_count (~103) sit far below
both caps -> clean graph-vs-eager comparison; those three carry the claim.  V* k0.50 / auto50 bands
(~585 / ~820 rows) split into sub-batches at 512 where the eager row's 1024 did not, so the V* 0.50
difference is cap-vs-cap, not eager-vs-graph -- both V* 0.50 cells stay WITHHELD as agreed.
Runner: latency_probe.py --family glm53 --keeps 0.50 auto:θ50 0.25 auto:θ25 --samples 36 --warmup 4
--push-delay-ms 150 (d150 anchor; total_* None by design), n=32 per cell, key
glm53_ilu_adaptive_tp2graph_fixed (eager row kept under glm53_ilu_adaptive).  crit = ms, median.

| dataset | full (ceiling) | eager crit k0.50 / auto50 / k0.25 / auto25 | graph (fixed) crit k0.50 / auto50 / k0.25 / auto25 |
|---|---|---|---|
| chartqa        |  98.0 | 106.5 / 105.4 / 102.2 / 104.2 | **65.6 / 64.1 / 65.1 / 64.3** |
| textvqa        | 114.0 | 103.9 / 106.9 / 105.2 / 103.6 | **71.8 / 68.7 / 65.7 / 61.9** |
| visdrone_count | 124.0 | 106.3 / 105.0 / 105.6 / 105.4 | **73.0 / 68.4 / 66.9 / 59.9** |
| vstar          | 390.7 | [135.7 / 137.8 withheld] / 120.8 / 117.8 | [186.4 / 162.4 withheld, cap 512] / **99.5 / 86.6** |

On the three clean datasets crit drops 35-43% vs the eager row and is now BELOW the full-image path
(chartqa 65 vs 98, textvqa 62-72 vs 114, visdrone 60-73 vs 124); adaptive stays within ~4 ms of
its fixed-keep pair (auto25 is the fastest cell everywhere).  V* 0.25/auto25: 120.8/117.8 -> 99.5/86.6.
Pass 1 16:25-16:40; ceiling pass 16:40-16:47 (full: chartqa 100.4, vstar 388.5, textvqa 116.4,
visdrone_count 119.2 ms; k1.00 crit 101.2 / 259.6 / 121.8 / 127.7).
Result json: /home/nxclab/glm53_lat/analysis/results/latency/inprocess_latency.json (private copy).

Evidence dirs (b200-8_logs): glm53_t6/server_SSLOTS.log ([nan-probe-slots]), glm53_nan_probe_attempt3/
(deep probe), glm53_size_test/, glm53_stock_batch64/, glm53_sync_bisect/, glm53_twoshot_test/,
glm53_trtllm_test/, glm53_nofiar_test/, glm53_contig0_test/, glm53_t2_zerows/, graph_gate_*/.


# STEP 2 (2026-09-14 evening): host-bound cost of the correct step's hooks

Per-region profile (worker-side, graph arm, chartqa P=101 / n_pad 128, final full-depth round;
launches = kernels launched from the host under the range, CUDA ms their device time, CPU ms the
range's host time incl. children):

| region | STEP 1 launches / CUDA / CPU | after (A) | after all levers |
|---|---|---|---|
| indexer.rewrite (11 layers) | 1,397 / 3.33 / 21.6 | 1,192 / 2.71 / 18.3 | 918 / 2.00 / 17.7 |
| indexer.topk               | 66 / 0.14 / 1.1      | 66 / 0.14 / 1.0    | 66 / 0.13 / 1.1 |
| rescan (34 KDA layers)     | 321 / 0.81 / 19.4    | 321 / 0.81 / 19.1  | 306 / 0.76 / 18.9 |
|   rescan.chunk             | 219 / 0.54 / 13.5    | 219 / 0.54 / 13.1  | 204 / 0.51 / 12.6 |
| kda.scatter / write / final | 102 / 0.65 / 1.8 ; 102 / 0.20 / 1.8 ; 34 / 0.05 / 4.4 | same ; same ; 34 / 0.06 / 6.1 | same ; same ; 4 / 0.00 / 2.8 |
| appcorr.layers (whole round) | 2,143 host launches / 58.9 CPU (profiled wall 62.6) | 1,938 / 56.4 (60.0) | 1,619 / 53.2 (57.0) |

Levers, each checked before adoption:
(A) per-round index plan: pools / rows / tail positions / tail j computed once on the host from
    the round's positions and uploaded once (`round_plan`), `n_valid` from `max_seq_len` (no
    per-layer `.item()`), `slots` and the k_cache page/off/idx cached per LAYER-CACHE identity
    (block_row ptr, num_states, bt_block_size) -- never on the assumption the sparse layers share
    a KV group.  Pure caching, no numerics change.
(B) `cu_seqlens=None` for the single-sequence KDA re-scan: the fla wrapper recomputed
    prepare_chunk_indices (a .tolist() host sync + H2D) on every call.  Bitwise identical to
    cu_seqlens=[0, L] on identical random inputs at the real shapes (H=32 local, D=128; L = 44,
    101, 585, 1192; outputs and final state) -- standalone check, not the accuracy gate.
(C) `pool_compress` exact-order vectorisation: elementwise work for all kpool slots at once, the
    two fp32 accumulations keep the loop's chain ((x0+x1)+x2)+x3, max is order-free.  Bitwise
    identical to the loop form on random inputs (P = 1/7/31/146, round_scale both), fp8 entry
    and scale.  Loop form kept as `_pool_compress_loop` (reference).
(D) tail `j` and k_cache index math cached in the plan (part of A).
(E) `_kda_write_back` scatter index cached per (request, block): 34 H2D per round -> 1.

Gate after levers (A)-(E) (v2; same E6 server config; rows graph_gate_*_v2):

| arm | MMVP unified k0.50 (300) | chartqa (36) | full-depth correct t_ms MMVP / chartqa |
|---|---|---|---|
| eager reference (v1 → v2) | 81.00 → 80.67 | 97.22 → 97.22 | 88.4 → 83.7 / 90.7 → 86.8 |
| graph (v1 → v2)           | 79.67 → 79.33 | 97.22 → 97.22 | 41.3 → 36.1 / 50.6 → 43.7 |

Accuracy unchanged within the arm's eager spread (80.0-81.0 across the day's eager runs; graph
arms 79.33-79.67); the levers took ~5 ms (MMVP) / ~7 ms (chartqa) off the graphed final round and
~4-5 ms off the eager one.

(F) stock kpool kernels for the rewrite (v3): `kpool_compress_and_write_cache` (softmax pool +
Hadamard-128 + ue8m0 fp8 + scale + cache write in ONE launch) and `kpool_seed_tail_cache` replace
the torch mirror (`pool_compress` + `k_cache_write` + `tail_write`, ~60 launches/layer).  The
torch mirror was written to match these kernels; on 20 x 146 random pools (2.9M fp8 entries) they
agree except ONE fp8 rounding tie (scales identical), so the kernel path is the stock prefill's own
numerics.  Flag `USE_STOCK_KPOOL_KERNELS` (CUDA + plan only; the torch path stays for CPU/tests).
Per-region profile with (F) (v3, same request set):

| region | STEP 1 | v2 (A-E) | v3 (A-F) |
|---|---|---|---|
| indexer.rewrite (11 layers) launches / CUDA ms / CPU ms | 1,397 / 3.33 / 21.6 | 918 / 2.00 / 17.7 | **304 / 0.71 / 10.2** |
| rescan (34 layers)            | 321 / 0.81 / 19.4 | 306 / 0.76 / 18.9 | 306 / 0.76 / 17.7 |
| kda.final                     | 34 / 0.05 / 4.4   | 4 / 0.00 / 2.8    | 4 / 0.00 / 2.6 |
| whole round: host launches / host ms (profiled wall) | 2,143 / 58.9 (62.6) | 1,619 / 53.2 (57.0) | **1,005 / 43.2 (46.8)** |

Remaining host cost is the fla KDA re-scan wrapper (34 calls x ~6 Triton launches at ~60 us each
= ~12 ms) and the KDA scatter/write index work; the re-scan cannot be batched across layers
because each layer's re-scan needs THAT layer's live activations of the round (scattered into
the side buffer just before), so the floor without kernel work is ~40 ms of host time per round.
Gate with (F) (v3; rows graph_gate_*_v3):

| arm | MMVP unified k0.50 (300) | chartqa (36) | full-depth correct t_ms MMVP / chartqa |
|---|---|---|---|
| eager reference (v1 → v2 → v3) | 81.00 → 80.67 → 80.67 | 97.22 → 97.22 → 94.44 | 88.4 → 83.7 → 80.3 / 90.7 → 86.8 → 82.6 |
| graph (v1 → v2 → v3)           | 79.67 → 79.33 → 78.67 | 97.22 → 97.22 → 94.44 | 41.3 → 36.1 → 32.3 / 50.6 → 43.7 → 40.3 |

v3 graph MMVP 78.67 is 2.0 pt under its eager run (2 rows of 300; the arm's run-to-run spread on
this model is ~1 pt eager-vs-eager and the graph arms have sat 1.3-2.0 pt under eager all day),
and chartqa equals eager.  Two more MMVP samples of the v3 graph arm on a fresh server
(graph_gate_graph_resample/rows, rows2): 79.33, 79.33 -> v3 graph arm 78.67 / 79.33 / 79.33 (mean
79.1) against v1 79.67 and v2 79.33: the ~1.3-1.7 pt offset of the graph arm under the eager
reference is a property of the graph path present since v1, not of (F).  (F) adopted.
Four-dataset probe with (A)-(F) (v3; key glm53_ilu_adaptive_tp2graph_fixed, E6, d150, n=32):

| dataset | full | eager crit k0.50 / auto50 / k0.25 / auto25 | graph v1 (fix only) | **graph v3 (fix + hooks)** |
|---|---|---|---|---|
| chartqa        | 100.4 | 106.5 / 105.4 / 102.2 / 104.2 | 65.6 / 64.1 / 65.1 / 64.3 | **55.2 / 54.2 / 54.9 / 55.8** |
| textvqa        | 116.4 | 103.9 / 106.9 / 105.2 / 103.6 | 71.8 / 68.7 / 65.7 / 61.9 | **62.1 / 59.5 / 58.0 / 55.0** |
| visdrone_count | 119.2 | 106.3 / 105.0 / 105.6 / 105.4 | 73.0 / 68.4 / 66.9 / 59.9 | **62.0 / 60.0 / 57.7 / 51.9** |
| vstar          | 388.5 | [withheld] / 120.8 / 117.8 | [withheld] / 99.5 / 86.6 | [162.5 / 142.9 withheld, cap 512] / **88.7 / 87.1** |

On the three clean datasets crit is now 45-51% below the eager row and ~45-55% of the full-image
path; the hook levers added 9-11 ms on top of the graph-path fix.  Ceiling pass 20:15-20:22 (full: chartqa 100.6, vstar 406.7, textvqa 117.4, visdrone_count 123.1;
k1.00 crit 102.8 / 253.5 / 126.1 / 127.0).  Result json copies: b200-8_logs/inprocess_latency.v1.json
(graph-path fix only), .v3.json (fix + hooks); private working json
/home/nxclab/glm53_lat/analysis/results/latency/inprocess_latency.json.

Deliverable (STEP 2): diffs of the working copy against the shared tree AS OF 17:01 (which
already carries the graph-path fix and the capture observability landed from the first hand-over)
-- b200-8_logs/glm53_step2_hooks.correct.py.diff (+26 lines: regions, cu_seqlens=None, cached
scatter index; one line of the landed slot-mapping block differs textually from mine,
`buf[P:n_pad].fill_(-1)`, reconcile on landing) and glm53_step2_hooks.glm53_indexer.py.diff
(+133 lines: round_plan, plan-aware rewrite_rows, store(hi), pool_compress v2 + loop reference,
k_cache_index/write_at, tail_write(j), stock-kernel path + flag); server.py untouched.  The record_function regions
(appcorr.rescan.*, appcorr.kda.*, appcorr.indexer.*) stay in as permanent, no-op-without-profiler
instrumentation; the env-gated profiler call-site wrap and the hook module are stripped (kept as
b200-8_logs/correct.py.v3_with_profiler, _correct_profile_hook.py.step2).  Final sanity on the
stripped files: MMVP 4/4 (34.4 ms), chartqa 4/4 (41.0 ms), server log
"[appcorr] capture_model: recurrent seam=kda indexer ops hooked=11" (glm53_final_sanity/, 20:22).


# LATTICE objective (2026-09-14 late evening): recompute count == captured graph size

Goal: a correct round's |P| lands exactly on a captured size (no pad rows), with the pad filled by
the next-highest-scoring groups instead of zeros -- latency-free accuracy.  Opt-in; the default
`bucket_quota` (shared by all four models) is untouched.

Design (appcorr/models/qwen_vl_axis.py `bucket_quota_lattice`, `axis.pscore_lattice`):
  q0 = bucket_quota(n_over, n_band, bucket)   (the standing 1/bucket rule; never below it)
  L  = min{l in lattice : l >= q0 + extra}    (the size the round would pad to)
  q  = min(n_band, L - extra)                 (no lattice point -> q0; sub-batches as today)
applied to the LAST band only -- the only round that is graphed (partial-depth frontier walks are
eager) and the one that carries the text suffix `extra = (seq-1) - (lo + n_groups)`.  Monotone
non-increasing in theta and >= q0 by construction (checked on n_band 7/42/85/600).
Calibration: threshold_sim.py --lattice --suffix-json applies the identical rule (imports it), with
per-image suffixes joined by dataset index (b200-8_logs/glm53_text_suffix_by_image.json; 36/36
npz image ids matched on every dataset).  Drivers: --pscore-lattice (accuracy driver, arm tag
`_lat`, row field pscore_lattice) and latency_probe.py --pscore-lattice (keys `auto<theta>_lat`).
Ladder: 8 16 32 48 64 80 96 112 128 160 192 224 256 320 384 448 512 640, --max-num-seqs 640
(V*'s fixed k0.50 band ~585 rows fits; the two withheld V* 0.50 cells become measurable).

Recalibrated thetas (mean realised k at target, 36-image npz, rms, eager pscore, bucket 8):

| dataset | theta50 old -> lattice | theta25 old -> lattice |
|---|---|---|
| chartqa        | 0.018935 -> 0.0203064 | 0.104678 -> 0.111496 |
| vstar          | 0.057005 -> 0.0575774 | 0.105332 -> 0.106838 |
| textvqa        | 0.039308 -> 0.0402505 | 0.0858648 -> 0.0871182 |
| visdrone_count | 0.0444106 -> 0.0453187 | 0.0829471 -> 0.0849944 |
| mmvp           | 0.043362 -> 0.053443  | 0.075766 -> 0.102235 |

(The lift adds rows to the last band, so the global cut moves up to hold the mean k.)
Accuracy gate, graph server (dense ladder, mns 640), rows lattice_gate/rows:

| arm (same target k) | MMVP unified (300) | chartqa (36) | realised k (MMVP) | final round on a captured size | pad rows (mean) | corrected rows per band, median (last incl. suffix) |
|---|---|---|---|---|---|---|
| auto50 old rule (theta 0.043362)            | 83.00 | 97.22 | 0.533 | 17/300  | 8.0 | 6 / 10 / 10 / 42 |
| auto50 lattice, recalibrated (0.053443_lat) | 81.00 | 94.44 | 0.544 | 210/300 | 1.7 | 4 / 7 / 8 / 48 |
| auto25 old rule (0.075766)                  | 82.00 | 94.44 | 0.264 | -       | -   | 2 / 4 / 3 / 37 |
| auto25 lattice, recalibrated (0.102235_lat) | 81.67 | 94.44 | 0.307 | -       | -   | 2 / 2 / 2 / 48 |

Mechanism (why "same target k" is NOT the monotone case on short-image prompts): MMVP's final round
is dominated by the text suffix (median 45 rows) and its last band holds only a few groups, so the
lattice lift can add at most those few; recalibrating theta to hold the mean k then REMOVES rows
from bands 0-2 (6/10/10 -> 4/7/8) to pay for them -- a redistribution, -2.0 pt at auto50 (6 rows of
300), -0.33 at auto25 (where the lattice floor keeps k at 0.307 > 0.264 anyway).  The monotone
claim ("the pad filled with real rows is free accuracy") is the OLD theta + lattice
(lattice_gate_oldtheta/rows): auto50 old theta + lattice = 81.67 (k 0.623, bands 6/10/10/48,
192/300 on a captured size) vs 83.00 old rule (k 0.533); auto25 old theta + lattice = 80.33 (k
0.398) vs 82.00 (k 0.264).  So on MMVP the lattice arms sit 0.3-2.0 pt UNDER the old rule with
equal or MORE corrected rows -- no accuracy gain is visible; the deltas are at or just past the
model's ~1 pt A-vs-A noise (the old-rule auto50 draw of 83.00 may itself be high; a second sample
of that pair is queued behind the probe).  ChartQA 36: old 97.22 / 94.44 vs lattice 94.44 / 94.44
(one row).  Latency probe, lattice mode (dense ladder + 640, mns 640, lattice thetas; key
glm53_ilu_adaptive_tp2graph_lat) against v3 (8-size ladder, mns 512, old thetas):

| dataset | v3: k0.50 / auto50 / k0.25 / auto25 | lattice: k0.50 / auto50_lat / k0.25 / auto25_lat |
|---|---|---|
| chartqa        | 55.2 / 54.2 / 54.9 / 55.8 | 54.0 / 55.3 / 51.9 / 53.0 |
| textvqa        | 62.1 / 59.5 / 58.0 / 55.0 | 56.8 / 61.0 / 54.7 / 54.5 |
| visdrone_count | 62.0 / 60.0 / 57.7 / 51.9 | 59.0 / 62.6 / 55.2 / 54.3 |
| vstar          | [162.5 / 142.9 withheld] / 88.7 / 87.1 | **103.6** (no longer sub-batched at 640) / [146.2 withheld: auto50 band ~820 > 640] / 90.0 / 85.8 |

Reading: the lattice auto arms are a wash (-1 to +3 ms vs the old auto arms -- they correct a few
more rows, and the round is host-bound, so the removed pad rows buy little); the DENSER LADDER helps
the fixed keeps by 1-5 ms; the 640 point makes the V* fixed-k0.50 cell measurable (103.6 ms, was
162.5 sub-batched at 512).  Ceiling pass 21:20-21:27 (full: chartqa 70.4 [unexplained drop from the 98-101 of every earlier
pass; recorded, not interpreted], vstar 389.2, textvqa 116.7, visdrone_count 119.9).  Result json
copy: b200-8_logs/inprocess_latency.lat.json (key glm53_ilu_adaptive_tp2graph_lat).

Deliverables (lattice objective; diffs against the shared tree @ 90be55a): b200-8_logs/
glm53_lattice.qwen_vl_axis.py.diff (+36: bucket_quota_lattice, axis.pscore_lattice, last-band
application with the text suffix, stats field), glm53_lattice.qwen_vllm_accuracy.py.diff (+11:
--pscore-lattice, arm tag _lat, row field), glm53_lattice.latency_probe.py.diff (+11: forwarding,
key tag), glm53_lattice.threshold_sim.py.diff (+32: --lattice, --suffix-json, last-band rule);
b200-8_logs/theta_lattice/thetas_lattice.json (old and lattice thetas for the five datasets, the
ladder, the score settings) and glm53_text_suffix_by_image.json.  The default rule and every
existing theta are untouched (opt-in only).
Second sample of the auto50 pair (lattice_gate_resample2/rows): old rule 82.67, old theta +
lattice 82.00 -> two samples each: 83.00 / 82.67 (old rule) vs 81.67 / 82.00 (old theta +
lattice), means 82.8 vs 81.8.

VERDICT (lattice objective): the pad rows are removed as designed (final round on a captured size
17/300 -> 210/300, pad rows 8.0 -> 1.7 on MMVP), but the promised free accuracy does not appear on
GLM-5.3: every lattice arm sits 0.3-2.0 pt under the old rule on MMVP (recalibrated and old-theta
alike, k equal or higher), i.e. inside the model's ~1 pt A-vs-A noise but never above it, and
chartqa 36 is one row lower; the latency effect of the lattice itself is a wash (-1..+3 ms) because
the graphed round is host-bound.  Recommendation: keep the old rule as the default (no campaign
re-run); land the opt-in code and the thetas for the record; ADOPT the denser ladder with the 640
point and --max-num-seqs 640 for GLM-5.3 servers -- it costs 0.26 GiB of capture memory (3.18 ->
3.44 GiB), buys 1-5 ms on the fixed keeps, and makes the withheld V* fixed-k0.50 cell measurable
(103.6 ms; the auto50 band ~820 rows still exceeds 640 and stays withheld).


# LATTICE, powered re-test (2026-09-14 night, B200-6): the payoff is the slope times the lift

The first lattice test ran on GLM-5.3 MMVP -- floor 81.33 / ceiling 84.67, a 3.33 pt gap at
p=0.099 on n=300. The whole dynamic range there is smaller than what 300 rows can resolve, so
that experiment could not have answered the question either way. Re-ran it on the cell with the
most headroom of the 40 in the table, ranked by gap x n:

    35B ChartQA: floor 47.52 / ceiling 89.08 = 41.56 pt, n=2500, 1070 floor->ceiling flips.

Design: SAME theta (0.027817) in both arms, one server session, arm A standing 1/8-bucket rule
then arm B lattice, ladder 8..1024 at --max-num-seqs 1024. Images whose count does not move are
the internal control.

CONTROLS (both clean):
  run-to-run, same theta, different box/engine/date: 87.72 -> 87.68, net -1 flip, p=1.00,
      realised k identical to 4 decimals (0.4914). 35B is repeatable; GLM-5.3 TP=2 is not.
  k-unchanged subset (205 images): 0 flips, 0 differing predictions.

MONOTONICITY holds exactly as designed: 2500/2500 images have lattice k >= standing-rule k,
2295 strictly greater, none lower. Recomputing MORE is what the rule does.

RESULT: mean k 0.4914 -> 0.5077 (+1.64 points of k). Accuracy 87.68 -> 87.64, net -1 flip,
p=1.00, d = -0.04 pt -- exactly the run-to-run noise. Both arms sit 1.40 / 1.44 pt under the
ceiling (p=0.0014), so 35 net flips of headroom were available and the lattice took none.

WHY, quantitatively. Accuracy against realised k on this cell:
    k 0.000  47.52        slope over the interval to the next point
    k 0.255  83.64          +141.7 pt per unit k
    k 0.491  87.72           +17.3 pt per unit k
    k 1.000  89.08            +2.67 pt per unit k
At the k the adaptive arm operates at, the curve has saturated to 2.67 pt per unit k. The lattice
grants 0.0164 of k. Expected payoff 2.67 x 0.0164 = +0.044 pt = ONE flip in 2500. Measured
-0.04 +- 0.04 pt. Prediction and measurement agree; there is no missing effect to find.

So the lattice's payoff is (local slope) x (extra k it grants), and the two factors are
anti-correlated across cells. The lift is large only where the last round is dominated by the
text suffix and the band is small (GLM-5.3 MMVP: +9 points of k) -- which is also where the
accuracy curve is flat. Where the curve is steep (low k), the round is large and the lift is
proportionally tiny. A DENSER ladder, which is what we built to cut padding waste, makes the
lift smaller still: the two goals are in direct tension, and the efficiency one won by
construction. Raising k itself is the only lever that moves this cell (k 0.49 -> 1.0 is worth
1.40 pt); the lattice reaches 1.6 % of the way there.

Default unchanged (`pscore_lattice = None`). Rows: analysis/results/
qwen_vllm_accuracy_il_pyr_adaptive/lattice_ab/ (gitignored).
