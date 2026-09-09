# Where the Qwen streaming vision pass spends its time (and what L1-L3 bought)

**Status date:** 2026-09-08
**Branch:** `develop/vision-correct-perf` (worktree `AppCorr-vcperf`)
**Scripts:** `analysis/experiments/qwen_correct_profile.py` (this table),
`analysis/experiments/qwen_axis_snapshot.py` (bitwise gate + per-arm wall),
`analysis/experiments/qwen_axis_cpu_unittest.py` (CPU bitwise + FLOPs identity vs the old tree)
**Raw logs:** `logs/vllm_stream/profile_122b_{old,new}.log`, `snap_{122b,7b}_{before,after}.log`

## Why

The 122B-FP8 vLLM campaign's streaming arm runs at 3.4-3.8 samples/s against 7.7-10/s for the
one-shot arms, and the driver's per-sample `t_vision_ms` said the HF-side vision pass
(`QwenVLStreamingAxis.streaming_forward`) is 4-9x the one-shot tower. The plan
(`~/.claude/plans/moonlit-foraging-frost.md`) read that as launch/sync-bound and lined up L1-L3
(sync-free single-segment correction, rows-only correction, prepare-once). This memo is the
Step 0 measurement, taken AFTER L1-L3 were written but on both trees, so it shows what they
bought and what is actually left. Vision-only 122B tower (bitwise the campaign's, `vision_only.py`),
GPU0 beside the idle campaign server, one RealWorldQA image resized to 2109 and 4816 image
tokens (= 8436 / 19264 tower rows), g=4, level 2 pyr base, NullSink, median of 5.

## Gates (all on the GPU towers, old tree -> new tree)

| gate | result |
|---|---|
| 122B tower, 32 RWQA samples, ceiling/floor embeds + every streaming chunk (bf16 bits, M-RoPE, delta), keep 1.0 & 0.5, `positions_mode=check` | **32/32 identical** |
| Qwen2.5-VL-7B tower (windowed, multi-segment correct path), same | **32/32 identical** |
| CPU tiny models, both forks, 2 grids x 2 seeds x g{1,4} x keep{1.0,0.5}, + fast-vs-slow switches | **36/36 identical** |
| `appcorr.flops` per-stage FLOPs, both forks, g{1,4} x keep{1.0,0.5} | **identical** (prepare/vision_base/vision_correct/merge/llm_prefill to the FLOP) |

## Wall time (ms), 122B tower

| image tokens | one-shot tower | streaming k=1.0 old | **new** | streaming k=0.5 old | **new** | received-attn alone |
|---:|---:|---:|---:|---:|---:|---:|
| ~1300 (RWQA median, snapshot) | 32 | 107 | **84** | 214 | **189** | - |
| 2109 | 59.5 | 154.7 (2.6x) | **130.8 (2.2x)** | 430.1 (7.2x) | **397.7 (6.7x)** | 287.8 |
| 4816 | 226.4 | 536.3 (2.4x) | **500.3 (2.2x)** | 1381.7 (6.1x) | **1348.4 (6.0x)** | 947.8 |

Per stage, k=1.0 @2109 (bracketed by synchronizes; sum matches the free-running wall within 1 ms):

| stage | old | new |
|---|---:|---:|
| prepare (patch-embed x2, grid prep) | 4.7 | 3.9 |
| vision_base (full-depth approx of the base image) | 57.7 | 57.7 |
| vision_correct x4 | 92.2 (22.7 / 22.6 / 24.1 / 22.8) | **68.7 (16.9 / 17.1 / 17.4 / 17.2)** |
| merge x4 | 0.3 | 0.4 |

Host-side inventory of one k=1.0 pass @2109 (torch.profiler):

| | old | new |
|---|---:|---:|
| kernel launches | 6346 | **3815** |
| `cudaStreamSynchronize` | 356 | **12** |
| `aten::nonzero` | 217 | **1** |
| `aten::item` | 121 | **5** |
| GPU idle (wall - summed kernel time) | 16.8 ms (11%) | **4.7 ms (4%)** |

## What the GPU is doing (k=1.0, new tree, raw kernel buckets)

| bucket | @2109 | @4816 |
|---|---:|---:|
| attention (`pytorch_flash::flash_fwd_kernel`, 135 = 27 layers x (1 base + 4 rounds)) | 78.1 ms (62%) | 398.4 ms (79%) |
| elementwise / norm / residual | 25.5 (20%) | 52.8 (10%) |
| GEMM (`nvjet_sm100_*`, 3 shapes) | 14.4 (11%) | 28.6 (6%) |
| gather / scatter / index (KV scatter, row gathers, cat) | 7.0 (6%) | 14.6 (3%) |

## Reading

1. **The plan's diagnosis was wrong for these image sizes.** The old code was already 89-96%
   GPU-busy; the 356 syncs and 6.3k launches cost 17-20 ms, not the 100+ ms the driver's
   numbers suggested. L1-L3 removed nearly all of that (syncs 356 -> 12, launches -40%, idle
   11% -> 4%), which is worth 15% @2109 and 7% @4816 -- real, bitwise, and now at the floor.
2. **k=1.0 streaming is compute-bound at ~2.2x the one-shot tower, and that is its FLOP count.**
   The base approx is one full tower; the four correction rounds attend Q_r x T with
   sum_r Q_r = T, i.e. exactly one more full attention plus one more set of per-row GEMMs. At
   these token counts the tower is attention-dominated (62-79% of kernel time; T^2 at 8-19k
   rows), so the arm is ~2 towers of attention. No in-pass lever below the kernel level is
   left: the only remaining non-attention share is 20% elementwise (L7 fusion, opt-in, not
   bitwise) and the ~4% idle.
3. **The 4-9x the driver measured is therefore driver-side or contention, not the pass.**
   The pass alone is 2.2x. Candidates, each now addressed in this branch but not yet measured
   end-to-end (needs the engine, i.e. after the chain): the 10 blocking `.cpu()` pushes per
   sample (now async D2H on a side stream), `get_rope_index` (now closed-form, no sync), the
   PIL image crossing the DataLoader queue (gone), the 250 MB fp32 `image_embeds` copy (skipped
   with a sink), and the engine sharing GPU0 (one-shot arms see the same contention and run
   7.7-10/s). If the A/B leaves streaming below ~5/s, L6 `--shard 2` overlaps two towers.
4. **keep<1 is entirely `_received_attention`: 288 ms @2109, 948 ms @4816 = 4-5x the whole
   tower**, unchanged by L1-L3 (as expected -- L4 was deferred pending this number). It is the
   chunked `[H, 1024, T]` bf16 score matrix per layer: bf16 GEMM (54 ms), bf16 softmax over
   8436-wide rows (109 ms), `.float()` copy (82 ms), fp32 reduce (23 ms), 243 chunks x 27
   layers. Memory traffic, not FLOPs (~16 B/elem x H x T^2 x 27). This is the number to
   attack next, and it is the paper-relevant one: a keep=0.5 arm that is 3x slower than
   keep=1.0 in wall time contradicts its own FLOP column.

## L4 plan (from the profile, in order of numerics risk)

- L4a (exact, cheap): fold the `.float()` + `sum(dim=1)` into `sum(dim=1, dtype=float32)`
  (drops the 82 ms copy + one fp32 read); check bitwise on the gate set -- the reduce tree may
  differ from the fp32-input one, in which case it is fp32-reassociation-only.
- L4b (near-exact, Triton): two-pass kernel over (Q, K): pass 1 row-max/lse, pass 2 column
  sums of P, both reproducing the reference's roundings (S rounded to bf16, scaled in bf16,
  P rounded to bf16 before the fp32 column sum); only the fp32 summation order differs.
  Expected ~2 QK^T GEMM passes (~30-40 ms @2109 vs 288). Gate: selection identity
  (`corrected_groups` + chunk hashes) on 192 RWQA + 448 VisDrone; ship opt-in if any group
  flips.
- NOT: bf16 accumulation or fp32-unrounded P by default -- changes the score itself.

Status (2026-09-08 evening): L4a is in (`RECV_ATTN_FUSED_SUM` in qwen35/vision/attention.py,
CPU FLOPs gate PASS; GPU bitwise gate queued for gap 2). L4b is written as
`qwen35/vision/recv_attn_triton.py` (`_rowstats_kernel` + `_colsum_kernel`, BM=BN=64, D padded to
128, libdevice expf/div_rn so p matches the softmax epilogue bit-for-bit given equal m, l) and wired
behind `APPCORR_RECV_ATTN=triton` (default stays torch). Gap-2 hook runs, in this order: L4a
bitwise vs `snap_122b_before.json`; kernel unit test (`qwen_recv_attn_triton_test.py`: numerics +
ms); RWQA-192 torch-vs-triton snapshot; keep-0.5 re-profile torch vs triton; VisDrone-256
torch-vs-triton (the remaining 192 VisDrone samples after the chain). Numbers land in the tables
above when the hook finishes.

## Gap-2 results (2026-09-08 21:23-21:35, 122B tower, server idle on the same GPU)

- L4a bitwise gate: PASS 32/32 vs the pre-L4a snapshot (keep 1.0 and 0.5).
- L4b selection-identity gate (`APPCORR_RECV_ATTN=triton` vs torch, same tree): PASS 192/192
  RealWorldQA + 256/256 VisDrone Det -- every `corrected_groups` count and every pushed-chunk hash
  identical. Snapshot medians, k=0.5: RWQA 155 -> 116 ms (k=1.0: 84), VisDrone 86 -> 79 (k=1.0: 59).
- keep-0.5 re-profile (fixed profiler; `cmdbuf_full` no longer booked as a kernel):

  | tokens | one-shot | k0.5 torch(L4a) | recv-attn torch(L4a) | k0.5 triton | recv-attn triton |
  |---:|---:|---:|---:|---:|---:|
  | 2109 | 59.5 | 309.4 | 199.6 | 230.8 | 121.5 |
  | 4816 | 225.6 | 893.3 | 491.6 | 969.0 | 567.8 |

  The Triton pair is `_rowstats_kernel` 42 ms + `_colsum_kernel` 78 ms over 27 layers at 2109 --
  ~4.5 ms per layer. The earlier "10x vs the unit test" reading was a size mistake, not a data
  effect: the profile's token counts are MERGED tokens, the kernel sees PATCH tokens (x4: 2109 ->
  8436, 4816 -> 19264). `qwen_recv_attn_real_test.py` (real q/k captured from the tower vs random
  of the same shape, one process): 8436 tokens torch 7.4 / triton 4.5 ms per layer, 19264 tokens
  torch 18.2 / triton 21.1 ms -- real == random to the 0.1 ms. So the kernel is simply slow at
  large T (the tile config 64x64/4 warps loses to torch's cuBLAS+softmax past ~10k tokens); the
  lever is tiling (`APPCORR_RECV_ATTN_CFG`, sweep in `qwen_recv_attn_triton_test.py --sweep`),
  and the elementwise exp/div work bounds the gain at roughly 2x. Any tile change alters the fp32
  summation order -> selection-identity re-gate before it becomes the default. L4b stays opt-in.
- Contiguous-band slice path (keep=1.0 only): `span=(g0*unit, g1*unit)` from the axis makes the
  row/cos/sin gathers and the 27 x 2 K/V scatters plain slices. CPU gate PASS; GPU gate PASS
  192/192 (bitwise on pushed bytes vs `snap_122b_rwqa192_torch.json`).
  Bounded by the gather bucket (5.5% / 2.9% of the k=1.0 pass), so a launch-count win, not a
  time win.

## Throughput A/B (VisDrone Det 448, streaming keep 1.0, --workers 6 --concurrency 4, one live
122B server, 21:35-21:40)

| tree | samples/s | median t_vision_ms | t_prep | server total_ms | client done_ms |
|---|---:|---:|---:|---:|---:|
| old (live campaign code) | 2.69 | 281 | 36 | 358 | 1463 |
| new (L1-L5 + span) | **4.46** | **99** | 51 | 529 | 901 |
| ceiling arm (campaign rows, for scale) | 7.8 | 74 | 17 | 326 | 664 |
| new + bridge reader thread (ab_rx, 2026-09-09) | **5.00** | 102 | 51 | 540 | 788 |

Bridge change: `result()` used to drain EVERY outstanding chunk ack -- including the newest
sample's five, each released only between engine steps -- before asking for the oldest sample's
result (median 101 ms of loop wait at c4). A reader thread now matches acks as they arrive and
the result request goes over a second socket; loop wait 101 -> 47 ms (details and the c1/c4/c8
rows in vllm_stream_design.md).

The vision pass is now 1.3x the one-shot tower (99 vs 74 ms) and the per-sample time is
engine-side: server total 529 ms (5 chunk prefills + 21 decode tokens) of the 901 ms client time.
Pred agreement with the campaign rows: old rerun 301/448, new 161/448. Root-caused (ceiling c1
old/new/old2 448/448 identical; streaming c1 old==old2 only 316/448, old==new 166, and a 40 ms
sleep after each push moves old==new_delay to 287): chunk-ARRIVAL timing decides which chunks
the scheduler prefills together, hence the FP8/MoE GEMM shapes, hence 1-px box shifts. Streaming
preds are timing-dependent at any concurrency; accuracy stays within +-1 pp. The driver is gated
on pushed bytes (bitwise), not on streaming preds.

## Caveats

- Profiled with the campaign server resident but idle on the same GPU; absolute numbers are
  for that state. The relative old/new comparison is same-run, same-image.
- "Command Buffer Full" is a profiler pseudo-event (host blocked on a full GPU queue); the
  first profile run booked it as a kernel and printed kernel time > wall on the keep<1 pass.
  The script now reports it separately.
