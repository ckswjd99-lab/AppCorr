# What the non-GEMM ~70% of CORRECT_FORWARD actually is

**Status date:** 2026-08-04
**Branch:** `develop/dinov3-approx-fp4`
**Script:** `analysis/experiments/dinov3_correct_profile.py`

## Why

[dinov3_nvfp4_speedup_gate.md](dinov3_nvfp4_speedup_gate.md) showed the five weight-GEMMs are only
~28% of the correction stage — NVFP4 buys 1.13× at ImageNet bs=128 and nothing below it — but nobody
had measured what the other ~70% *is*. Profiled one full 40-block `correct_partial_token` pass at
ImageNet shapes (B=128, N=261, M=8832, 100% keep).

## GPU: where the kernels go

Raw CUDA kernels only (`aten::*` wrappers excluded — they double-count their own children), against
a 183.8 ms measured wall time for the pass:

| bucket | kernels | ms | % of wall |
|---|---|---:|---:|
| **GEMM** | `nvjet_sm100_*` ×3 | **73.3** | **40%** |
| **index / gather / scatter** | `index_elementwise_kernel` ×2 | **43.2** | **24%** |
| elementwise | `elementwise_kernel`, `vectorized_elementwise_kernel` | 13.4 | 7% |
| attention | `cudnn_generated_fort_native_sdpa_*` | 12.9 | 7% |
| RoPE | `_rope_active_inplace_kernel` | 6.3 | 3% |
| token update | `_active_token_update_kernel` | 5.9 | 3% |
| LayerNorm | `vectorized_layer_norm` | 4.9 | 3% |
| unaccounted | (tail below the print cutoff) | ~24 | 13% |

**The single largest non-GEMM cost is index/gather/scatter at 24%** — i.e. `aten::index`
(the `x[active_batch_idx, active_token_idx]` gather that packs selected tokens, 17.4 ms) and
`aten::_index_put_impl_` (the K/V scatter-back into `{tag}_kv` plus the `q_padded` scatter, 25.8 ms).
That is pure data movement in service of the packed-sparse representation, and it costs more than
attention and LayerNorm combined.

*(Caveat: the script's own "coarse attribution" print is unreliable — its bucket patterns match
`aten::addmm` into the elementwise bucket. Use the table above, which is derived from raw kernel
names.)*

## CPU: the correction stage is launch-bound, not compute-bound

| | `aten::item` | `cudaStreamSynchronize` | `aten::nonzero` |
|---|---:|---:|---:|
| calls / CPU ms | 200 / **144.74 ms** | 320 / **144.48 ms** | 80 / 4.36 ms |

Against a 183.8 ms GPU wall, the host is stalled ~145 ms. The CPU cannot run ahead to queue work, so
the GPU bubbles. The syncs come from `_build_packed_query_state` (`block.py:382-442`), which calls
`.item()` to materialise `max_keep` and `nonzero()` to build the active-token index.

## The plan cache is silently disabled on ImageNet

`_shared_partial_token_plan_key` (`block.py:530-563`) returns `None` — disabling the query-plan cache
entirely — unless `server_pscore_weight == 0` or the pscore is one of `_LAYERMEAN_SERVER_PSCORES`.

- `ade20k_m2f_interleaved_static.json` uses `patch_attn_prob_layermean` → **cache ON**, plan built
  once and reused across all 40 layers.
- `imnet_interleaved_g4.json` resolves to `cls_attn_prob` → **cache OFF**, so the plan (and its
  `.item()` sync) is rebuilt **once per block per correction round**.

A/B on the same shapes, changing only the pscore:

| | plan cache OFF (`cls_attn_prob`) | plan cache ON (`*_layermean`) |
|---|---:|---:|
| GPU wall, 40-block pass | 183.8 ms | **166.4 ms (1.10×)** |
| `aten::item` | 200 calls / 144.74 ms | 5 calls / 0.10 ms |
| `cudaStreamSynchronize` | 320 calls / 144.48 ms | — |
| `aten::nonzero` | 80 calls / 4.36 ms | 2 calls / 0.13 ms |

**Enabling the plan cache removes 195 of 200 host syncs and is worth ~10% of the correction stage**
— comparable to the entire 1.13× NVFP4 win, for a one-line config change.

## Fix shipped: sync-free query plan when every candidate is kept — **1.56×**

The `.item()` exists because `max_keep` **sizes a tensor** (`update_indice = zeros(B, max_active)`),
and shapes must be host ints; the two `nonzero()` calls stall for the same structural reason (their
output length is data-dependent). None of that is avoidable in general.

But the builder is data-dependent *only* through `keep_patch_mask`. When that mask is all-True —
exactly `token_keep_thres is None and token_keep_ratio >= 1.0` — everything becomes static:
`max_keep` is just `dindice_patches.shape[1]`, the packed layout is `[dindice_pre | dindice_patches]`
in order, and `nonzero()` on an all-True mask is plain row-major order reproducible with
`arange`/`repeat_interleave`. `_build_packed_query_state_all_keep` builds the whole plan that way,
with no host round-trip. Verified to produce **bit-identical** `PackedQueryState` output to the
general builder across B∈{1,3,4,128}.

Measured on `imnet_interleaved_g4.json`, bs=64, 10 warmup / 30 measured:

| | baseline | sync-free | |
|---|---:|---:|---|
| `CORRECT_FORWARD` avg | 389.04 ms | **248.78 ms** | **1.56×** |
| min | 277.69 | 234.14 | |
| max | 507.39 | 332.37 | |
| top-1 / top-5 | 91.25 / 99.375 | 91.25 / 99.375 | **identical** |

The isolated profiler showed only 1.06× (183.8 → 173.0 ms) because there the GPU had enough queued
work to partly hide the stalls; in the real pipeline the host block genuinely prevents the GPU worker
from running ahead, so the win is far larger. All `aten::item` / `nonzero` / `cudaStreamSynchronize`
calls disappear from the profile.

**This is worth more than the entire NVFP4 effort** (1.56× vs 1.13×), costs no accuracy at all
(identical, not "within noise"), and is ~50 lines. It confirms the profiling-first ordering: the
correction stage was launch-bound, not compute-bound.

**Coverage:** the fast path triggers on ImageNet-style configs (`token_keep_ratio: 1.0`). ADE20K
uses threshold selection so it takes the general builder — but it already has the layermean plan
cache, which reduces the same syncs to once per round. The two workloads are covered by different
mechanisms, and notably ImageNet no longer needs to be switched to a layermean pscore (which would
have changed token selection and required accuracy re-validation).

## What to do

1. ~~Switch ImageNet to a layermean pscore.~~ **Superseded** by the sync-free builder above, which
   gets a larger win (1.56× vs the ~1.10× the plan cache was worth here) without changing token
   selection at all.
2. **Attack the 24% index/gather/scatter.** The packed-sparse layout pays a gather to build
   `x_active` and two scatters to write results back. Options worth measuring: fusing the gather into
   the first LayerNorm, keeping K/V updates in the packed layout until the end of the round instead
   of scattering per block, or a fused gather+norm Triton kernel alongside the existing
   `active_token_update` / `rope_active_inplace` ones.
3. **Remove the remaining `.item()` even when the plan cache is on.** With the cache the sync is paid
   once per round rather than per block, but a fully device-side plan (bounded `max_keep`, no host
   round-trip) would remove it entirely — relevant for the ADE20K path where rounds are frequent.
4. Only then revisit quantization: at 40% of GPU time the GEMMs are worth roughly what the gate
   measured, and the items above are cheaper wins.

## Reproduce

```bash
PYTHONPATH="$PWD/appcorr/models:$PWD" python analysis/experiments/dinov3_correct_profile.py \
    --batch-size 128 --num-groups 4                                  # plan cache OFF (imnet-like)
PYTHONPATH="$PWD/appcorr/models:$PWD" python analysis/experiments/dinov3_correct_profile.py \
    --batch-size 128 --num-groups 4 --server-pscore patch_attn_prob_layermean   # cache ON
```

---

## 2026-08-04: aten gather/scatter replaced with row kernels

Re-profiled at ADE20K-like shapes (B=4, N=1029, M=2068, `patch_attn_prob_layermean`, so the plan
cache is on) before and after `scatter_rows_triton` / `gather_rows_triton`:

| | before | after |
|---|---:|---:|
| **one 40-block correction pass (wall)** | **38.9 ms** | **35.8 ms** (−8.0%) |
| `aten::_index_put_impl_` | 5.70 ms / 80 calls | 2.25 ms / 40 calls |
| `aten::index` | 3.87 ms / 80 calls | 2.12 ms / 40 calls |

**The gain is smaller than the microbenchmark implies (−3.1 ms, not the ~7.6 ms projected).** Two
reasons, both visible in the call counts: only *half* of each aten op's calls were the sites that
were converted (40 of 80 each), and the replacement kernels are not free -- ~12-17 us x 80 calls
adds back ~1.3 ms. Net ≈ 3.9 ms of aten time removed, 3.1 ms of it showing up in wall.

**Remaining, and worth the next look:** 40 calls of `_index_put_impl_` (2.25 ms) and 40 of
`aten::index` (2.12 ms) are still on aten at other sites -- the `q_padded` scatter and the `dv_cache`
write in `attention.py`, plus the group-slice gathers in `block.py`. Same fix should apply, for
roughly another 3 ms.

Also note the coarse-attribution block in the script remains unreliable (it buckets `aten::addmm`
under elementwise); read the raw kernel rows.

---

## 2026-08-05: the threshold path's syncs cost 23% of the pass in GPU idle

The profiler now prints wall against the sum of kernel times for the same pass (`launch gap`).
At the real correction shape (B=2, N=3141), FP4:

| selection | wall | kernels | idle |
|---|---:|---:|---:|
| `token_keep_thres=4e-5` (what ADE20K runs) | 77.5 ms | 59.62 ms | **17.84 ms (23.0%)** |
| `token_keep_ratio=0.408` (top-k) | 28.1 ms | 28.31 ms | **-0.22 ms (0%)** |
| BF16, threshold | 94.8 ms | 90.97 ms | 3.86 ms (4.1%) |

**Same FP4 code, same kernel count -- only the selection path differs, and the idle goes from 23% to
zero.** The cause is `_build_packed_query_state`: threshold selection needs `.item()` for `max_keep`
and `nonzero()` for the active index, and those host round-trips stall the launch pipeline. Top-k
takes `_build_packed_query_state_fixed_k`, which is sync-free, and the GPU stays saturated.

This corrects two earlier conclusions on this branch:

* **"FP4's extra kernels cause the idle."** They do not. FP4 issues ~280 more kernels than BF16, and
  batch-scaling put its batch-independent cost at 11.78 ms against BF16's 0.90 ms, which looked like
  per-kernel overhead. But holding the kernel count fixed and changing only the selection path
  removes the idle entirely.
* **"Top-k is not worth it -- the syncs only cost 0.2 ms."** That 0.2 ms was *CPU* time in the
  profiler's sync counters. It never measured the GPU bubble those syncs open, which is ~90x larger.

Caveat: the two configurations have different M (top-k keeps 40.8% of candidates; the threshold
passes far more in this synthetic setup, 28 vs 77.5 ms eager). So the attribution is strongly
suggested, not proven. A controlled comparison at matched M is the first thing to run.

### CUDA graphs are not the answer

Capture of the 40-block loop **fails** on the threshold path -- `cudaErrorStreamCaptureInvalidated`,
because capture forbids any sync. It **succeeds** on top-k, and replays at 1.058x -- which is
nothing, because that configuration already has zero idle to recover. Graphs were being considered
as a way to reclaim the 17.84 ms; the sync removal reclaims it directly and graphs then have no
work left to do.

### What to do

Make the threshold path sync-free rather than switching to top-k. Top-k gets there but trades away
the threshold's content adaptivity. The syncs are avoidable: `max_keep` only sizes a tensor, so it
can be replaced by a fixed upper bound (the candidate count, or a bucketed one) with the surplus
rows masked out -- the same shape as `_build_packed_query_state_all_keep`, which already builds its
plan from static shapes. `nonzero()` likewise has a masked equivalent.
