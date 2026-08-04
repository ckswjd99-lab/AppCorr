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

## What to do

1. **Switch ImageNet to a layermean pscore.** `cls_attn_prob_layermean` is already in
   `_VALID_SERVER_PSCORES` and `_LAYERMEAN_SERVER_PSCORES`, so it is a drop-in for
   `imnet_interleaved_g4.json`. ⚠️ It changes *which* tokens get selected (mean-over-layers vs
   per-layer score), so re-measure top-1 before adopting — this is a selection change, not just a
   caching change.
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
