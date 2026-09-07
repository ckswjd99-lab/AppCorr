# DINOv3 tail-full and L2-L1-L0 calibration

## Implemented semantics

- `final_full_layers=N` partitions only layers `[0, 40-N)` across progressive correction groups. The final group first corrects that prefix, then runs `[40-N, 40)` once through the stock block forward without creating correction caches.
- `L2L1L0ProgressiveLaplacian` emits L2 base, one complete L1 residual group, then four L0 residual groups. A coarse L1 patch maps to its 2×2 fine ViT token cells. Selection uses L1/L0 residual energy multiplied by layer-mean CLS attention.
- CLS and four register tokens remain mandatory correction queries. Existing `ProgressiveLaplacian` and N=0 scheduling remain available unchanged.

## Evaluation protocol

DINOv3 ViT-7B/16 (BF16, 40 layers, hidden size 4096) was run on one or two NVIDIA B200 GPUs. Calibration uses the same deterministic 1,024 ImageNet validation samples for every configuration, batch size 32, and 10,000 paired bootstrap resamples. Dominant FLOPs include ViT projection, attention, and SwiGLU matmuls; they exclude codec and host work.

## Main findings

- Full resolution reaches 87.30% top-1; L2-only reaches 83.50% (-3.81 points).
- On the existing L2-L0 path, N=3/K=25% reaches 86.52% (-0.78 points) at 1.189× dominant FLOPs. Compared with N=0/K=25%, tail deferral reduces correction overhead by 6.0% while preserving the measured top-1.
- The matching L2-L1-L0 N=3/K=25% point reaches 86.33% at 1.266× FLOPs and 298.6 KiB/image. It is dominated by L2-L0 (259.1 KiB/image).
- Every progressive point exceeds 1.0× dominant backbone FLOPs because all layers still run one approximate/full pass and correction is additional work. Any end-to-end gain must therefore come from pipeline overlap or codec behavior, not compute reduction.

## Full calibration table

All rows use identical deterministic ImageNet sample indices. Accuracy deltas and confidence intervals are paired against the `full_sequential` predictions. FLOPs count dominant ViT block matmuls; measured timings include selector and correction runtime.

| setting | top-1 | Δ full (95% CI) | FLOPs/full | backbone p50 | correction p50 | wire/image |
|---|---:|---:|---:|---:|---:|---:|
| full_sequential | 87.30% | +0.00 [+0.00, +0.00] | 1.000× | 192.1 ms | 0.0 ms | 170.6 KiB |
| l2l0_n0_k1 | 87.30% | +0.00 [-0.88, +0.88] | 1.661× | 320.2 ms | 116.1 ms | 259.1 KiB |
| l2l0_n2_k1 | 87.21% | -0.10 [-0.98, +0.78] | 1.641× | 339.3 ms | 120.4 ms | 259.1 KiB |
| l2l1l0_n2_k1 | 87.21% | -0.10 [-0.78, +0.59] | 1.920× | 527.6 ms | 275.5 ms | 298.6 KiB |
| l2l1l0_n0_k1 | 87.21% | -0.10 [-0.78, +0.59] | 1.940× | 444.9 ms | 232.2 ms | 298.6 KiB |
| l2l0_n3_k1 | 87.11% | -0.20 [-1.07, +0.68] | 1.621× | 337.1 ms | 121.0 ms | 259.1 KiB |
| l2l1l0_n3_k1 | 87.11% | -0.20 [-0.88, +0.39] | 1.901× | 444.6 ms | 231.0 ms | 298.6 KiB |
| l2l1l0_n2_k0.6 | 86.82% | -0.49 [-1.37, +0.39] | 1.570× | 404.1 ms | 189.9 ms | 298.6 KiB |
| l2l0_n3_k0.6 | 86.72% | -0.59 [-1.66, +0.39] | 1.387× | 313.4 ms | 80.5 ms | 259.1 KiB |
| l2l1l0_n2_k0.4 | 86.72% | -0.59 [-1.56, +0.39] | 1.395× | 420.9 ms | 187.0 ms | 298.6 KiB |
| l2l1l0_n0_k0.6 | 86.72% | -0.59 [-1.56, +0.29] | 1.582× | 438.6 ms | 200.4 ms | 298.6 KiB |
| l2l0_n2_k0.4 | 86.62% | -0.68 [-1.76, +0.29] | 1.279× | 330.5 ms | 131.5 ms | 259.1 KiB |
| l2l1l0_n3_k0.4 | 86.62% | -0.68 [-1.66, +0.29] | 1.387× | 414.2 ms | 180.6 ms | 298.6 KiB |
| l2l1l0_n0_k0.4 | 86.62% | -0.68 [-1.66, +0.29] | 1.404× | 432.3 ms | 182.6 ms | 298.6 KiB |
| l2l1l0_n3_k0.6 | 86.62% | -0.68 [-1.56, +0.20] | 1.558× | 440.2 ms | 201.5 ms | 298.6 KiB |
| l2l0_n3_k0.25 | 86.52% | -0.78 [-1.86, +0.29] | 1.189× | 307.9 ms | 93.8 ms | 259.1 KiB |
| l2l0_n0_k0.25 | 86.52% | -0.78 [-1.86, +0.29] | 1.201× | 280.8 ms | 58.7 ms | 259.1 KiB |
| l2l0_n2_k0.6 | 86.52% | -0.78 [-1.86, +0.20] | 1.400× | 353.2 ms | 145.3 ms | 259.1 KiB |
| l2l1l0_n0_k0.25 | 86.43% | -0.88 [-1.95, +0.10] | 1.278× | 412.8 ms | 161.6 ms | 298.6 KiB |
| l2l0_n0_k0.4 | 86.43% | -0.88 [-1.95, +0.10] | 1.287× | 327.5 ms | 82.5 ms | 259.1 KiB |
| l2l0_n0_k0.6 | 86.43% | -0.88 [-1.95, +0.10] | 1.412× | 322.0 ms | 96.1 ms | 259.1 KiB |
| l2l1l0_n3_k0.25 | 86.33% | -0.98 [-2.05, +0.00] | 1.266× | 432.0 ms | 162.8 ms | 298.6 KiB |
| l2l0_n2_k0.25 | 86.23% | -1.07 [-2.25, +0.00] | 1.195× | 289.9 ms | 66.7 ms | 259.1 KiB |
| l2l0_n3_k0.4 | 86.23% | -1.07 [-2.15, -0.10] | 1.270× | 325.6 ms | 87.1 ms | 259.1 KiB |
| l2l1l0_n2_k0.25 | 86.23% | -1.07 [-2.15, -0.10] | 1.272× | 413.6 ms | 171.4 ms | 298.6 KiB |
| l2_approx_only | 83.50% | -3.81 [-5.47, -2.25] | 1.000× | 187.8 ms | 0.0 ms | 14.0 KiB |

The calibration set ranks configurations; it is not the final ImageNet validation claim. Promote Pareto candidates to the full 50,000-image validation set before reporting final accuracy.

## Isolated B200 latency

These runs use one B200 with no simultaneous model loading, three warm-up batches, and 32 measured batch-32 requests. Component times overlap and therefore must not be summed.

| setting | top-1 | FLOPs/full | request p50 / p95 | speedup vs full (95% CI) | encode | decode | backbone |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_sequential_isolated | 87.30% | 1.000× | 761.1 / 853.0 ms | 1.000× [1.000, 1.000] | 394.3 ms | 45.9 ms | 191.9 ms |
| l2l0_n3_k0.25_isolated | 86.52% | 1.189× | 535.0 / 670.3 ms | 1.423× [1.296, 1.467] | 318.3 ms | 233.0 ms | 326.5 ms |
| l2l1l0_n3_k0.25_isolated | 86.33% | 1.266× | 668.9 / 835.4 ms | 1.138× [1.074, 1.178] | 348.9 ms | 327.6 ms | 436.4 ms |

The progressive request speedup is a pipeline/codec effect, not compute reduction: its backbone time and dominant FLOPs are both higher than full inference.

## Limitations

- The 1,024-image calibration confidence intervals are wide; these are ranking results, not final ImageNet claims.
- Calibration p95 latency is affected by simultaneous checkpoint mmap on the other GPU. Only the isolated table is used for latency conclusions.
- The isolated run has no imposed uplink delay. Finite-bandwidth crossover must be measured separately because progressive streams transmit more bytes.
- Progressive decode remains CPU-heavy. The L1 stage adds both decode work and wire bytes, explaining much of its end-to-end regression.
