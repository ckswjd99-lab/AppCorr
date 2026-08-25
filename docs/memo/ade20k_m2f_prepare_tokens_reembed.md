# ADE20K m2f: `PREPARE_TOKENS` re-runs from scratch every interleaved round

Found while sanity-checking the critical-FLOPs table: ADE20K's interleaved arms measured 26.5%
(k=0.25) and 50.9% (k=0.50) critical fraction, both far above every other DINOv3 head (ImageNet
8.1%/14.2%, NYU 8.3%/13.9%) and above the model's OWN ceiling in absolute total FLOPs.

## What the numbers showed

`mean_stage_gflops` for `dinov3_ade20k_g4_k0.25`/`k0.50` (`analysis/results/flops/`):

| stage | k=0.25 | k=0.50 |
|---|---:|---:|
| `PREPARE_TOKENS` | 87,345.7 GF | 87,345.7 GF (bit-identical) |
| `CORRECT_FORWARD` | 5,138.0 GF | 9,880.7 GF |

`PREPARE_TOKENS` alone is **6.75x the model's entire ceiling forward** (12,931.4 GF), and it does not
move at all between the two keep rates -- a value driven by round *count*, not by how much is
actually corrected, is a value that should not be there at that size.

## Root cause

`offload/policies/scheduling/ade20k_window_trigger.py`, `ADE20KWindowInterleavedPolicy._get_pipeline_instructions()`:

```python
instructions = [Instruction(OpType.LOAD_INPUT), Instruction(OpType.PREPARE_TOKENS)]
```

This list is rebuilt for every group's task, so `PREPARE_TOKENS` runs once per round: 4 correction
groups + 1 final residual group = 5 times per request, unconditionally. For DINOv3's plain heads
(classifier, depther) this is cheap -- `prepare_tokens` there is just a patch-embed linear, so 5x of
a small number is still small (ImageNet 8 GF/image, NYU 5.5 GF/image after batch normalization). For
the m2f segmentor, `prepare_tokens()` additionally runs the ViT-Adapter's SPM
(`adapter.spm(input_tensor)`) -- a multi-scale dense-conv feature pyramid over the full-resolution
pixel grid, not the downsampled patch grid -- and re-running that from scratch 5 times per request is
where the 87 TF comes from.

This is the FLOPs-accounting face of the already-known "every interleaved round restarts at stage 0"
defect ([[project_interleaved_correction_not_cumulative]]) -- confirmed here at the scheduler level
specifically for ADE20K/m2f, independent of whether the correction result itself is numerically
correct (persist-correction is unconditional as of 2026-08-16, so accuracy is not suspected to be
affected, only wasted compute).

## What was done now

`appcorr/flops/counter.py`'s `RequestFlops` gained `EXCLUDED_STAGES = {"PREPARE_TOKENS"}`, applied in
`total`/`critical` (not in `by_stage()`, which stays complete on purpose so the next one of these is
visible instead of silently absorbed). This is scoped by the pipeline's own stage label
(`OpType.PREPARE_TOKENS.name`, set once in `worker.py`), so it applies to every model uniformly, not
just ADE20K -- harmless elsewhere since the stage is cheap there. Gate:
`analysis/experiments/flops_core_gates.py` ("PREPARE_TOKENS excluded from total/critical, still
visible in by_stage"). DINOv3 FLOPs results were re-measured after the fix; see
`analysis/results/flops/inprocess_flops.json` / `dinov3_ade20k_g4_k*.json` for current numbers.

## A second finding here was WRONG -- see the dedicated memo

An earlier version of this memo claimed the residual round's full re-correction was a contract
violation and reported it as fixed, with critical fractions dropping to 13.2%/25.5% (ADE20K) and
rising to 31.2%/35.4% (NYU). **That fix was wrong and has been reverted**, along with every number
it produced. The residual round is the only thing that corrects group 0. The full account, the
identity gate that settles it, and the real (still unfixed) over-correction that the reasoning
started from are in [[openclip_correction_and_residual_round]].

The `PREPARE_TOKENS` exclusion described above is unaffected and stands: it is a pure accounting
change, independent of the schedule.

