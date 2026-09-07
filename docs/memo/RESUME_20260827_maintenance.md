# Resume sheet — server maintenance 2026-08-27 24:00 (processes killed, /NHNHOME survives)

State at write time (~12:00). Everything below is committed on `develop/qwen35-appcorr`
unless noted; results land under `analysis/results/` progressively, and the accuracy
driver RESUMES from its own jsonl (finished indices are skipped), so a mid-run kill
loses at most the in-flight sample.

## Running / queued on GPU0 (in order)
1. Qwen3.5-35B ChartQA 3-arm (`analysis/results/qwen35_chartqa_accuracy.log`) --
   ceiling done (88.56), floor in flight. Resume: rerun the same command; jsonl resume.
     python analysis/experiments/qwen35_accuracy.py --dataset chartqa --arms ceiling floor streaming --groups 4
2. VSR chain (`scripts/vsr_chain.sh`, waits on QWEN35_ACCURACY_COMPLETE in the chartqa
   log): qwen35 3-arm -> gemma3 4-arm -> ov2 4-arm. gemma3/ov2 arms skip on existing
   json, so rerunning the script resumes. VSR images: `analysis/qwen_vl_prefill/_vsr_images/`
   -- 687 SYMLINKS into /NHNHOME/share/cjpark/data/coco_train2017/train2017 (survive) +
   28 downloaded files (survive).

## GH200 (separate box; maintenance scope unknown -- warned)
Text-per-round fix for the Qwen2.5 executor + g=1 gate + 2000-subset A/B, expected to
land on `develop/critical-flops-accounting`. If their numbers arrive, totals ~136/157%
replace 177/199% in inprocess_flops.json (keep the old as *_textallrounds if re-measured).

## Open items (priority order)
- Gemma3/OV2 ddagger: progressive-arm accuracy re-evals (the only remaining ddagger).
- VGGT ours-below-floor anomaly (padding-independent, confirmed).
- 122B: FP8 kernels broken on B200 (sm_100 vs deep-gemm sm_90). Options recorded in
  memory `project_qwen35_fp8_chunking_lossy.md`: 2-GPU bf16 device_map, or an FP8Linear
  dequant-to-bf16 shim (recommended; awaiting user go).
- OpenCLIP one-shot row decision; avg-attn ImageNet 2 arms (deprioritized).

## Table
`analysis/experiments/make_eval_table.py --format latex --keeps 0.25 0.50`
68/68 accuracy cells + Qwen2.5 complete; Qwen3.5 RealWorldQA row live; ChartQA/VSR rows
land as the runs above finish.
