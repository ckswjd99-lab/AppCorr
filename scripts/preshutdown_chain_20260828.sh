#!/usr/bin/env bash
# Pre-maintenance GPU0 chain (server dies 18:00 KST 2026-08-28).
#
# 1. Qwen3.5-35B keep-arm FLOPs (short; fills the k0.25/k0.50 Comp cells).
# 2. Gemma3/OV2 ddagger re-evals under the canonical progressive arm, wide-gap
#    datasets first. Each run gates on the wall clock: nothing starts after
#    17:15, so the last run either finishes or was never begun (a run cut at
#    18:00 writes no json and would waste its whole slot).
#
# Output tags are progressive_g4_k{keep}.json -- NEW paths (the launch checklist
# forbids reusing interleaved_g4_* paths), and exactly what make_eval_table.py
# now prefers over the upfront arm's files.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
unset HF_TOKEN
DEADLINE=$(date -d "17:15" +%s)

python analysis/experiments/flops_report_qwen35.py \
    --datasets chartqa realworldqa --keeps 0.25 0.50 1.00 --samples 12 \
    --out-json analysis/results/flops/qwen35_flops_keeps.json \
    > analysis/results/qwen35_flops_keeps.log 2>&1
echo "PS: qwen35 keep-flops rc=$? $(date)"

run () {  # model dataset keep est_label
  local M=$1 DS=$2 K=$3
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "PS: DEADLINE reached, skipping $M/$DS k$K"; return; }
  local out=analysis/results/${M}_${DS}/progressive_g4_k${K}.json
  [ -s "$out" ] && { echo "PS: skip $M/$DS k$K (exists)"; return; }
  python analysis/experiments/${M}_oracle.py --dataset "$DS" --full \
      --arm progressive --keep "$K" --groups 4 --level 2 \
      --out-json "$out" > "${out%.json}.log" 2>&1
  echo "PS: $M/$DS k$K rc=$? $(grep -aoE '"accuracy": [0-9.]+' ${out%.json}.log | head -1) $(date)"
}

# Wide-gap first; longest blocks earliest so the deadline gate cuts cleanly.
run gemma3 docvqa  0.25; run gemma3 docvqa  0.50
run ov2    docvqa  0.25; run ov2    docvqa  0.50
run gemma3 chartqa 0.25; run gemma3 chartqa 0.50
run ov2    chartqa 0.25; run ov2    chartqa 0.50
run gemma3 infovqa 0.25; run gemma3 infovqa 0.50
run ov2    infovqa 0.25; run ov2    infovqa 0.50
run gemma3 textvqa 0.25; run gemma3 textvqa 0.50
run ov2    textvqa 0.25; run ov2    textvqa 0.50
echo "PRESHUTDOWN_CHAIN_COMPLETE $(date)"
