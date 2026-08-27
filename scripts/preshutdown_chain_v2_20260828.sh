#!/usr/bin/env bash
# v2 (user directive 02:20): real-photo datasets FIRST for the progressive
# re-evals; long non-real-world sets (docvqa/chartqa/infovqa/textvqa) go to the
# very back. Short runs lead so dataset coverage is maximal when 17:15 gates the
# rest. Appends to the v1 log so the armed monitor keeps streaming stages.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
unset HF_TOKEN
DEADLINE=$(date -d "17:15" +%s)

run () {
  local M=$1 DS=$2 K=$3
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "PS: DEADLINE reached, skipping $M/$DS k$K"; return; }
  local out=analysis/results/${M}_${DS}/progressive_g4_k${K}.json
  [ -s "$out" ] && { echo "PS: skip $M/$DS k$K (exists)"; return; }
  mkdir -p "$(dirname "$out")"
  python analysis/experiments/${M}_oracle.py --dataset "$DS" --full \
      --arm progressive --keep "$K" --groups 4 --level 2 \
      --out-json "$out" > "${out%.json}.log" 2>&1
  echo "PS: $M/$DS k$K rc=$? $(grep -aoE '"accuracy": [0-9.]+' ${out%.json}.log | head -1) $(date)"
}

# --- real-photo sets, shortest first ------------------------------------------
run gemma3 realworldqa 0.25; run gemma3 realworldqa 0.50
run ov2    realworldqa 0.25; run ov2    realworldqa 0.50
run gemma3 vsr         0.25; run gemma3 vsr         0.50
run ov2    vsr         0.25; run ov2    vsr         0.50
run gemma3 pope        0.25; run gemma3 pope        0.50
run ov2    pope        0.25; run ov2    pope        0.50
run ov2    refcoco     0.25; run ov2    refcoco     0.50
run gemma3 gqa         0.25; run gemma3 gqa         0.50
run ov2    gqa         0.25; run ov2    gqa         0.50
# --- documents/charts: very low priority (user 2026-08-28) --------------------
run gemma3 chartqa 0.25; run gemma3 chartqa 0.50
run ov2    chartqa 0.25; run ov2    chartqa 0.50
run gemma3 docvqa  0.25; run gemma3 docvqa  0.50
run ov2    docvqa  0.25; run ov2    docvqa  0.50
echo "PRESHUTDOWN_CHAIN_COMPLETE $(date)"
