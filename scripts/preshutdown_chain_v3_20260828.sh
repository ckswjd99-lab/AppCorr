#!/usr/bin/env bash
# v3 (user directive ~04:00): the new real-photo benchmarks (cvbench, mmvp) get
# bounds + progressive arms AHEAD of pope/refcoco/gqa. Waits for whatever oracle
# run v2 left in flight, then takes over. Same 17:15 start gate, same
# skip-if-exists, same log (the armed monitor keeps streaming).
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
unset HF_TOKEN
DEADLINE=$(date -d "17:15" +%s)

while pgrep -f "_oracle.py --dataset" > /dev/null; do sleep 30; done

run () {  # model dataset arm keep(tag "" for bounds)
  local M=$1 DS=$2 ARM=$3 K=${4:-}
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "PS: DEADLINE, skip $M/$DS $ARM$K"; return; }
  local out
  if [ -n "$K" ]; then out=analysis/results/${M}_${DS}/progressive_g4_k${K}.json
  else out=analysis/results/${M}_${DS}/${ARM}.json; fi
  [ -s "$out" ] && { echo "PS: skip $M/$DS $ARM$K (exists)"; return; }
  mkdir -p "$(dirname "$out")"
  local args=(--dataset "$DS" --full --arm "$ARM" --level 2 --out-json "$out")
  [ -n "$K" ] && args+=(--keep "$K" --groups 4)
  python analysis/experiments/${M}_oracle.py "${args[@]}" > "${out%.json}.log" 2>&1
  echo "PS: $M/$DS $ARM$K rc=$? $(grep -aoE '"accuracy": [0-9.]+' ${out%.json}.log | head -1) $(date)"
}

# -- finish the short real-photo block v2 started ------------------------------
run gemma3 vsr progressive 0.25; run gemma3 vsr progressive 0.50
run ov2    vsr progressive 0.25; run ov2    vsr progressive 0.50
# -- NEW real-photo primaries: bounds + both keeps -----------------------------
for M in gemma3 ov2; do
  run $M cvbench ceiling; run $M cvbench floor
  run $M cvbench progressive 0.25; run $M cvbench progressive 0.50
  run $M mmvp ceiling; run $M mmvp floor
  run $M mmvp progressive 0.25; run $M mmvp progressive 0.50
done
# -- rest of the real-photo re-evals ------------------------------------------
run gemma3 pope progressive 0.25; run gemma3 pope progressive 0.50
run ov2    pope progressive 0.25; run ov2    pope progressive 0.50
run ov2    refcoco progressive 0.25; run ov2 refcoco progressive 0.50
run gemma3 gqa progressive 0.25; run gemma3 gqa progressive 0.50
run ov2    gqa progressive 0.25; run ov2    gqa progressive 0.50
# -- documents: very last (user 2026-08-28) ------------------------------------
run gemma3 chartqa progressive 0.25; run gemma3 chartqa progressive 0.50
run ov2    chartqa progressive 0.25; run ov2 chartqa progressive 0.50
run gemma3 docvqa progressive 0.25; run gemma3 docvqa progressive 0.50
run ov2    docvqa progressive 0.25; run ov2 docvqa progressive 0.50
echo "PRESHUTDOWN_CHAIN_COMPLETE $(date)"
