#!/usr/bin/env bash
# Last runner of the day: re-does what earlier chain generations invalidated
# after their scripts were already buffered (MG mmvp bounds, post-channel-fix).
# Waits for the box-remeasure chain to finish or die, then runs under the gate.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
unset HF_TOKEN
DEADLINE=$(date -d "17:15" +%s)
TF515=/NHNHOME/share/cjpark/tf515
MG=meta-models/Muse-Glimmer-30B
MG_CAP=3211264

while pgrep -f "qwen35_box_remeasure" > /dev/null || pgrep -f "chain_v5" > /dev/null \
      || pgrep -f "_oracle.py --dataset" > /dev/null; do sleep 60; done

run_mg () {
  local DS=$1 ARM=$2 CAP=$3 MNT=$4
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "PS: DEADLINE, skip catchup MG/$DS $ARM"; return; }
  local out=analysis/results/museglimmer30b_${DS}/${ARM}.json
  [ -s "$out" ] && { echo "PS: skip catchup MG/$DS $ARM"; return; }
  local extra=()
  [ -n "$CAP" ] && extra=(--degrade-max-px "$CAP")
  python analysis/experiments/vlm_bounds_oracle.py --model "$MG" --dataset "$DS" \
      --arm "$ARM" --full --max-new-tokens "$MNT" --transformers-path "$TF515" \
      --out-json "$out" "${extra[@]}" > "${out%.json}.log" 2>&1
  echo "PS: catchup MG/$DS $ARM rc=$? $(grep -aoE '"accuracy": [0-9.null]+' ${out%.json}.log | head -1) $(date)"
}

run_mg mmvp ceiling "" 24
run_mg mmvp floor "$MG_CAP" 24
echo "FINAL_CATCHUP_COMPLETE $(date)"
