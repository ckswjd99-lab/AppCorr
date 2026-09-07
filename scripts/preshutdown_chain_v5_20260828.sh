#!/usr/bin/env bash
# v5 (03:40): v4 + two conformance fixes, after the degradation-convention audit
# (user question) and the Muse Glimmer channel-protocol incident:
#   1. Pyramid convention: BOX down + model-sampled-resolution cap
#      (AGENTS.md; docs/memo/pyramid_degradation_native_vs_canvas.md). All
#      degradation-based mmvp results from v4 were invalidated and re-run here;
#      ceilings were kept (no degradation touches them).
#   2. Muse Glimmer: ATEM channel protocol -- reasoning_strength=low, 64-token
#      budget, and the oracle now scores only the to=user channel. The 49%
#      chance-level scores were reasoning-echo artifacts, deleted.
# Caps: gemma4 2520*256 (in its l2_degrade default), mistral 1540^2=2371600,
# muse glimmer 4096 tokens * (2*14)^2 = 3211264.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
unset HF_TOKEN
DEADLINE=$(date -d "17:15" +%s)
TF515=/NHNHOME/share/cjpark/tf515
MG=meta-models/Muse-Glimmer-30B
MISTRAL=mistralai/Mistral-Small-3.1-24B-Instruct-2503
MG_CAP=3211264
MISTRAL_CAP=2371600

while pgrep -f "_oracle.py --dataset" > /dev/null; do sleep 30; done

gate () { [ "$(date +%s)" -lt "$DEADLINE" ]; }

g4 () {  # dataset arm keep [max_new]
  local DS=$1 ARM=$2 K=${3:-} MNT=${4:-24}
  gate || { echo "PS: DEADLINE, skip gemma4/$DS $ARM$K"; return; }
  local out
  if [ -n "$K" ]; then out=analysis/results/gemma4_${DS}/corrected_k${K}.json
  else out=analysis/results/gemma4_${DS}/${ARM}.json; fi
  [ -s "$out" ] && { echo "PS: skip gemma4/$DS $ARM$K"; return; }
  mkdir -p "$(dirname "$out")"
  local args=(--dataset "$DS" --full --arm corrected --keep "$K" --max-new-tokens "$MNT" --out-json "$out")
  [ -z "$K" ] && args=(--dataset "$DS" --full --arm "$ARM" --max-new-tokens "$MNT" --out-json "$out")
  python analysis/experiments/gemma4_oracle.py "${args[@]}" > "${out%.json}.log" 2>&1
  echo "PS: gemma4/$DS $ARM$K rc=$? $(grep -aoE '"accuracy": [0-9.null]+' ${out%.json}.log | head -1) $(date)"
}

q35 () {
  local DS=$1
  gate || { echo "PS: DEADLINE, skip qwen35/$DS"; return; }
  local out=analysis/results/qwen35_accuracy
  [ -s "$out/${DS}_ceiling.jsonl" ] && { echo "PS: skip qwen35/$DS"; return; }
  mkdir -p "$out"
  python analysis/experiments/qwen35_accuracy.py --dataset "$DS" \
      --arms floor streaming ceiling --out "$out" > "$out/run_${DS}.log" 2>&1
  echo "PS: qwen35/$DS rc=$? $(grep -aoE '"acc": [0-9.]+' $out/run_${DS}.log | tail -3 | tr '\n' ' ') $(date)"
}

bounds () {  # model_id tag dataset arm tfpath max_new cap_px reasoning
  local MID=$1 TAG=$2 DS=$3 ARM=$4 TFP=$5 MNT=$6 CAP=$7 RS=$8
  gate || { echo "PS: DEADLINE, skip $TAG/$DS $ARM"; return; }
  local out=analysis/results/${TAG}_${DS}/${ARM}.json
  [ -s "$out" ] && { echo "PS: skip $TAG/$DS $ARM"; return; }
  mkdir -p "$(dirname "$out")"
  local extra=()
  [ -n "$TFP" ] && extra+=(--transformers-path "$TFP")
  [ -n "$CAP" ] && extra+=(--degrade-max-px "$CAP")
  [ -n "$RS" ] && extra+=(--reasoning-strength "$RS")
  python analysis/experiments/vlm_bounds_oracle.py --model "$MID" --dataset "$DS" \
      --arm "$ARM" --full --max-new-tokens "$MNT" --out-json "$out" "${extra[@]}" \
      > "${out%.json}.log" 2>&1
  echo "PS: $TAG/$DS $ARM rc=$? $(grep -aoE '"accuracy": [0-9.null]+' ${out%.json}.log | head -1) $(date)"
}

# ---- Fixup block: convention-compliant re-runs of invalidated results --------
g4 mmvp floor; g4 mmvp corrected 0.25; g4 mmvp corrected 0.50
bounds "$MISTRAL" mistral24b mmvp floor "" 24 "$MISTRAL_CAP" ""
bounds "$MG" museglimmer30b mmvp ceiling "$TF515" 64 "" low
bounds "$MG" museglimmer30b mmvp floor   "$TF515" 64 "$MG_CAP" low

# ---- cvbench ----------------------------------------------------------------
g4 cvbench ceiling; g4 cvbench floor
q35 cvbench
bounds "$MISTRAL" mistral24b cvbench ceiling "" 24 "" ""
bounds "$MISTRAL" mistral24b cvbench floor   "" 24 "$MISTRAL_CAP" ""
bounds "$MG" museglimmer30b cvbench ceiling "$TF515" 64 "" low
bounds "$MG" museglimmer30b cvbench floor   "$TF515" 64 "$MG_CAP" low
g4 cvbench corrected 0.50

# ---- wildvision dumps (128 tokens; caps ACTIVE here -- 2295px images) --------
g4 wildvision ceiling "" 128; g4 wildvision floor "" 128; g4 wildvision corrected 0.50 128
bounds "$MISTRAL" mistral24b wildvision ceiling "" 128 "" ""
bounds "$MISTRAL" mistral24b wildvision floor   "" 128 "$MISTRAL_CAP" ""
bounds "$MG" museglimmer30b wildvision ceiling "$TF515" 128 "" low
bounds "$MG" museglimmer30b wildvision floor   "$TF515" 128 "$MG_CAP" low

# ---- leftovers ---------------------------------------------------------------
g4 cvbench corrected 0.25
echo "PRESHUTDOWN_CHAIN_COMPLETE $(date)"
