#!/usr/bin/env bash
# v4 (user directive ~05:10, TOP priority): 30B+ models x {mmvp, wildvision,
# cvbench}. Order maximizes models-x-benches coverage before the 17:15 gate:
# small benches across every ready model first, then cvbench by model, WV last
# of the measured blocks (decode-heavy dumps), Muse Glimmer / Mistral bounds
# inserted as soon as their weights (and the tf515 tree for MG) exist.
# Qwen2.5-32B runs on the GH200 (delegated). gemma3/ov2 re-evals resume
# post-maintenance via the v3 script (skip-if-exists).
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
unset HF_TOKEN
DEADLINE=$(date -d "17:15" +%s)
TF515=/NHNHOME/share/cjpark/tf515
MG=meta-models/Muse-Glimmer-30B
MISTRAL=mistralai/Mistral-Small-3.1-24B-Instruct-2503

while pgrep -f "_oracle.py --dataset" > /dev/null; do sleep 30; done

gate () { [ "$(date +%s)" -lt "$DEADLINE" ]; }

g4 () {  # dataset arm keep("" = bounds arm) [max_new]
  local DS=$1 ARM=$2 K=${3:-} MNT=${4:-24}
  gate || { echo "PS: DEADLINE, skip gemma4/$DS $ARM$K"; return; }
  local out
  if [ -n "$K" ]; then out=analysis/results/gemma4_${DS}/corrected_k${K}.json
  else out=analysis/results/gemma4_${DS}/${ARM}.json; fi
  [ -s "$out" ] && { echo "PS: skip gemma4/$DS $ARM$K"; return; }
  mkdir -p "$(dirname "$out")"
  local args=(--dataset "$DS" --full --arm "${K:+corrected}" --max-new-tokens "$MNT" --out-json "$out")
  [ -z "$K" ] && args=(--dataset "$DS" --full --arm "$ARM" --max-new-tokens "$MNT" --out-json "$out")
  [ -n "$K" ] && args+=(--keep "$K")
  python analysis/experiments/gemma4_oracle.py "${args[@]}" > "${out%.json}.log" 2>&1
  echo "PS: gemma4/$DS $ARM$K rc=$? $(grep -aoE '"accuracy": [0-9.null]+' ${out%.json}.log | head -1) $(date)"
}

q35 () {  # dataset
  local DS=$1
  gate || { echo "PS: DEADLINE, skip qwen35/$DS"; return; }
  local out=analysis/results/qwen35_accuracy
  [ -s "$out/${DS}_ceiling.jsonl" ] && { echo "PS: skip qwen35/$DS"; return; }
  mkdir -p "$out"
  python analysis/experiments/qwen35_accuracy.py --dataset "$DS" \
      --arms floor streaming ceiling --out "$out" > "$out/run.log" 2>&1
  echo "PS: qwen35/$DS rc=$? $(grep -aoE '"acc": [0-9.]+' $out/run.log | tail -3 | tr '\n' ' ') $(date)"
}

bounds () {  # model_id tag dataset arm [tfpath] [max_new]
  local MID=$1 TAG=$2 DS=$3 ARM=$4 TFP=${5:-} MNT=${6:-24}
  gate || { echo "PS: DEADLINE, skip $TAG/$DS $ARM"; return; }
  local out=analysis/results/${TAG}_${DS}/${ARM}.json
  [ -s "$out" ] && { echo "PS: skip $TAG/$DS $ARM"; return; }
  mkdir -p "$(dirname "$out")"
  local extra=()
  [ -n "$TFP" ] && extra=(--transformers-path "$TFP")
  python analysis/experiments/vlm_bounds_oracle.py --model "$MID" --dataset "$DS" \
      --arm "$ARM" --full --max-new-tokens "$MNT" --out-json "$out" "${extra[@]}" \
      > "${out%.json}.log" 2>&1
  echo "PS: $TAG/$DS $ARM rc=$? $(grep -aoE '"accuracy": [0-9.null]+' ${out%.json}.log | head -1) $(date)"
}

mg_ready () { [ -f "$TF515/transformers/__init__.py" ] && \
              ls /NHNHOME/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots/*/model*.safetensors >/dev/null 2>&1; }
mistral_ready () { ls /NHNHOME/huggingface/hub/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/*/model*.safetensors >/dev/null 2>&1; }

# ---- Block 1: mmvp across every model ---------------------------------------
g4 mmvp ceiling; g4 mmvp floor; g4 mmvp corrected 0.25; g4 mmvp corrected 0.50
q35 mmvp
if mistral_ready; then bounds "$MISTRAL" mistral24b mmvp ceiling; bounds "$MISTRAL" mistral24b mmvp floor
else echo "PS: mistral weights not ready, mmvp bounds deferred"; fi
if mg_ready; then bounds "$MG" museglimmer30b mmvp ceiling "$TF515"; bounds "$MG" museglimmer30b mmvp floor "$TF515"
else echo "PS: muse-glimmer not ready, mmvp bounds deferred"; fi

# ---- Block 2: cvbench, big models first -------------------------------------
g4 cvbench ceiling; g4 cvbench floor
q35 cvbench
if mistral_ready; then bounds "$MISTRAL" mistral24b cvbench ceiling; bounds "$MISTRAL" mistral24b cvbench floor; fi
if mg_ready; then bounds "$MG" museglimmer30b cvbench ceiling "$TF515"; bounds "$MG" museglimmer30b cvbench floor "$TF515"; fi
g4 cvbench corrected 0.50

# ---- Block 3: wildvision dumps (decode-heavy, 128 tokens) -------------------
g4 wildvision ceiling "" 128; g4 wildvision floor "" 128; g4 wildvision corrected 0.50 128
if mistral_ready; then bounds "$MISTRAL" mistral24b wildvision ceiling "" 128; bounds "$MISTRAL" mistral24b wildvision floor "" 128; fi
if mg_ready; then bounds "$MG" museglimmer30b wildvision ceiling "$TF515" 128; bounds "$MG" museglimmer30b wildvision floor "$TF515" 128; fi

# ---- Block 4: leftover cvbench arm ------------------------------------------
g4 cvbench corrected 0.25
echo "PRESHUTDOWN_CHAIN_COMPLETE $(date)"