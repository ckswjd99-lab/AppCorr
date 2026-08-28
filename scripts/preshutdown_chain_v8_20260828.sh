#!/usr/bin/env bash
# v8 (user: "table isn't filling" -- 09:00): deterministic single queue,
# TABLE-CELL work first. Fixes my omission of qwen3.5's keep arms on the new
# benches (the standing 25/50 rule), pulls gemma4 cvbench k0.25 forward, then
# WV dumps for the new models, then the box remeasure WITHOUT chartqa (document
# bench = lowest priority per user), then the MG mmvp catch-up.
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
QOUT=analysis/results/qwen35_accuracy
BOX=analysis/results/qwen35_accuracy_box

while pgrep -f "_oracle.py --dataset" > /dev/null; do sleep 30; done
gate () { [ "$(date +%s)" -lt "$DEADLINE" ]; }

q35k () {  # dataset keep
  local DS=$1 K=$2
  gate || { echo "PS: DEADLINE, skip qwen35/$DS k$K"; return; }
  [ -s "$QOUT/${DS}_streaming_g4_k${K}.jsonl" ] && { echo "PS: skip qwen35/$DS k$K"; return; }
  python analysis/experiments/qwen35_accuracy.py --dataset "$DS" --arms streaming \
      --keep "$K" --out "$QOUT" > "$QOUT/run_${DS}_k${K}.log" 2>&1
  echo "PS: qwen35/$DS k$K rc=$? $(grep -aoE '"acc": [0-9.]+' $QOUT/run_${DS}_k${K}.log | tail -1) $(date)"
}

g4 () {
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

bounds () {
  local MID=$1 TAG=$2 DS=$3 ARM=$4 TFP=$5 MNT=$6 CAP=$7
  gate || { echo "PS: DEADLINE, skip $TAG/$DS $ARM"; return; }
  local out=analysis/results/${TAG}_${DS}/${ARM}.json
  [ -s "$out" ] && { echo "PS: skip $TAG/$DS $ARM"; return; }
  mkdir -p "$(dirname "$out")"
  local extra=()
  [ -n "$TFP" ] && extra+=(--transformers-path "$TFP")
  [ -n "$CAP" ] && extra+=(--degrade-max-px "$CAP")
  python analysis/experiments/vlm_bounds_oracle.py --model "$MID" --dataset "$DS" \
      --arm "$ARM" --full --max-new-tokens "$MNT" --out-json "$out" "${extra[@]}" \
      > "${out%.json}.log" 2>&1
  echo "PS: $TAG/$DS $ARM rc=$? $(grep -aoE '"accuracy": [0-9.null]+' ${out%.json}.log | head -1) $(date)"
}

runq_box () {  # dataset arms...   (K env optional)
  local DS=$1; shift
  gate || { echo "PS: DEADLINE, skip box-remeasure $DS $*"; return; }
  local KARG=()
  [ -n "${K:-}" ] && KARG=(--keep "$K")
  python analysis/experiments/qwen35_accuracy.py --dataset "$DS" --arms "$@" \
      --degrade-filter box "${KARG[@]}" --out "$BOX" > "$BOX/run_${DS}_$*${K:+_k$K}.log" 2>&1
  echo "PS: box-remeasure qwen35/$DS $*${K:+ k=$K} rc=$? $(grep -aoE '"acc": [0-9.]+' "$BOX/run_${DS}_$*${K:+_k$K}.log" | tail -2 | tr '\n' ' ') $(date)"
}

# ---- A: the missing qwen35 keep arms on the new benches (25/50 rule) --------
q35k mmvp 0.25; q35k mmvp 0.50
# ---- B: gemma4 cvbench k0.25 pulled forward ---------------------------------
g4 cvbench corrected 0.25
# ---- C: qwen35 cvbench keep arms --------------------------------------------
q35k cvbench 0.25; q35k cvbench 0.50
# ---- D: WV dumps for the new models -----------------------------------------
bounds "$MISTRAL" mistral24b wildvision ceiling "" 128 ""
bounds "$MISTRAL" mistral24b wildvision floor   "" 128 "$MISTRAL_CAP"
bounds "$MG" museglimmer30b wildvision ceiling "$TF515" 128 ""
bounds "$MG" museglimmer30b wildvision floor   "$TF515" 128 "$MG_CAP"
# ---- E: probe2 + box remeasure (chartqa DEFERRED -- document bench) ---------
mkdir -p "$BOX" analysis/results/degrade_filter_probe/pyr
python analysis/experiments/qwen35_accuracy.py --dataset realworldqa \
    --arms floor --samples 50 --degrade-filter pyr \
    --out analysis/results/degrade_filter_probe/pyr > analysis/results/degrade_filter_probe/pyr.log 2>&1
echo "PS: probe qwen35/rwqa floor filt=pyr rc=$? $(grep -aoE '"acc": [0-9.]+' analysis/results/degrade_filter_probe/pyr.log | head -1) $(date)"
python - <<'PYEOF'
import json
def load(p):
    return {json.loads(l)["i"]: json.loads(l) for l in open(p) if l.strip() and "skip" not in l}
a = load("analysis/results/degrade_filter_probe/box/realworldqa_floor.jsonl")
c = load("analysis/results/degrade_filter_probe/pyr/realworldqa_floor.jsonl")
common = sorted(set(a) & set(c))
flips = [i for i in common if a[i]["ok"] != c[i]["ok"]]
pdiff = sum(1 for i in common if a[i]["pred"].strip() != c[i]["pred"].strip())
print(f"PS: PROBE2_RESULT box-vs-pyr n={len(common)} score_flips={len(flips)} pred_text_diff={pdiff}", flush=True)
PYEOF
runq_box mmvp floor streaming
runq_box realworldqa floor streaming
K=0.25 runq_box realworldqa streaming
K=0.50 runq_box realworldqa streaming
runq_box vsr floor
K=0.25 runq_box vsr streaming
K=0.50 runq_box vsr streaming
# ---- F: MG mmvp catch-up (channel-fixed) ------------------------------------
bounds "$MG" museglimmer30b mmvp ceiling "$TF515" 24 ""
bounds "$MG" museglimmer30b mmvp floor   "$TF515" 24 "$MG_CAP"
echo "PRESHUTDOWN_CHAIN_COMPLETE $(date)"
