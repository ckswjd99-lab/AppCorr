#!/usr/bin/env bash
# LLaVA-OneVision-2 across datasets. Baselines by default; correction arms only when asked.
#
# floor and ceiling are METHOD-INDEPENDENT. They depend on the model, the dataset and the
# degradation level -- not on how correction is scheduled -- so measuring them broadly is durable
# work that survives any change to the correction arm. The correction arms are not: keep ratio,
# band count and the method itself are all still moving, and every run of them is provisional until
# those settle.
#
# So `ARMS=baselines` (the default) measures only floor and ceiling, and `ARMS=all` adds the
# streaming arm. Run the baselines everywhere first; run correction arms when the configuration is
# final and the numbers are meant to be quoted.
#
# ORDER: cheap and interpretable first. A dataset whose floor-ceiling gap turns out to be narrow
# cannot carry a preservation number at all (Gemma 3's POPE gap was 1.20pp and RealWorldQA's
# 0.78pp, which makes preservation arithmetic rather than evidence), so knowing the gap early is
# what decides whether a correction arm is worth running there later.
#
# Every arm on a dataset runs over the SAME examples, which `--full` guarantees. Existing results
# are reused, never recomputed.
set -u
cd /NHNHOME/share/cjpark/AppCorr-ov2
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0}
# NOT `GROUPS`: that is a bash builtin holding the caller's group ids, so `${GROUPS:-4}` silently
# yields the primary GID (1999 here) and the sweep launches with --groups 1999.
BANDS=${BANDS:-4}
ARMS=${ARMS:-baselines}          # baselines | all
DATASETS=${DATASETS:-"realworldqa pope chartqa textvqa infovqa docvqa"}

run () {   # run <dataset> <tag> <extra args...>
  local ds=$1 tag=$2; shift 2
  local out=analysis/results/ov2_$ds; mkdir -p "$out"
  if [ -s "$out/$tag.json" ]; then echo "[skip ] $ds/$tag"; return; fi
  echo "[start] $ds/$tag  $(date +%H:%M:%S)"
  $PY analysis/experiments/ov2_oracle.py --dataset "$ds" --full --level 2 \
      --out-json "$out/$tag.json" "$@" > "$out/$tag.log" 2>&1
  local rc=$?
  echo "[done ] $ds/$tag  $(date +%H:%M:%S)  rc=$rc  $(grep -aoE '"accuracy": [0-9.]+' "$out/$tag.log" | head -1)"
}

for DS in $DATASETS; do
  run "$DS" ceiling --arm ceiling
  run "$DS" floor   --arm floor
  if [ "$ARMS" = "all" ]; then
    run "$DS" "streaming_g${BANDS}" --arm streaming --groups "$BANDS"
  fi
done
echo "OV2 SWEEP COMPLETE (ARMS=$ARMS) $(date)"
