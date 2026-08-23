#!/usr/bin/env bash
# Accuracy campaign for the evaluation table: floor / ours(30%) / ours(50%) / ceiling, on every
# dataset both VLMs can actually load.
#
# Two recompute rates, 25% and 50% -- the pair the evaluation table reports.
#
# Ordered CHEAPEST FIRST, deliberately. The full matrix is on the order of days on one GPU, so the
# order decides what exists if it is stopped early -- and a completed cheap dataset is worth more
# than four half-finished expensive ones. Cost is roughly n x image size: POPE's 9000 small images
# are cheaper per dataset than DocVQA's 5349 page scans.
#
# Within a dataset the arms run ceiling -> floor -> ours(30) -> ours(50), so the two bounds that
# make the middle interpretable land before the middle does. An arm whose json already exists is
# skipped, so re-running the script resumes rather than restarts.
#
# VSR is absent on purpose: its spec filters to an empty split in this environment.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0}
# NOT `GROUPS`: bash owns that name -- it holds the caller's group ids, so `GROUPS=${BANDS:-4}`
# leaves the array in place and `${GROUPS}` reads its first element (1999 here). The same trap
# already cost a sweep in run_ov2_multidataset.sh; the assert in ov2_oracle.py catches it now, but
# the fix is to not use the name.
BANDS=${BANDS:-4}
# Ordered by INTERPRETABILITY first, cost second. Cheapest-first spent 14 hours on GQA, POPE and
# RefCOCO -- rows whose floor-ceiling gap is 1-4pp, where a preservation rate has no resolution --
# before reaching TextVQA, InfoVQA and DocVQA, the wide-gap rows the table rests on. GQA and
# RefCOCO are extras beyond the table's rows and now run last.
DATASETS=${DATASETS:-"textvqa infovqa docvqa pope gqa refcoco realworldqa mmmu chartqa"}
MODELS=${MODELS:-"ov2 gemma3"}

arm_run () {   # arm_run <model> <dataset> <tag> <extra args...>
  local m=$1 ds=$2 tag=$3; shift 3
  local out=analysis/results/${m}_${ds}; mkdir -p "$out"
  if [ -s "$out/$tag.json" ]; then echo "[skip ] $m/$ds/$tag"; return; fi
  local script=analysis/experiments/ov2_oracle.py
  [ "$m" = "gemma3" ] && script=analysis/experiments/gemma3_oracle.py
  echo "[start] $m/$ds/$tag  $(date +%H:%M:%S)"
  $PY "$script" --dataset "$ds" --full --level 2 --out-json "$out/$tag.json" "$@" \
      > "$out/$tag.log" 2>&1
  echo "[done ] $m/$ds/$tag rc=$? $(date +%H:%M:%S)  $(grep -aoE '"accuracy": [0-9.]+' "$out/$tag.log" | head -1)"
}

for DS in $DATASETS; do
  for M in $MODELS; do
    arm_run "$M" "$DS" ceiling --arm ceiling
    arm_run "$M" "$DS" floor   --arm floor
    for K in 0.25 0.50; do
      arm_run "$M" "$DS" "interleaved_g${BANDS}_k${K}" \
              --arm interleaved --keep "$K" --groups "$BANDS"
    done
  done
done
echo "ACCURACY CAMPAIGN COMPLETE $(date)"
