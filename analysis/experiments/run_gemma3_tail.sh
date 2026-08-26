#!/usr/bin/env bash
# The last six table cells: Gemma 3 on realworldqa and chartqa, floor + ours(25/50).
#
# Runs on GPU 1 in parallel with the main campaign on GPU 0, which is still working through refcoco.
# Order matters and is not alphabetical: the main campaign's dataset order is
#   ... gqa -> refcoco -> realworldqa -> mmmu -> chartqa
# so it reaches realworldqa SOONER than chartqa. Doing realworldqa first here gives the arm with the
# smaller time buffer the earlier start, and refcoco (8811 samples x 4 arms x 2 models) is hours of
# cover for both.
#
# The collision this avoids: `run_accuracy_campaign.sh` skips an arm whose json already exists, so
# once a cell lands here the campaign steps over it. What it cannot do is notice an arm that is
# currently mid-run somewhere else -- the json does not exist until the end. Hence the ordering
# rather than a lock; if the campaign ever does catch up, kill the duplicate rather than letting two
# processes write the same output.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-1}
BANDS=4   # NOT `GROUPS`: bash owns that name and the assignment silently does nothing.

arm_run () {   # arm_run <dataset> <tag> <extra args...>
  local ds=$1 tag=$2; shift 2
  local out=analysis/results/gemma3_${ds}; mkdir -p "$out"
  if [ -s "$out/$tag.json" ]; then echo "[skip ] gemma3/$ds/$tag"; return; fi
  echo "[start] gemma3/$ds/$tag  $(date +%H:%M:%S)"
  $PY analysis/experiments/gemma3_oracle.py --dataset "$ds" --full --level 2 \
      --out-json "$out/$tag.json" "$@" > "$out/$tag.log" 2>&1
  echo "[done ] gemma3/$ds/$tag rc=$? $(date +%H:%M:%S)  $(grep -aoE '"accuracy": [0-9.]+' "$out/$tag.log" | head -1)"
}

for DS in realworldqa chartqa; do
  arm_run "$DS" floor --arm floor
  for K in 0.25 0.50; do
    arm_run "$DS" "interleaved_g${BANDS}_k${K}" --arm interleaved --keep "$K" --groups "$BANDS"
  done
done
echo "GEMMA3_TAIL_COMPLETE $(date)"
