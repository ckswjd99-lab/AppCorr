#!/usr/bin/env bash
# Gemma 3 4B on full ChartQA test, four arms.
#
# ChartQA because the gap has to be measurable: Qwen2.5-VL's approx-only collapsed 84.64 -> 29.00
# there, the largest of its twelve benchmarks. A narrow gap measures nothing -- SA-Co/attributes had
# 0.5% and no recovery number could be read from it.
#
# Three corrected arms, differing ONLY in how the vision and LLM selections relate. Same vision
# budget throughout, so the comparison is about the mapping, not the amount:
#   corrected    separate budgets -- the patch score is pooled 16:1 and the LLM runs its own top-k
#   corrected_j  joint, VISION-driven -- a token is corrected if ANY of its 16 patches was
#   corrected_t  joint, TOKEN-driven -- tokens picked first from the pooled score, then exactly
#                their 16 patches corrected, so every selected token is FULLY refreshed
#
# Patch score is the standard residual energy x average attention. An earlier version of this run
# used energy alone; those numbers are discarded.
#
# 24-sample smoke (direction only): floor 25.0 | corrected_j 29.2 | corrected 45.8 | corrected_t 54.2,
# with corrected_j correcting 231 LLM tokens against 141 for the other two. corrected_j recomputing
# 40% MORE and scoring 25pp LOWER is what this run has to confirm: the cost of correcting a token
# from a mixture of fresh and approximate patches rather than a coherent set.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-gemma3
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/gemma3_chartqa
mkdir -p "$LOG"; cd "$REPO" || exit 1

run() {
  local tag=$1; shift
  echo "##### START $tag @ $(date +%F' '%H:%M:%S) #####"
  for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
  sleep 4
  HF_TOKEN="${HF_TOKEN:?}" CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 43200 \
    "$PY" analysis/experiments/gemma3_oracle.py --dataset chartqa --full \
      --out-json "$LOG/${tag}.json" "$@" > "$LOG/${tag}.log" 2>&1
  echo "##### END $tag rc=$? @ $(date +%F' '%H:%M:%S) #####"
  grep -aoE '=== Final Summary: .*' "$LOG/${tag}.log" | tail -1
}

run ceiling      --arm ceiling
run floor        --arm floor
run corrected    --arm corrected   --keep 0.55
run corrected_t  --arm corrected_t --keep 0.55
run corrected_j  --arm corrected_j --keep 0.55
echo "GEMMA3_CHARTQA_DONE"
