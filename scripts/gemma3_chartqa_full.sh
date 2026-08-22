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
# Identity gate before any of this is read: at keep=1.0 nothing is approximated, so the corrected
# result must reproduce the exact forward. It does, bit-exact (rel 0.000e+00), and matches the
# ceiling arm's accuracy exactly. Five driver bugs were found by that gate; three earlier ChartQA
# result sets are withdrawn.
#
# 120-sample smoke: ceiling 68.33 | corrected 69.17 | corrected_t 70.00 | corrected_j 73.33 |
# floor 21.67. The corrected arms sit above the ceiling, which at n=120 is inside noise (se ~4.3pp)
# and not separable -- that is what the full set is for. The 47pp floor-ceiling gap is the real
# signal: ChartQA is where the approximation costs the most of anything measured so far.
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
run identity     --arm corrected   --keep 1.0
run corrected    --arm corrected   --keep 0.55
run corrected_t  --arm corrected_t --keep 0.55
run corrected_j  --arm corrected_j --keep 0.55
echo "GEMMA3_CHARTQA_DONE"
