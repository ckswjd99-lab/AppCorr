#!/usr/bin/env bash
# BOX re-measurement of every qwen35 degraded arm (probe verdict 2026-08-28:
# 4/50 paired flips, 3:1 toward box -- filter is not neutral). Ceilings are
# untouched (no degradation). Output goes to qwen35_accuracy_box/ -- the
# bicubic-era jsonls stay where they are; mixing files would let the resume
# logic silently blend filters.
# Runs AFTER the v5/v6 sweep chain (waits for its completion marker or its
# death), still under the 17:15 gate; whatever the gate cuts resumes
# post-maintenance via the same skip/resume logic.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
unset HF_TOKEN
DEADLINE=$(date -d "17:15" +%s)
OUT=analysis/results/qwen35_accuracy_box

while pgrep -f "preshutdown_chain_v5" > /dev/null || pgrep -f "_oracle.py --dataset" > /dev/null; do
  sleep 60
done

runq () {  # dataset arms... (keep handled per-call via env K)
  local DS=$1; shift
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "PS: DEADLINE, skip box-remeasure $DS $*"; return; }
  local KARG=()
  [ -n "${K:-}" ] && KARG=(--keep "$K")
  python analysis/experiments/qwen35_accuracy.py --dataset "$DS" --arms "$@" \
      --degrade-filter box "${KARG[@]}" --out "$OUT" > "$OUT/run_${DS}_$*${K:+_k$K}.log" 2>&1
  echo "PS: box-remeasure qwen35/$DS $*${K:+ k=$K} rc=$? $(grep -aoE '"acc": [0-9.]+' "$OUT/run_${DS}_$*${K:+_k$K}.log" | tail -2 | tr '\n' ' ') $(date)"
}

mkdir -p "$OUT"
# Short first, then wide-gap; streaming arms consume the degraded base too.
runq mmvp floor streaming
runq realworldqa floor streaming
K=0.25 runq realworldqa streaming
K=0.50 runq realworldqa streaming
runq vsr floor
K=0.25 runq vsr streaming
K=0.50 runq vsr streaming
runq chartqa floor streaming
K=0.25 runq chartqa streaming
K=0.50 runq chartqa streaming
echo "QWEN35_BOX_REMEASURE_COMPLETE $(date)"
