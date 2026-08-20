#!/usr/bin/env bash
# InternVL3-8B on the full RealWorldQA test split (765), three arms.
#
# Same benchmark the Qwen2.5-VL work measured, and the same `score_answer`, so the two model
# families can be read against each other. Qwen's reference points, full 765: 32B baseline 68.89%,
# 72B baseline 72.29% (analysis/experiments/qwen25vl_keeprate_sweep_results.md). Those are a
# different model family at different scales -- useful as an order-of-magnitude check on our
# ceiling, not as a target.
#
# Preservation (arm / ceiling) is the figure to lead with; recovery divides by the floor-ceiling gap
# and stops meaning anything when that gap is narrow, which is a property of the data rather than
# the method.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-internvl
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/internvl_realworldqa
MODEL="${MODEL:-OpenGVLab/InternVL3-8B-hf}"
mkdir -p "$LOG"; cd "$REPO" || exit 1

run() {
  local tag=$1; shift
  echo "##### START $tag @ $(date +%F' '%H:%M:%S) #####"
  for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
  sleep 4
  HF_TOKEN="${HF_TOKEN:?}" CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 43200 \
    "$PY" analysis/experiments/internvl_oracle.py --model "$MODEL" --full \
      --out-json "$LOG/${tag}.json" "$@" > "$LOG/${tag}.log" 2>&1
  echo "##### END $tag rc=$? @ $(date +%F' '%H:%M:%S) #####"
  grep -aoE '=== Final Summary: .*' "$LOG/${tag}.log" | tail -1
}

run ceiling   --arm ceiling
run floor     --arm floor
run corrected --arm corrected --keep-ratio 0.55
echo "INTERNVL_REALWORLDQA_DONE"
