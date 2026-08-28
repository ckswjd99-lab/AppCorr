#!/usr/bin/env bash
# v9: fill the new rows' FLOPs columns. Runs after the v8 accuracy queue.
# qwen35 report (registry datasets work as-is), gemma4 report (new), generic
# full-pass for the bounds-only models. Merge into inprocess_flops.json happens
# in the session (not here) so the json edit is reviewed, not blind.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
unset HF_TOKEN
DEADLINE=$(date -d "17:15" +%s)
TF515=/NHNHOME/share/cjpark/tf515

while pgrep -f "chain_v8" > /dev/null || pgrep -f "_oracle.py --dataset" > /dev/null \
      || pgrep -f "qwen35_accuracy.py" > /dev/null; do sleep 60; done
gate () { [ "$(date +%s)" -lt "$DEADLINE" ]; }

gate && { python analysis/experiments/flops_report_qwen35.py \
    --datasets mmvp cvbench --keeps 0.25 0.50 1.00 --samples 12 \
    --out-json analysis/results/flops/qwen35_flops_newbench.json \
    > analysis/results/flops/qwen35_newbench.log 2>&1
  echo "PS: flops qwen35 newbench rc=$? $(date)"; }
gate && { python analysis/experiments/flops_report_gemma4.py \
    --datasets mmvp cvbench --keeps 0.25 0.50 --samples 12 \
    > analysis/results/flops/gemma4_flops.log 2>&1
  echo "PS: flops gemma4 rc=$? $(date)"; }
gate && { python analysis/experiments/flops_report_generic.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 --tag mistral24b \
    --datasets mmvp cvbench > analysis/results/flops/mistral24b.log 2>&1
  echo "PS: flops mistral24b rc=$? $(date)"; }
gate && { python analysis/experiments/flops_report_generic.py \
    --model meta-models/Muse-Glimmer-30B --tag museglimmer30b \
    --datasets mmvp cvbench --transformers-path "$TF515" \
    > analysis/results/flops/museglimmer30b.log 2>&1
  echo "PS: flops museglimmer30b rc=$? $(date)"; }
echo "FLOPS_FILL_COMPLETE $(date)"
