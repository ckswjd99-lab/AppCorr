#!/usr/bin/env bash
# v6 = BOX-vs-BICUBIC floor sensitivity probe (user-approved), then the v5 chain.
# Probe: qwen35-35B RealWorldQA floor, SAME 50 strided indices, only the
# downsampling filter differs. Paired per-sample comparison decides whether the
# table's bicubic-floor numbers need re-measuring (flips, not means -- n=50
# means are powerless at 1-2pp).
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
unset HF_TOKEN
PROBE=analysis/results/degrade_filter_probe
mkdir -p "$PROBE"

while pgrep -f "_oracle.py --dataset" > /dev/null; do sleep 30; done

for FILT in box bicubic; do
  python analysis/experiments/qwen35_accuracy.py --dataset realworldqa \
      --arms floor --samples 50 --degrade-filter "$FILT" \
      --out "$PROBE/$FILT" > "$PROBE/$FILT.log" 2>&1
  echo "PS: probe qwen35/rwqa floor filt=$FILT rc=$? $(grep -aoE '"acc": [0-9.]+' $PROBE/$FILT.log | head -1) $(date)"
done
python - <<'PYEOF'
import json
def load(p):
    return {json.loads(l)["i"]: json.loads(l) for l in open(p) if l.strip() and "skip" not in l}
a = load("analysis/results/degrade_filter_probe/box/realworldqa_floor.jsonl")
b = load("analysis/results/degrade_filter_probe/bicubic/realworldqa_floor.jsonl")
common = sorted(set(a) & set(b))
flips = [(i, a[i]["ok"], b[i]["ok"]) for i in common if a[i]["ok"] != b[i]["ok"]]
pred_diff = sum(1 for i in common if a[i]["pred"].strip() != b[i]["pred"].strip())
print(f"PROBE_RESULT n={len(common)} score_flips={len(flips)} "
      f"(box_wins={sum(1 for _, x, y in flips if x > y)}, "
      f"bicubic_wins={sum(1 for _, x, y in flips if y > x)}) "
      f"pred_text_diff={pred_diff}", flush=True)
PYEOF
exec bash scripts/preshutdown_chain_v5_20260828.sh
