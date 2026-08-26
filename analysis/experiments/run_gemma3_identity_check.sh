#!/usr/bin/env bash
# Does Gemma 3's interleaved arm reproduce the ceiling when it corrects everything?
#
# Why this exists. On RealWorldQA the finished row reads:
#     floor 42.09  <  ceiling 42.88  <  ours k0.25 43.53  <  ours k0.50 44.84
# Both corrected arms BEAT the exact forward, k=0.50 by 15 samples of 765 -- 2.5x the entire
# floor-ceiling gap. NYU shows the identical monotone-but-above-ceiling shape on a different model
# and a different task, so it is not a one-off.
#
# docs/memo/interleaved_correction_contract.md is unambiguous about what that pattern usually means:
# "Any arm that appears to beat the ceiling ... is a leak, not a discovery." The plausible innocent
# explanation -- that a partly-corrected input is a third distribution, neither floor nor ceiling,
# and mild low-pass acts as a regulariser for this task -- is exactly the kind of story that was
# wrong three times today when it was reached for before measuring.
#
# The identity settles it. At g=1 with keep=1.0 EVERY token is recomputed at full resolution over
# the full depth, so the result must equal the exact forward. Not approximately: identically.
#   - reproduces the ceiling  -> the machinery is sound, and "partial correction beats the exact
#                               forward" is a real property worth investigating rather than a bug
#   - does not reproduce it   -> there is a leak, and every number in that row is unusable
#
# g must be 1, not 4. At g>1 even keep=1.0 is not exact: band 0 is corrected only to bounds[0] and
# is never revisited once later bands are corrected. An earlier gate here asserted keep=1.0 alone
# and failed at rel 0.42-0.76 -- the implementation was right and the expectation was wrong.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-1}
OUT=analysis/results/gemma3_realworldqa
mkdir -p "$OUT"

TAG=identity_g1_k1.00
if [ -s "$OUT/$TAG.json" ]; then
  echo "[skip ] $TAG"
else
  echo "[start] $TAG  $(date +%H:%M:%S)"
  $PY analysis/experiments/gemma3_oracle.py --dataset realworldqa --full --level 2 \
      --arm interleaved --keep 1.0 --groups 1 \
      --out-json "$OUT/$TAG.json" > "$OUT/$TAG.log" 2>&1
  echo "[done ] $TAG rc=$? $(date +%H:%M:%S)"
fi

echo
echo "=== identity check: g=1 keep=1.0 must equal the ceiling ==="
$PY - <<'PY'
import json, os
d = "analysis/results/gemma3_realworldqa"
def acc(tag):
    p = os.path.join(d, f"{tag}.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))["summary"]
    return s["accuracy"] * 100, s.get("correct"), s.get("num_samples")
ceil = acc("ceiling")
ident = acc("identity_g1_k1.00")
for name, v in (("ceiling", ceil), ("g=1 keep=1.0", ident)):
    print(f"  {name:<14} {v[0]:6.2f}%  {v[1]}/{v[2]}" if v else f"  {name:<14} missing")
if ceil and ident:
    same = ceil[1] == ident[1]
    print()
    print("  PASS: identity holds at the aggregate." if same else
          f"  FAIL: {ident[1]} vs {ceil[1]} correct -- a leak, not a discovery.")
    print("  NOTE: this compares aggregates. Two arms can tie while individual predictions differ"
          " in both directions -- that is how a Qwen gate passed while broken today. If this"
          " passes, confirm per-sample before trusting the row.")
PY
echo "GEMMA3_IDENTITY_COMPLETE $(date)"
