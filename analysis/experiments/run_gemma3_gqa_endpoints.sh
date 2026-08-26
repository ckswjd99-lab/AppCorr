#!/usr/bin/env bash
# Does Gemma 3 on GQA reproduce its own endpoints?
#
# The row reads ceiling 42.84 < floor 42.96 < ours@25 43.23 < ours@50 43.42 -- the approximate-only
# arm scores ABOVE the exact forward, and correcting further moves further above it. Paired analysis
# says none of those differences is distinguishable from zero (floor-vs-ceiling p=0.75, with 846
# samples flipping one way and 832 the other), so the ordering is noise around a flat landscape.
#
# But "it is noise" is a statistical statement, not a mechanistic one, and this repo's rule is to
# measure the mechanism rather than explain the number. So: check the two ENDPOINTS, which are
# identities rather than trends.
#
#   keep=1.0, g=1  MUST equal the ceiling.  Every token recomputed at full resolution over the full
#                  depth -- there is nothing left approximate. A difference here is a leak.
#   keep -> 0      MUST approach the floor.  Correcting (almost) nothing is the approximate pass.
#                  Note `--keep 0` still corrects one token (`max(1, round(keep*n))`), so the match
#                  is "very close", not exact, by construction.
#
# If both hold, the machinery is exact where it can be checked and the flat middle is a property of
# Gemma 3 on GQA rather than a defect. Supporting evidence already gathered: the L2 degradation
# removes MORE relative signal on GQA (0.277) than on TextVQA (0.169), yet TextVQA's gap is 13.1pp
# and GQA's is ~0 -- so the degradation is working and the task simply does not use what it removes.
#
# 1500 samples, not the full 12578: this is an identity check, and an identity that holds on 1500
# paired samples is not going to break on the rest. Compared against ceiling/floor re-scored on the
# SAME indices, never against the full-split numbers.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0}
OUT=analysis/results/gemma3_gqa
N=${N:-1500}
mkdir -p "$OUT"

run () {   # run <tag> <extra args...>
  local tag=$1; shift
  if [ -s "$OUT/$tag.json" ]; then echo "[skip ] $tag"; return; fi
  echo "[start] $tag  $(date +%H:%M:%S)"
  $PY analysis/experiments/gemma3_oracle.py --dataset gqa --num-samples "$N" --level 2 \
      --out-json "$OUT/$tag.json" "$@" > "$OUT/$tag.log" 2>&1
  echo "[done ] $tag rc=$? $(date +%H:%M:%S)  $(grep -aoE '"accuracy": [0-9.]+' "$OUT/$tag.log" | head -1)"
}

run probe_ceiling                --arm ceiling
run probe_floor                  --arm floor
run probe_identity_g1_k1.00      --arm interleaved --keep 1.0  --groups 1
run probe_nearzero_g4_k0.01      --arm interleaved --keep 0.01 --groups 4

echo
echo "=== endpoint check (same $N samples throughout) ==="
$PY - <<'PY'
import json, os
d = "analysis/results/gemma3_gqa"
def get(tag):
    p = f"{d}/{tag}.json"
    if not os.path.exists(p): return None
    s = json.load(open(p))["summary"]
    return s["accuracy"]*100, s.get("correct"), s.get("num_samples")
rows = [("ceiling","probe_ceiling"), ("g=1 keep=1.0","probe_identity_g1_k1.00"),
        ("floor","probe_floor"), ("g=4 keep=0.01","probe_nearzero_g4_k0.01")]
vals = {}
for lbl, tag in rows:
    v = get(tag); vals[lbl] = v
    print(f"  {lbl:<15} {v[0]:6.2f}%  {v[1]}/{v[2]}" if v else f"  {lbl:<15} missing")

def cmp(a, b, what):
    va, vb = vals.get(a), vals.get(b)
    if not (va and vb): return
    dc = va[1] - vb[1]
    print(f"  {what}: {a} - {b} = {dc:+d} samples ({va[0]-vb[0]:+.2f}pp)")
print()
cmp("g=1 keep=1.0", "ceiling", "TOP endpoint (must be 0)")
cmp("g=4 keep=0.01", "floor", "BOTTOM endpoint (must be near 0)")
PY
echo "GQA_ENDPOINTS_COMPLETE $(date)"
