#!/usr/bin/env bash
# Why does OpenCLIP correction recover so little, and where is its actual ceiling?
#
# Measured on COCO retrieval (5000 images, full split): floor 50.14 i2t, ceiling 67.92, and
# interleaved g=4 recovering 3.0% of that gap at keep=0.25 and 50.4% at keep=0.50. Both far below
# what DINOv3 gets at the same keep rates. Two candidate causes, and they need separating before
# anything is claimed:
#
#   (a) SELECTION -- 50% of patches is simply not enough for CLIP, whatever is chosen. Then keep=1.0
#       would land at the ceiling and the curve is just steep.
#   (b) MULTI-ROUND STALENESS -- structural, and it would cap recovery no matter the keep rate. The
#       fork's own docstring says so: "a round-1 token's cached K/V at layer >= 1 is stale w.r.t.
#       later-round corrections and never revisited". CLIP's tower is bidirectional, so an early
#       group's tokens attend to later groups that were still degraded when their K/V was cached.
#       DINOv3 is bidirectional too, but SAM 3 measures 55% recompute recovering ~90%, so this is
#       not a universal bidirectional penalty -- something about this schedule or this tower.
#
# A 2x2 over (rounds, keep) separates them. keep=1.0 removes selection entirely; g=1 removes
# multi-round staleness entirely (one correction round, everything arrives together).
#
#   g=4 k=0.50  59.10  already on disk -- both effects present
#   g=4 k=1.00     ?   staleness only. If this is at the ceiling, (b) is dead and it is all (a).
#   g=1 k=1.00     ?   NEITHER effect -- the contract's identity gate. Must equal the ceiling
#                      exactly; anything else means the fork is broken, not that the method is
#                      weak. (Held at 640 images: 90.46875 == ceiling to the digit. Never checked
#                      at full scale, and 640 class-sorted images is not a real check.)
#   g=1 k=0.50     ?   selection only. Compare against g=4 k=0.50 and the difference IS the
#                      staleness cost at that keep rate.
#
# Order matters: g=1 k=1.00 runs FIRST. It is the gate -- if the identity does not hold at full
# scale, every other number here is uninterpretable and the rest of the sweep is wasted GPU.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
OUT=analysis/results/vfm_accuracy
CFG=offload/config/coco_retrieval_clip_bigg_interleaved_g4.json
mkdir -p "$OUT"

run () {
  local tag=$1 g=$2 k=$3
  if [ -s "$OUT/$tag.log" ] && grep -qa "Final Summary" "$OUT/$tag.log"; then
    echo "[skip ] $tag"; return
  fi
  echo "[start] $tag  g=$g keep=$k  $(date +%F' '%H:%M:%S)"
  RECV_PORT=${RECV_PORT:-39950} SEND_PORT=${SEND_PORT:-39951} \
    bash offload/run_local.sh "$CFG" -nw 0 \
      --set transmission_kwargs.num_groups="$g" \
      --set appcorr_kwargs.token_keep_ratio="$k" > "$OUT/$tag.log" 2>&1
  echo "[done ] $tag rc=$?  $(date +%F' '%H:%M:%S)"
  grep -aoE "Final Summary: \{[^}]*\}" "$OUT/$tag.log" | tail -1
}

run cocoret_g1_k1.00 1 1.0     # the gate -- expect i2t 67.92, the measured ceiling
run cocoret_g4_k1.00 4 1.0
run cocoret_g1_k0.50 1 0.5

echo "OPENCLIP_KEEP_DIAGNOSIS_COMPLETE $(date)"
