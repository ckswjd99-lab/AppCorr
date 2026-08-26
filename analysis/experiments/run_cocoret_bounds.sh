#!/usr/bin/env bash
# COCO retrieval floor and ceiling -- the two arms an earlier trim wrongly dropped.
#
# The trim rule was "skip a bound when the table already carries a literal our own measurement is
# consistent with". Applied to COCO retrieval it was wrong twice over:
#
#   * There is NO floor literal for these rows. Only a ceiling (i2t R@1 67.96, t2i R@1 50.70). So
#     the floor was skipped on the strength of a value that does not exist, leaving the ours arms
#     with nothing below them. This repo's standing rule is floor AND ceiling, every time, and
#     breaking it is what produced a wrong conclusion about the ADE20K residual round earlier today.
#
#   * The ceiling literal came from a different eval script (the merged `experiment/clip-appcorr`
#     work). `COCOCaptionsLoader` is new as of today and computes recall its own way: a caption
#     embedding cache built once, a retrieval pool restricted to the images actually scored, i2t
#     counted as "any caption of the image in the top k". Those are reasonable choices and they are
#     not necessarily the choices the other script made. Until our own ceiling is measured through
#     THIS loader, `ours / 67.96` compares two protocols rather than two arms.
#
# So both bounds get measured here, through the same loader as the ours arms. If our ceiling lands
# on 67.96 / 50.70 the protocols agree and the literal is corroborated; if it does not, the literal
# describes a different measurement and must not be used as our denominator.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
OUT=analysis/results/vfm_accuracy
mkdir -p "$OUT"

run () {
  local tag=$1 cfg=$2; shift 2
  if [ -s "$OUT/$tag.log" ] && grep -qa "Final Summary" "$OUT/$tag.log"; then
    echo "[skip ] $tag"; return
  fi
  echo "[start] $tag  $(date +%F' '%H:%M:%S)"
  RECV_PORT=${RECV_PORT:-39990} SEND_PORT=${SEND_PORT:-39991} \
    bash offload/run_local.sh "$cfg" -nw 0 "$@" > "$OUT/$tag.log" 2>&1
  echo "[done ] $tag rc=$?  $(date +%F' '%H:%M:%S)"
  grep -aoE "Final Summary: \{[^}]*\}" "$OUT/$tag.log" | tail -1
}

# Full 5000-image split, no -nr: recall depends on the size of the retrieval pool, so a subset is
# not comparable to anything (a 640-image probe scored i2t R@1 86.6 against a literature 67.96
# purely because the pool was 8x smaller).
run cocoret_ceiling offload/config/coco_retrieval_clip_bigg_sequential.json
run cocoret_floor   offload/config/coco_retrieval_clip_bigg_approx_only_l2.json

echo "COCORET_BOUNDS_COMPLETE $(date)"
