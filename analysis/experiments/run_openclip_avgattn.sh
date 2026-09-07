#!/usr/bin/env bash
# OpenCLIP with the RECEIVED-attention score, as the other half of a CLS-vs-avg comparison.
#
# Every OpenCLIP result before 2026-08-26 selected patches with `residual_energy x CLS attention` --
# the CLS token's attention ROW, one row of 257. Every other model in this project uses the COLUMN
# mean: how much attention each token RECEIVES, averaged over layers (`patch_attn_prob_layermean` on
# DINOv3, `vision_patch_attn_layermean` on Gemma 3). CLIP was the odd one out, and nothing in the
# repo recorded why.
#
# There IS a real argument for the CLS row on CLIP specifically: the image embedding IS the CLS
# output, so where CLS looks proxies for contribution to the output. That argument is strongest at
# the final layer and weakens once the rows are averaged across layers -- a patch CLS ignores at
# layer 3 may feed the patch CLS reads at layer 20 -- and it throws away all patch-to-patch
# interaction. Strong enough to keep, not strong enough to assume. So both get measured.
#
# The CLS arms are already on disk (`openclip_imagenet_k*`, `cocoret_k*`). These are their pair:
# same model, same splits, same schedule, same keep rates -- only the server-side signal differs.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH   # run_local.sh calls a bare `python`
OUT=analysis/results/vfm_accuracy
mkdir -p "$OUT"

run () {
  local tag=$1 cfg=$2; shift 2
  # A `Final Summary` line, never the exit code: run_local.sh exits 0 on a missing dataset loader,
  # an invalid device ordinal and a busy port alike.
  if [ -s "$OUT/$tag.log" ] && grep -qa "Final Summary" "$OUT/$tag.log"; then
    echo "[skip ] $tag"; return
  fi
  echo "[start] $tag  $(date +%F' '%H:%M:%S)"
  RECV_PORT=${RECV_PORT:-39990} SEND_PORT=${SEND_PORT:-39991} \
    bash offload/run_local.sh "$cfg" -nw 0 "$@" > "$OUT/$tag.log" 2>&1
  echo "[done ] $tag rc=$?  $(date +%F' '%H:%M:%S)"
  grep -aoE "Final Summary: \{[^}]*\}" "$OUT/$tag.log" | tail -1
}

# Retrieval first: 5000 images against ImageNet's 50000, so the comparison exists sooner if this is
# interrupted. Full splits throughout -- recall depends on the size of the retrieval pool, so a
# subset is not comparable to the CLS arms it is being paired against.
for K in 0.25 0.50; do
  run "cocoret_avgattn_k${K}" offload/config/coco_retrieval_clip_bigg_interleaved_g4_avgattn.json \
      --set appcorr_kwargs.token_keep_ratio="$K"
done
for K in 0.25 0.50; do
  run "openclip_imagenet_avgattn_k${K}" offload/config/imagenet_clip_bigg_interleaved_g4_avgattn.json \
      --set appcorr_kwargs.token_keep_ratio="$K"
done
echo "OPENCLIP_AVGATTN_COMPLETE $(date)"
