#!/usr/bin/env bash
# OpenCLIP ImageNet zero-shot accuracy: floor / ceiling / ours(25%,50%), full 50k val.
#
# Split out from run_vfm_accuracy_campaign.sh rather than edited into it, because that script was
# already running and bash reads a script incrementally -- editing it in place corrupts the running
# shell. Chained on its completion MARKER, not on `pgrep -f`, which matches this waiter's own argv.
#
# Was disabled in the VFM campaign because every arm returned ~3% top-1. That was
# `openclip_executor.get_final_results` returning only sample 0 of a 32-image batch while the
# evaluator counted all 32 in the denominator, capping accuracy at 1/32 = 3.125%. Fixed; the g=1,
# keep=1.0 identity gate now reproduces the ceiling bit-identically (90.46875 top-1 on a 640-image
# probe), so the correction path itself is sound. See docs/memo/openclip_correction_and_residual_round.md.
#
# COCO retrieval is included now: `COCOCaptionsLoader` was added to offload/mobile/dataset.py (the
# `coco_captions` name previously resolved to nothing, so every retrieval run died at handshake --
# and still exited 0, so a campaign counted it as done). Caption embeddings come from a cache built
# once by `analysis/experiments/build_coco_caption_embeds.py`; the loader raises if it is absent
# rather than scoring against nothing.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH   # run_local.sh calls a bare `python` that lacks transformers
OUT=analysis/results/vfm_accuracy
mkdir -p "$OUT"

run () {   # run <tag> <config> [--set ...]
  local tag=$1 cfg=$2; shift 2
  if [ -s "$OUT/$tag.log" ] && grep -qa "Final Summary" "$OUT/$tag.log"; then
    echo "[skip ] $tag"; return
  fi
  echo "[start] $tag  $(date +%F' '%H:%M:%S)"
  RECV_PORT=${RECV_PORT:-39990} SEND_PORT=${SEND_PORT:-39991} \
    bash offload/run_local.sh "$cfg" -nw 0 "$@" > "$OUT/$tag.log" 2>&1
  echo "[done ] $tag rc=$?  $(date +%F' '%H:%M:%S)"
  # Read the number from the log, never from anywhere else.
  grep -aoE "Final Summary: \{[^}]*\}" "$OUT/$tag.log" | tail -1
}

# Bounds are NOT re-measured: the table already carries floor 65.92 / ceiling 77.14 for ImageNet
# and i2t 67.96 / t2i 50.70 for retrieval. Those stay literal, so the preservation rates below are
# quoted against numbers from another branch -- re-run them if an ours arm ever lands outside them,
# which is the check that caught VGGT.
for K in 0.25 0.50; do
  run "openclip_imagenet_k${K}" offload/config/imagenet_clip_bigg_interleaved_g4.json \
      --set appcorr_kwargs.token_keep_ratio="$K"
done
# --- COCO val2017 image-text retrieval, same four arms -------------------------------------- #
# Recall depends on the size of the retrieval pool, so these must run over the FULL 5000 images to
# be comparable with published numbers (a 640-image probe scored i2t R@1 86.6 against a literature
# 67.96 purely because the pool was 8x smaller). No `-nr` here, deliberately.
for K in 0.25 0.50; do
  run "cocoret_k${K}" offload/config/coco_retrieval_clip_bigg_interleaved_g4.json \
      --set appcorr_kwargs.token_keep_ratio="$K"
done

echo "OPENCLIP_ACCURACY_COMPLETE $(date)"
