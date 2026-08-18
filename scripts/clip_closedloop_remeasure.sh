#!/usr/bin/env bash
# Re-measure the CLIP AppCorr floor / ceiling / interleaved-g4 arms on the closed-loop
# transmission fix (378e21d), i.e. on experiment/clip-appcorr rebased onto main.
#
# Full datasets only: all 50,000 ImageNet val images, all 5,000 COCO val2017 images.
# The ceiling (sequential) was never run at full scale on the original branch -- it is measured
# here so the interleaved number is interpretable for the first time.
set -u

REPO=/NHNHOME/share/cjpark/AppCorr-clip-closedloop
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOGDIR="$REPO/logs/clip_closedloop"
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

run() {  # tag  driver  config  scale-flag
  local tag=$1 driver=$2 cfg=$3 scale=$4
  local log="$LOGDIR/${tag}.log"
  echo "##### START $tag @ $(date +%F_%H:%M:%S) #####"
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
    "$PY" "analysis/experiments/${driver}" \
      --config "offload/config/${cfg}" $scale --device cuda:0 \
      > "$log" 2>&1
  echo "##### END   $tag rc=$? @ $(date +%F_%H:%M:%S) #####"
  grep -aE "top1_acc|R@1|R@5|R@10|Summary" "$log" | tail -8
  echo "ARM_DONE $tag"
}

# --- ImageNet-1k zero-shot, all 50,000 val images ---
run imnet_floor_approx_only  clip_zeroshot_offload_eval.py imagenet_clip_bigg_approx_only_l2.json --full
run imnet_ceiling_sequential clip_zeroshot_offload_eval.py imagenet_clip_bigg_sequential.json      --full
run imnet_interleaved_g4     clip_zeroshot_offload_eval.py imagenet_clip_bigg_interleaved_g4.json  --full

# --- COCO captions retrieval, all 5,000 val2017 images ---
run coco_floor_approx_only   clip_coco_retrieval_offload_eval.py coco_retrieval_clip_bigg_approx_only_l2.json --full
run coco_ceiling_sequential  clip_coco_retrieval_offload_eval.py coco_retrieval_clip_bigg_sequential.json      --full
run coco_interleaved_g4      clip_coco_retrieval_offload_eval.py coco_retrieval_clip_bigg_interleaved_g4.json  --full

echo "CLIP_CLOSEDLOOP_ALL_DONE @ $(date +%F_%H:%M:%S)"
