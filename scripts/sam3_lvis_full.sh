#!/usr/bin/env bash
# SAM 3 detector path on FULL LVIS v1 val (19,626 usable images, 1203 categories), three arms.
#
# Four arms. floor and ceiling are mandatory references for any absolute number; one-shot and
# pre_global are the technique with and without interleaving. Three was the plan when an arm looked
# like ~3 hours; after the vision pass was hoisted out of the prompt loop it is ~56 min, so the
# one-shot comparison is cheap enough to confirm on a long-tail dataset rather than assumed from COCO.
#
# The `lvis` package (0.5.3, 2020) calls `np.float`, removed in numpy 1.24. Patched in place at
# site-packages/lvis/eval.py (original kept as eval.py.orig). It took a completed 19,626-image
# ceiling arm down with it before predictions were being dumped before scoring -- they are now, so
# a scoring failure costs `--score-only`, not a rerun.
#
# NO --max-boxes cap. The 20-box limit was set for COCO's tracker path; on LVIS it discards 38% of
# the rare annotations (1,200 -> 745) by truncating each image's annotation list, which is exactly
# the long tail LVIS exists to measure. Rare is already thin at 1,200 annotations over 178
# categories -- 6.7 each -- so there is nothing to spare.
#
# Subsetting was considered and rejected for the same reason: 5,000 images keeps only 38% of rare
# categories and 21% of rare annotations, which leaves APr too noisy to separate the arms.
#
# LVISEval, not COCOeval: LVIS annotations are federated (each image exhaustively labelled for only
# some categories), so COCOeval would count correct detections of un-annotated objects as false
# positives. LVISEval also reports APr/APc/APf, the axis this dataset exists for.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-sam3
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/sam3_lvis_full
mkdir -p "$LOG"; cd "$REPO" || exit 1

run() {
  local tag=$1; shift
  echo "##### START $tag @ $(date +%F' '%H:%M:%S) #####"
  for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
  sleep 4
  HF_TOKEN="${HF_TOKEN:?}" CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 86400 \
    "$PY" analysis/experiments/sam3_coco_oracle.py --dataset lvis --path detector \
      --full --max-boxes 100000 --det-per-cat 30 --det-max-dets 300 \
      --out-json "$LOG/${tag}.summary.json" "$@" > "$LOG/${tag}.log" 2>&1
  echo "##### END $tag rc=$? @ $(date +%F' '%H:%M:%S) #####"
  grep -aoE '=== Final Summary: .*' "$LOG/${tag}.log" | tail -1
}

run ceiling         --arm ceiling
run floor           --arm floor
run inter_preglobal --arm corrected --keep-ratio 0.55 --groups 4 --bounds pre_global
run oneshot         --arm corrected --keep-ratio 0.55 --groups 1
echo "SAM3_LVIS_FULL_DONE"
