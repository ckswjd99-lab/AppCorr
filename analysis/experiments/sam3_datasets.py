"""Dataset registry for the SAM 3 approx/correct harness.

The oracle driver was written against COCO with the paths as module constants. This pulls the three
dataset-specific things behind one interface so a new benchmark is a registry entry rather than a
copy of the driver:

    where the images are     — LVIS val draws from *two* COCO splits, so this is a lookup, not a dir
    what counts as an item   — (image_id, annotations) after the dataset's own filtering rules
    how a result is scored   — COCOeval and LVISEval are not interchangeable

**LVIS is not COCO with more classes.** Its evaluator exists because the annotations are federated:
each image is exhaustively annotated for only *some* categories. `not_exhaustive_category_ids` marks
categories where unlabelled instances exist, and `neg_category_ids` marks categories verified absent.
LVISEval ignores predictions for categories outside the image's known set instead of counting them
as false positives, and reports AP split by frequency bucket (rare / common / frequent), which is the
axis the dataset exists to expose. Scoring LVIS with COCOeval silently punishes correct detections of
un-annotated objects, so it is not offered here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

COCO_ROOT = "/home/nxclab/fiftyone/coco-2017"
COCO_VAL_IMAGES = f"{COCO_ROOT}/validation/data"
COCO_INSTANCES = f"{COCO_ROOT}/raw/instances_val2017.json"

DATA_ROOT = "/NHNHOME/share/cjpark/data"
LVIS_VAL_JSON = f"{DATA_ROOT}/lvis/lvis_v1_val.json"
COCO_TRAIN_IMAGES = f"{DATA_ROOT}/coco_train2017/train2017"


@dataclass
class Item:
    """One image and the annotations to prompt with."""
    image_id: int
    file_path: str
    width: int
    height: int
    anns: List[dict]
    # Categories the image is known to contain / verified not to contain. LVIS only; COCO leaves
    # these None and every category present in the GT is prompted.
    pos_cat_ids: List[int] | None = None
    neg_cat_ids: List[int] | None = None


@dataclass
class Benchmark:
    name: str
    items: List[Item]
    cat_name: Dict[int, str]
    evaluate: Callable[[List[dict], List[int]], Dict[str, float]]
    # Extra keys folded into the run summary so a results table records what it was scored against.
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------- #

def build_coco(max_boxes: int, limit: int | None) -> Benchmark:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO(COCO_INSTANCES)
    items: List[Item] = []
    for iid in sorted(coco.getImgIds()):
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=False))
                if a.get("area", 0) > 1 and a.get("bbox")]
        if not anns:
            continue
        info = coco.loadImgs(iid)[0]
        items.append(Item(iid, os.path.join(COCO_VAL_IMAGES, info["file_name"]),
                          info["width"], info["height"], anns[:max_boxes]))
    if limit:
        items = items[:limit]

    def evaluate(results, image_ids):
        ev = COCOeval(coco, coco.loadRes(results), "segm")
        # Score only the images actually run, or a subset is graded against all 5,000 and the AP
        # reads as a model failure rather than a bookkeeping one.
        ev.params.imgIds = list(image_ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()
        return {"mask_AP": float(ev.stats[0]), "mask_AP50": float(ev.stats[1]),
                "mask_AP75": float(ev.stats[2]), "mask_AP_small": float(ev.stats[3]),
                "mask_AP_medium": float(ev.stats[4]), "mask_AP_large": float(ev.stats[5])}

    return Benchmark("coco", items, {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())},
                     evaluate, {"annotations": COCO_INSTANCES})


def build_lvis(max_boxes: int, limit: int | None, require_local: bool = True) -> Benchmark:
    """LVIS v1 val: 19,809 images over 1,203 categories, drawn from BOTH COCO splits.

    `require_local` drops images whose file is not on disk and records how many were dropped --
    without it a partial download silently becomes a different benchmark that still reports an
    "LVIS AP". Only 4,809 of the 19,809 come from val2017; the rest are train2017, and the rare
    categories live mostly in that portion, so a val2017-only run measures almost nothing of what
    LVIS is for.
    """
    from lvis import LVIS, LVISEval, LVISResults

    lvis = LVIS(LVIS_VAL_JSON)
    cats = lvis.load_cats(lvis.get_cat_ids())
    cat_name = {c["id"]: c["name"].replace("_", " ") for c in cats}

    items, missing = [], 0
    for iid in sorted(lvis.get_img_ids()):
        info = lvis.load_imgs([iid])[0]
        fname = os.path.basename(info["coco_url"])
        split = "val2017" if "val2017" in info["coco_url"] else "train2017"
        path = os.path.join(COCO_VAL_IMAGES if split == "val2017" else COCO_TRAIN_IMAGES, fname)
        if require_local and not os.path.exists(path):
            missing += 1
            continue
        anns = [a for a in lvis.load_anns(lvis.get_ann_ids(img_ids=[iid]))
                if a.get("area", 0) > 1 and a.get("bbox")]
        if not anns:
            continue
        items.append(Item(iid, path, info["width"], info["height"], anns[:max_boxes],
                          pos_cat_ids=info.get("not_exhaustive_category_ids"),
                          neg_cat_ids=info.get("neg_category_ids")))
    if missing:
        print(f"[lvis] {missing} images not on disk and skipped", flush=True)
    if limit:
        items = items[:limit]

    def evaluate(results, image_ids):
        ids = list(image_ids)
        ev = LVISEval(lvis, LVISResults(lvis, results, max_dets=300), "segm")
        ev.params.img_ids = ids
        ev.run()
        r = ev.get_results()
        # LVISEval's own keys; APr/APc/APf are the reason to use it at all.
        out = {"mask_AP": float(r["AP"]), "mask_AP50": float(r["AP50"]), "mask_AP75": float(r["AP75"]),
               "mask_AP_small": float(r["APs"]), "mask_AP_medium": float(r["APm"]),
               "mask_AP_large": float(r["APl"])}
        for k in ("APr", "APc", "APf"):
            if k in r:
                out[f"mask_{k}"] = float(r[k])
        return out

    return Benchmark("lvis", items, cat_name, evaluate,
                     {"annotations": LVIS_VAL_JSON, "images_missing": missing,
                      "num_categories": len(cat_name)})


BUILDERS = {"coco": build_coco, "lvis": build_lvis}


def build(name: str, max_boxes: int, limit: int | None) -> Benchmark:
    if name not in BUILDERS:
        raise SystemExit(f"unknown dataset {name!r}; have {sorted(BUILDERS)}")
    return BUILDERS[name](max_boxes, limit)
