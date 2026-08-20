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
    """One image, the annotations to prompt with, and what to ask about it."""
    image_id: int
    file_path: str
    width: int
    height: int
    anns: List[dict]
    # Categories the image is known to contain / verified not to contain. LVIS only; COCO leaves
    # these None and every category present in the GT is prompted.
    pos_cat_ids: List[int] | None = None
    neg_cat_ids: List[int] | None = None
    # What to prompt the detector with, as (result_id, text). `result_id` is what predictions are
    # keyed by, which is NOT always `image_id`:
    #
    #   COCO / LVIS  the prompts are derived from the GT -- one per category present -- and every
    #                prediction belongs to this image, so result_id == image_id and the category is
    #                carried separately.
    #   SA-Co        a datapoint IS an (image, noun-phrase) pair with its own id, and one image
    #                carries ~10 of them. Predictions are keyed per pair, so result_id is the pair's
    #                id and the image is shared. 80% of pairs have no masks at all -- they ask about
    #                a concept that is absent, and answering "none" is the thing being scored. A
    #                harness that only prompts with categories taken from the GT can never be wrong
    #                in that direction and so cannot produce cgF1's IL_MCC term.
    #
    # None means "derive from the GT categories", the COCO/LVIS behaviour.
    prompts: List[tuple] | None = None


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


SACO_GOLD_ANN = f"{DATA_ROOT}/saco_gold/annotations"
SACO_GOLD_IMAGES = f"{DATA_ROOT}/saco_gold/images"
SACO_GOLD_SUBSETS = ("metaclip", "sa1b", "attributes", "crowded",
                     "wiki_common", "fg_food", "fg_sports_equipment")


def build_saco_gold(max_boxes: int, limit: int | None, subset: str = "attributes",
                    require_local: bool = True) -> Benchmark:
    """SA-Co/Gold, one subset. Promptable concept segmentation, scored by cgF1.

    Three things make this unlike COCO/LVIS, and all three are why the harness needed `Item.prompts`:

    **A datapoint is an (image, noun-phrase) pair.** `images` rows carry `text_input`; `categories`
    is a single dummy `object` entry the format requires and the task ignores. One image file appears
    in ~10 rows with different phrases, so items are grouped by file and the vision tower runs once
    per image rather than once per phrase.

    **80% of pairs are negative** -- the phrase does not occur in the image, and answering "none" is
    the thing being measured. They are kept, with empty `anns`.

    **Scoring is `cgF1 = positive_micro_F1 x IL_MCC`**, from Meta's own evaluator, vendored under
    `saco_eval/`. Matching is Hungarian, not COCOeval's greedy. Do not reimplement it.

    The three files per subset are three independent annotators. `CGF1Evaluator` takes all three and
    scores against each, keeping the most favourable -- the "oracle setting" the benchmark defines,
    since annotators genuinely disagree on mask borders, instance counts, and whether the phrase is
    present at all. It also drops any pair not `is_instance_exhaustive` in *all* three. Passing one
    file instead of three is a different, harsher benchmark.
    """
    import json

    if subset not in SACO_GOLD_SUBSETS:
        raise SystemExit(f"unknown SA-Co/Gold subset {subset!r}; have {SACO_GOLD_SUBSETS}")
    gt_paths = [f"{SACO_GOLD_ANN}/gold_{subset}_merged_{v}_release_test.json" for v in "abc"]
    missing_gt = [p for p in gt_paths if not os.path.exists(p)]
    if missing_gt:
        raise SystemExit(f"missing SA-Co/Gold annotations: {missing_gt}")

    # Annotator 'a' defines the prompt list; the other two are only ever used by the evaluator.
    gt = json.load(open(gt_paths[0]))
    anns_by_row = {}
    for a in gt["annotations"]:
        anns_by_row.setdefault(a["image_id"], []).append(a)

    by_file: Dict[str, List[dict]] = {}
    for row in gt["images"]:
        by_file.setdefault(row["file_name"], []).append(row)

    items, missing_img = [], 0
    for fname, rows in by_file.items():
        path = os.path.join(SACO_GOLD_IMAGES, fname)
        if require_local and not os.path.exists(path):
            missing_img += 1
            continue
        r0 = rows[0]
        items.append(Item(
            image_id=r0["id"], file_path=path, width=r0["width"], height=r0["height"],
            anns=[a for r in rows for a in anns_by_row.get(r["id"], [])],
            prompts=[(r["id"], r["text_input"]) for r in rows],
        ))
    if missing_img:
        print(f"[saco_gold:{subset}] {missing_img} image files not on disk and skipped", flush=True)
    if limit:
        items = items[:limit]

    def evaluate(results, image_ids):
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "saco_eval"))
        from cgf1_eval import CGF1Evaluator
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            pred_path = f.name
        try:
            ev = CGF1Evaluator(gt_path=gt_paths, iou_type="segm", verbose=True)
            # Score only the prompts actually run. CGF1Evaluator walks `self.eval_img_ids`, which is
            # every instance-exhaustive pair in the whole subset, and a pair with no predictions
            # counts as IL_FN if it is positive. On a 200-image slice of `attributes` that meant 757
            # prompts scored against 9,222, driving IL_recall to 0.135 and cgF1 to 0.015 -- a
            # bookkeeping artifact that reads as total model failure. Same trap as COCOeval's
            # `params.imgIds`, which this evaluator has no equivalent of.
            ran = {r["image_id"] for r in results}
            before = len(ev.eval_img_ids)
            ev.eval_img_ids = [i for i in ev.eval_img_ids if i in ran]
            if len(ev.eval_img_ids) != before:
                print(f"[saco_gold] scoring {len(ev.eval_img_ids)} of {before} exhaustive prompts "
                      f"(the ones this run produced predictions for)", flush=True)
            if not ev.eval_img_ids:
                raise SystemExit("no scored prompts: predictions and GT prompt ids do not intersect")
            summary = ev.evaluate(pred_path)
        finally:
            os.unlink(pred_path)
        pick = {"cgF1": "cgF1_eval_segm_cgF1",
                "IL_MCC": "cgF1_eval_segm_IL_MCC",
                "positive_micro_F1": "cgF1_eval_segm_positive_micro_F1",
                "precision": "cgF1_eval_segm_precision",
                "recall": "cgF1_eval_segm_recall",
                "IL_F1": "cgF1_eval_segm_IL_F1",
                "IL_FPR": "cgF1_eval_segm_IL_FPR"}
        return {k: float(summary[v]) for k, v in pick.items() if v in summary}

    n_pos = sum(1 for it in items for pid, _ in it.prompts if anns_by_row.get(pid))
    n_prompt = sum(len(it.prompts) for it in items)
    return Benchmark(f"saco_gold:{subset}", items, {}, evaluate,
                     {"annotations": gt_paths, "images_missing": missing_img,
                      "num_prompts": n_prompt, "num_positive_prompts": n_pos,
                      "subset": subset})


BUILDERS = {"coco": build_coco, "lvis": build_lvis, "saco_gold": build_saco_gold}


def build(name: str, max_boxes: int, limit: int | None, **kw) -> Benchmark:
    """`name` may carry a sub-selection after a colon, e.g. "saco_gold:crowded"."""
    name, _, sub = name.partition(":")
    if name not in BUILDERS:
        raise SystemExit(f"unknown dataset {name!r}; have {sorted(BUILDERS)}")
    if sub:
        kw["subset"] = sub
    return BUILDERS[name](max_boxes, limit, **kw)
