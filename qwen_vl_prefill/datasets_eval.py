"""
datasets_eval.py -- per-dataset load / prompt / score, so the prototype's accuracy scripts are
dataset-agnostic. Each spec exposes:
    hf, split                       -- HF dataset id + split
    prepare(ex, smart_resize, factor, min_px, max_px) -> (image_resized_PIL, prompt, gold)
    score(pred_text, gold)          -> (correct: int(0/1), score: float)

RealWorldQA is the small (765) default test set -- full-dataset runs are cheap, so no nr-sanity
caveat. RefCOCO (8811, grounding, IoU) is kept for the fine-grained continuous metric.
"""

import re

from PIL import Image

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
STANDALONE_LETTER_RE = re.compile(r"\b([A-Za-z])\b")


def _parse_bbox(text):
    nums = NUM_RE.findall(text)
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in nums[:4])
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _iou(a, b):
    if a is None:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


GROUNDING_PROMPT = (
    'Locate the region described by: "{expr}". Output ONLY the bounding box as four numbers '
    "x1,y1,x2,y2 (top-left and bottom-right pixel coordinates in this image), with no other text."
)


class RefCOCOSpec:
    name = "refcoco"
    hf = "lmms-lab/RefCOCO"
    split = "val"

    def load(self, load_dataset):
        return load_dataset(self.hf, split=self.split)

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        image = ex["image"].convert("RGB")
        expr = ex["answer"][0] if isinstance(ex["answer"], list) and ex["answer"] else str(ex["answer"])
        bx, by, bw, bh = ex["bbox"]
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        sx, sy = tw / image.width, th / image.height
        gold = (bx * sx, by * sy, (bx + bw) * sx, (by + bh) * sy)  # bbox in resized frame
        image_r = image.resize((tw, th), Image.BILINEAR)
        return image_r, GROUNDING_PROMPT.format(expr=expr), gold

    def score(self, pred_text, gold):
        i = _iou(_parse_bbox(pred_text), gold)
        return int(i > 0.5), i


class RealWorldQASpec:
    name = "realworldqa"
    hf = "lmms-lab/RealWorldQA"
    split = "test"

    def load(self, load_dataset):
        return load_dataset(self.hf, split=self.split)

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        image = ex["image"].convert("RGB")
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        image_r = image.resize((tw, th), Image.BILINEAR)
        return image_r, ex["question"], str(ex["answer"]).strip()

    def score(self, pred_text, gold):
        p = pred_text.strip()
        if len(gold) == 1 and gold.isalpha():          # MCQ letter answer
            m = STANDALONE_LETTER_RE.search(p) or re.search(r"[A-Za-z]", p)
            pred = m.group(1).upper() if (m and m.lastindex) else (m.group(0).upper() if m else "")
            ok = int(pred == gold.upper())
        else:                                           # free-form word/number
            gnorm = re.sub(r"[^a-z0-9]", "", gold.lower())
            pnorm = re.sub(r"[^a-z0-9]", "", p.lower())
            ok = int(len(gnorm) > 0 and gnorm in pnorm)
        return ok, float(ok)


class GQASpec:
    """GQA testdev-balanced (12578 questions over 398 images). Open-ended short-answer VQA -- a
    second, larger, HARDER VQA point than RealWorldQA (MCQ). Instructions and images are separate HF
    configs joined on imageId; the 398 images fit in memory."""
    name = "gqa"
    hf = "lmms-lab/GQA"
    split = "testdev"

    def load(self, load_dataset):
        imgs = load_dataset(self.hf, "testdev_balanced_images", split=self.split)
        self.imap = {r["id"]: r["image"] for r in imgs}
        return load_dataset(self.hf, "testdev_balanced_instructions", split=self.split)

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        image = self.imap[ex["imageId"]].convert("RGB")
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        image_r = image.resize((tw, th), Image.BILINEAR)
        prompt = ex["question"].strip() + "\nAnswer the question using a single word or phrase."
        return image_r, prompt, str(ex["answer"]).strip().lower()

    def score(self, pred_text, gold):
        # GQA exact-match, robustified: gold as a standalone word/phrase in the prediction
        # (handles "No, it is clear." -> matches gold "no"; avoids "no" in "now").
        g = gold.strip().lower()
        ok = int(bool(re.search(r"\b" + re.escape(g) + r"\b", pred_text.lower()))) if g else 0
        return ok, float(ok)


_ARTICLES = {"a", "an", "the"}


def _vqa_norm(s):
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return " ".join(t for t in s.split() if t not in _ARTICLES)


class TextVQASpec:
    """TextVQA (OCR-heavy VQA, validation 5000). Standard VQA soft-accuracy: min(#matching answers/3,
    1) over the 10 human answers, after VQA normalization. Headline metric = mean soft score."""
    name = "textvqa"
    hf = "lmms-lab/textvqa"
    split = "validation"

    def load(self, load_dataset):
        return load_dataset(self.hf, split=self.split)

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        image = ex["image"].convert("RGB")
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        image_r = image.resize((tw, th), Image.BILINEAR)
        prompt = ex["question"].strip() + "\nAnswer the question using a single word or phrase."
        return image_r, prompt, list(ex["answers"])

    def score(self, pred_text, gold):
        p = _vqa_norm(pred_text)
        m = sum(1 for a in gold if _vqa_norm(a) == p)
        sc = min(m / 3.0, 1.0)
        return int(sc >= 0.5), float(sc)


SPECS = {"refcoco": RefCOCOSpec, "realworldqa": RealWorldQASpec, "gqa": GQASpec, "textvqa": TextVQASpec}


def get_spec(name):
    if name not in SPECS:
        raise ValueError(f"unknown dataset {name}; choices: {list(SPECS)}")
    return SPECS[name]()
