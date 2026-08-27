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


def _to_float(s):
    m = re.search(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(m.group()) if m else None


class ChartQASpec:
    """ChartQA (test ~2500). RELAXED accuracy: numeric answers correct within 5% relative tolerance,
    text answers exact-match after VQA normalization. Single gold answer."""
    name = "chartqa"
    hf = "lmms-lab/ChartQA"
    split = "test"

    def load(self, load_dataset):
        return load_dataset(self.hf, split=self.split)

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        image = ex["image"].convert("RGB")
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        image_r = image.resize((tw, th), Image.BILINEAR)
        prompt = ex["question"].strip() + "\nAnswer the question using a single word or phrase."
        return image_r, prompt, str(ex["answer"])

    def score(self, pred_text, gold):
        g, p = _to_float(gold), _to_float(pred_text)
        if g is not None and p is not None:
            ok = int(abs(p - g) <= 0.05 * abs(g)) if g != 0 else int(abs(p) < 1e-9)
        else:
            ok = int(_vqa_norm(pred_text) == _vqa_norm(gold))
        return ok, float(ok)


def _levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class DocVQASpec:
    """DocVQA (config 'DocVQA', validation ~5349, document scans). Official metric = ANLS (Average
    Normalized Levenshtein Similarity), threshold 0.5, max over the multiple gold answers.
    Headline metric = mean ANLS."""
    name = "docvqa"
    hf = "lmms-lab/DocVQA"
    split = "validation"

    def load(self, load_dataset):
        return load_dataset(self.hf, "DocVQA", split=self.split)

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        image = ex["image"].convert("RGB")
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        image_r = image.resize((tw, th), Image.BILINEAR)
        prompt = ex["question"].strip() + "\nAnswer the question using a single word or phrase."
        return image_r, prompt, list(ex["answers"])

    def score(self, pred_text, gold):
        p = pred_text.strip().lower()
        best = 0.0
        for a in gold:
            a = str(a).strip().lower()
            ml = max(len(p), len(a))
            nl = _levenshtein(p, a) / ml if ml > 0 else 0.0
            best = max(best, (1.0 - nl) if nl < 0.5 else 0.0)
        return int(best >= 0.5), float(best)


class InfoVQASpec(DocVQASpec):
    """InfographicVQA (config 'InfographicVQA' of lmms-lab/DocVQA, validation). Same ANLS metric /
    prompt as DocVQA; infographic images (large, dense text+graphics)."""
    name = "infovqa"

    def load(self, load_dataset):
        return load_dataset(self.hf, "InfographicVQA", split=self.split)


_YESNO_RE = re.compile(r"\b(yes|no)\b")


class POPESpec:
    """POPE object-hallucination benchmark (test, yes/no 'Is there a X in the image?'). Accuracy on
    yes/no; fits the per-sample driver like the other VQA sets."""
    name = "pope"
    hf = "lmms-lab/POPE"
    split = "test"

    def load(self, load_dataset):
        return load_dataset(self.hf, split=self.split)

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        image = ex["image"].convert("RGB")
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        image_r = image.resize((tw, th), Image.BILINEAR)
        prompt = ex["question"].strip() + "\nAnswer the question using a single word (yes or no)."
        return image_r, prompt, str(ex["answer"]).strip().lower()

    def score(self, pred_text, gold):
        m = _YESNO_RE.search(pred_text.lower())
        pred = m.group(1) if m else ""
        return int(pred == gold), float(pred == gold)


class MMMUSpec:
    """MMMU (college-level multi-discipline, validation). Our pipeline is single-image, so we keep the
    single-image questions only (image_2..7 all None) -- report as 'MMMU single-image subset'. MCQ
    scored on the extracted option letter; open questions on normalized substring match."""
    name = "mmmu"
    hf = "lmms-lab/MMMU"
    split = "validation"

    def load(self, load_dataset):
        return load_dataset(self.hf, split=self.split).filter(lambda ex: ex["image_2"] is None)

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        import ast
        image = ex["image_1"].convert("RGB")
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        image_r = image.resize((tw, th), Image.BILINEAR)
        q = re.sub(r"<image\s*\d+>", "", ex["question"]).strip()
        letters = "ABCDEFGHIJ"
        if ex["question_type"] == "multiple-choice":
            try:
                opts = ast.literal_eval(ex["options"])
            except Exception:
                opts = []
            optstr = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
            prompt = f"{q}\n{optstr}\nAnswer with the option's letter from the given choices directly."
        else:
            prompt = q + "\nAnswer the question using a single word or phrase."
        return image_r, prompt, (str(ex["answer"]).strip(), ex["question_type"])

    def score(self, pred_text, gold):
        ans, qtype = gold
        p = pred_text.strip()
        if qtype == "multiple-choice":
            m = re.search(r"\b([A-J])\b", p) or re.search(r"([A-J])", p)
            pred = m.group(1).upper() if m else ""
            ok = int(pred == ans.upper())
        else:
            ok = int(len(_vqa_norm(ans)) > 0 and _vqa_norm(ans) in _vqa_norm(p))
        return ok, float(ok)


import os

_VSR_IMG_DIR = os.path.join(os.path.dirname(__file__), "_vsr_images")
_TF_RE = re.compile(r"\b(true|false)\b")


class VSRSpec:
    """VSR zeroshot (visual spatial reasoning, test 1222). True/false whether a spatial caption holds.
    Images are COCO train2017 referenced by URL -> pre-cached to `_vsr_images/` (see the downloader);
    rows whose image is missing from the cache are filtered out. Accuracy on true/false."""
    name = "vsr"
    hf = "cambridgeltl/vsr_zeroshot"
    split = "test"

    def load(self, load_dataset):
        ds = load_dataset(self.hf, split=self.split)
        # load_from_cache_file=False, non-negotiably: this filter's result depends on the
        # FILESYSTEM (which images sit in _vsr_images), and `datasets` fingerprints only the
        # lambda -- a cached run from before the image cache was populated returned 0 rows
        # forever after, and a driver on top of it "completed" three arms in 38 seconds with
        # rc=0 and nothing scored.
        return ds.filter(lambda r: os.path.exists(os.path.join(_VSR_IMG_DIR, r["image"])),
                         load_from_cache_file=False)

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        image = Image.open(os.path.join(_VSR_IMG_DIR, ex["image"])).convert("RGB")
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        image_r = image.resize((tw, th), Image.BILINEAR)
        prompt = (f'Based on the image, is the following statement true or false? "{ex["caption"]}"\n'
                  "Answer with a single word: true or false.")
        return image_r, prompt, ("true" if ex["label"] == 1 else "false")

    def score(self, pred_text, gold):
        p = pred_text.strip().lower()
        m = _TF_RE.search(p)
        pred = m.group(1) if m else ("true" if re.search(r"\byes\b", p) else ("false" if re.search(r"\bno\b", p) else ""))
        return int(pred == gold), float(pred == gold)


SPECS = {"refcoco": RefCOCOSpec, "realworldqa": RealWorldQASpec, "gqa": GQASpec,
         "textvqa": TextVQASpec, "chartqa": ChartQASpec, "docvqa": DocVQASpec,
         "infovqa": InfoVQASpec, "pope": POPESpec, "mmmu": MMMUSpec, "vsr": VSRSpec}


def get_spec(name):
    if name not in SPECS:
        raise ValueError(f"unknown dataset {name}; choices: {list(SPECS)}")
    return SPECS[name]()
