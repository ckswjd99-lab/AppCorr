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


class CVBenchSpec:
    """CV-Bench (NYU, 2638 test): pure perception/geometry MCQ over real photos
    (ADE20K/COCO/Omni3D sources). The dataset ships a ready 'prompt' (question +
    lettered choices) and answers like '(C)' -- scored on the extracted letter."""
    name = "cvbench"
    hf = "nyu-visionx/CV-Bench"
    split = "test"

    def load(self, load_dataset):
        return load_dataset(self.hf, split=self.split)

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        image = ex["image"].convert("RGB")
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        image_r = image.resize((tw, th), Image.BILINEAR)
        prompt = ex["prompt"].strip() + "\nAnswer with the option's letter only."
        gold = re.sub(r"[^A-Da-d]", "", str(ex["answer"])).upper()
        return image_r, prompt, gold

    def score(self, pred_text, gold):
        m = re.search(r"\(([A-Da-d])\)", pred_text) or STANDALONE_LETTER_RE.search(pred_text)
        pred = (m.group(1) if m else "").upper()
        return int(pred == gold), float(pred == gold)


class MMVPSpec:
    """MMVP (300): subtle visual discrimination on CLIP-blind pairs. The HF repo
    stores images as '<Index>.jpg' plus Questions.csv (Index, Question, Options
    '(a) X (b) Y', Correct Answer '(a)'). We read the CSV and open images by
    index -- the imagefolder ordering is filename-sorted and NOT index-aligned,
    so never zip the two datasets together."""
    name = "mmvp"
    hf = "MMVP/MMVP"

    def load(self, load_dataset):
        import csv
        import glob
        import os
        from huggingface_hub import snapshot_download
        root = snapshot_download(self.hf, repo_type="dataset")
        img_dir = os.path.join(root, "MMVP Images")
        rows = []
        with open(os.path.join(root, "Questions.csv"), newline="") as f:
            for r in csv.DictReader(f):
                idx = r["Index"].strip()
                cand = glob.glob(os.path.join(img_dir, f"{idx}.*"))
                if not cand:
                    raise RuntimeError(f"MMVP image missing for index {idx}")
                rows.append({"path": cand[0], "question": r["Question"],
                             "options": r["Options"], "answer": r["Correct Answer"]})
        if len(rows) == 0:
            raise RuntimeError("VACUOUS: MMVP Questions.csv parsed to 0 rows")
        return rows

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        image = Image.open(ex["path"]).convert("RGB")
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        image_r = image.resize((tw, th), Image.BILINEAR)
        prompt = (f"{ex['question'].strip()}\n{ex['options'].strip()}\n"
                  "Answer with (a) or (b) only.")
        gold = re.sub(r"[^ab]", "", ex["answer"].lower())
        return image_r, prompt, gold

    def score(self, pred_text, gold):
        m = re.search(r"\(([ab])\)", pred_text.lower()) or re.search(r"\b([ab])\b", pred_text.lower())
        pred = m.group(1) if m else ""
        return int(pred == gold), float(pred == gold)


class MMERealWorldSpec:
    """MME-RealWorld (lmms-eval export, 23.6k QA): high-resolution real-world
    perception MCQ (five options (A)-(E), answer is the bare letter). Images ride
    as base64 jpeg in 'bytes'. Category breakdown lives in ex['category'] --
    keep it in mind when reading aggregates: OCR/diagram categories are not
    natural photos even though the imagery is high-res real-world capture."""
    name = "mmerealworld"
    hf = "yifanzhang114/MME-RealWorld-Lmms-eval"
    split = "train"          # the export ships everything under 'train'

    def load(self, load_dataset):
        return load_dataset(self.hf, split=self.split)

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        import base64
        import io
        image = Image.open(io.BytesIO(base64.b64decode(ex["bytes"]))).convert("RGB")
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        image_r = image.resize((tw, th), Image.BILINEAR)
        opts = ex["multi-choice options"]
        if isinstance(opts, str) and opts.startswith("["):
            import ast
            opts = ast.literal_eval(opts)
        opts = "\n".join(opts) if isinstance(opts, (list, tuple)) else str(opts)
        prompt = (f"{ex['question'].strip()}\n{opts}\n"
                  "Answer with the option's letter only.")
        return image_r, prompt, str(ex["answer"]).strip().upper()

    def score(self, pred_text, gold):
        m = re.search(r"\(([A-Ea-e])\)", pred_text) or STANDALONE_LETTER_RE.search(pred_text)
        pred = (m.group(1) if m else "").upper()
        return int(pred == gold), float(pred == gold)


class WildVisionSpec:
    """WildVision-Bench (500): real user instructions -- OPEN-ENDED, no reference
    answer, officially judged by a pairwise LLM judge. score() therefore raises:
    use this spec only for generation dumps (degradation A/Bs, judge runs later),
    never inside an accuracy loop."""
    name = "wildvision"
    hf = "WildVision/wildvision-bench"
    config = "vision_bench_0701"
    split = "test"

    def load(self, load_dataset):
        return load_dataset(self.hf, self.config, split=self.split)

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        image = ex["image"].convert("RGB")
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        image_r = image.resize((tw, th), Image.BILINEAR)
        return image_r, ex["instruction"].strip(), ""

    def score(self, pred_text, gold):
        raise NotImplementedError(
            "WildVision has no reference answers; it needs a pairwise judge. "
            "Use generation-dump mode, not the accuracy loop.")




class _VisDroneBase:
    """VisDrone2019-DET-val (548 drone images, ~2000x1500) via the banu4prasad HF mirror --
    YOLO-format labels (class cx cy w h, normalized; classes 0-9 = pedestrian, people, bicycle,
    car, van, truck, tricycle, awning-tricycle, bus, motor). Tiny objects at altitude are exactly
    what a level-2 pyramid destroys, which is why this is on the resolution-sensitive track.
    Rows are DERIVED per (image, category) -- both specs group rows by image path so a driver can
    reuse per-image vision work across the image's questions."""
    repo = "banu4prasad/VisDrone-Dataset"
    # question-name -> label-class set (pedestrian+people merged: the walking/standing split is
    # a detection-annotation nicety no VLM question should depend on)
    CATS = {"people": {0, 1}, "cars": {3}, "vans": {4}, "trucks": {5},
            "buses": {8}, "motorcycles": {9}, "bicycles": {2}}

    def _images(self):
        import glob
        import os
        from huggingface_hub import snapshot_download
        root = snapshot_download(self.repo, repo_type="dataset",
                                 allow_patterns=["VisDrone2019-DET-val/*"])
        base = os.path.join(root, "VisDrone2019-DET-val")
        rows = []
        for ip in sorted(glob.glob(os.path.join(base, "images", "*.jpg"))):
            lp = os.path.join(base, "labels",
                              os.path.basename(ip).rsplit(".", 1)[0] + ".txt")
            if not os.path.exists(lp):
                continue
            boxes = []  # (cls, cx, cy, w, h) normalized
            with open(lp) as f:
                for line in f:
                    t = line.split()
                    if len(t) == 5:
                        boxes.append((int(t[0]), *(float(v) for v in t[1:])))
            rows.append({"path": ip, "boxes": boxes})
        if not rows:
            raise RuntimeError("VACUOUS: VisDrone val parsed to 0 images")
        return rows


class VisDroneCountSpec(_VisDroneBase):
    """Counting: one row per (image, category) with 1 <= N <= 30 ground-truth instances
    (beyond ~30 the GT itself outruns any VLM's counting; below 1 there is nothing to count).
    score = (exact match, soft score 1 - min(1, |pred-N|/N)) -- headline is exact-match rate,
    the soft score plays the mIoU role for graded reporting."""
    name = "visdrone_count"

    def load(self, load_dataset):
        rows = []
        for im in self._images():
            for cname, clset in self.CATS.items():
                n = sum(1 for b in im["boxes"] if b[0] in clset)
                if 1 <= n <= 30:
                    rows.append({"path": im["path"], "cat": cname, "count": n})
        if not rows:
            raise RuntimeError("VACUOUS: no countable (image, category) rows")
        return rows

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        image = Image.open(ex["path"]).convert("RGB")
        prompt = (f"How many {ex['cat']} are in this image? "
                  "Answer with a single number.")
        return image, prompt, int(ex["count"])

    def score(self, pred_text, gold):
        m = re.search(r"\d+", pred_text.replace(",", ""))
        if not m:
            return 0, 0.0
        pred = int(m.group(0))
        return int(pred == gold), max(0.0, 1.0 - min(1.0, abs(pred - gold) / gold))


class VisDroneDetSpec(_VisDroneBase):
    """Unique-instance grounding: one row per (image, category) with EXACTLY one instance --
    "the {singular} " is unambiguous there, so free-form bbox output scores like RefCOCO
    (Acc@IoU0.5, mean IoU) without detection-AP matching machinery. gold is native-pixel
    x1,y1,x2,y2 (converted from the YOLO-normalized labels at load)."""
    name = "visdrone_det"
    SINGULAR = {"people": "person", "cars": "car", "vans": "van", "trucks": "truck",
                "buses": "bus", "motorcycles": "motorcycle", "bicycles": "bicycle"}

    def load(self, load_dataset):
        rows = []
        for im in self._images():
            for cname, clset in self.CATS.items():
                inst = [b for b in im["boxes"] if b[0] in clset]
                if len(inst) == 1:
                    rows.append({"path": im["path"], "cat": cname, "box": inst[0][1:]})
        if not rows:
            raise RuntimeError("VACUOUS: no unique-instance rows")
        return rows

    def prepare(self, ex, smart_resize, factor, min_px, max_px):
        image = Image.open(ex["path"]).convert("RGB")
        W, H = image.size
        cx, cy, w, h = ex["box"]
        gold = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
        prompt = (f"Provide the bounding box of the {self.SINGULAR[ex['cat']]} in this image "
                  "as x1,y1,x2,y2.")
        return image, prompt, gold

    def score(self, pred_text, gold):
        i = _iou(_parse_bbox(pred_text), gold)
        return int(i > 0.5), i


SPECS = {"refcoco": RefCOCOSpec, "realworldqa": RealWorldQASpec, "gqa": GQASpec,
         "textvqa": TextVQASpec, "chartqa": ChartQASpec, "docvqa": DocVQASpec,
         "infovqa": InfoVQASpec, "pope": POPESpec, "mmmu": MMMUSpec, "vsr": VSRSpec,
         "cvbench": CVBenchSpec, "mmvp": MMVPSpec, "mmerealworld": MMERealWorldSpec,
         "wildvision": WildVisionSpec,
         "visdrone_count": VisDroneCountSpec, "visdrone_det": VisDroneDetSpec}


def get_spec(name):
    if name not in SPECS:
        raise ValueError(f"unknown dataset {name}; choices: {list(SPECS)}")
    return SPECS[name]()
