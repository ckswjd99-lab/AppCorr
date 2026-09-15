"""Accuracy + serving-time campaign driver for the two-process form: AppCorr vision in THIS
process (appcorr env), the LLM in a vLLM streaming server (`appcorr.vllm_stream.server`,
appcorr-vllm env), chunks over a socket. One driver for both Qwen families:

  --family qwen25vl   Qwen2.5-VL 7B/32B/72B   (windowed tower; `Qwen25VLAxis`)
  --family qwen35     Qwen3.5-35B-A3B / 122B-A10B-FP8 (`Qwen35Axis`)

Arms (all decoded by the SAME mechanism -- vLLM temperature-0 greedy -- so the decode path is
never a confound between arms, the rule qwen35_accuracy.py's docstring explains):

  floor      stock tower on the degraded base, one chunk       (`oneshot_embeds`)
  ceiling    stock tower on the full image, one chunk          (`oneshot_embeds`)
  streaming  approx-then-correct per band, `--groups` chunks, `--keep` fraction corrected
             (`streaming_forward(..., sink=)`); the server prefills band r while the tower is
             still correcting band r+1 -- that overlap is what the timing columns measure.

`--llm-schedule interleaved` runs the same vision path against the engine's `correct` op instead:
the whole approximate prompt goes in at t=0 and each band REWRITES its own rows in place, so the
work left after the last arrival is keep/groups of the image rows plus the text suffix rather than
the last band's chunk (docs/memo/vllm_interleaved_design.md). Its rows are named `interleaved_*`
and sit beside the streaming ones.

Per row the jsonl carries the score AND the server's clock: TTFT measured from the LAST chunk
(what the user waits after the image finished arriving), TTFT from the first chunk, total time,
prompt tokens, chunk count, and the client's vision wall time. `--concurrency N` keeps N requests
in flight (push N, then wait for the oldest) -- the engine batches them, which is the throughput
lever; N=1 is the latency form. `--backend hf` runs the identical loop in-process with the shared
explicit greedy loop (a consistency reference, not a campaign arm).

Same degrade()/get_spec()/record() conventions as qwen35_accuracy.py; output naming
  {dataset}_{model-slug}_{arm|llm-schedule}[_g{groups}[_k{keep}|_auto{theta}]][_c{concurrency}].jsonl
`_auto{theta}` is the adaptive arm (`--keep auto`): a threshold, not a budget, so the
realised k is per sample and lives in the rows (`keep_realised`), not in the file name.
under --out, resumable by row index.

Run (appcorr env; server first, in the appcorr-vllm env, same --model):
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  PYTHONPATH=$PWD <appcorr-vllm python> -m appcorr.vllm_stream.server \
      --model Qwen/Qwen2.5-VL-7B-Instruct --port 5591 --gpu-mem 0.35
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python analysis/experiments/qwen_vllm_accuracy.py \
      --family qwen25vl --model Qwen/Qwen2.5-VL-7B-Instruct --port 5591 --dataset gqa \
      --arms floor streaming ceiling --groups 4 --samples 240
Single-GPU form (2026-09-08, GPU1 off limits): with the vllm backend the HF side loads only the
vision tower + embed_tokens (`--load vision`, default; `appcorr.models.vision_only`, ~3 GB for
122B, bitwise the full model's tower), so the 122B-FP8 server (--gpu-mem ~0.85) and this driver
share GPU0. Throughput form: `--workers 4-8` (forked CPU workers for degrade + image processor;
the 200-420 ms/sample of CPU prep otherwise starves the engine) plus `--concurrency 2-4`.
"""
import argparse, json, os, re, sys, time
from collections import deque
import torch

from appcorr.vllm_stream.bridge import BridgeError  # raised per sink; caught per sample

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from PIL import Image
def degrade(*args, **kwargs):
    """Lazy re-export of `qwen35_accuracy.degrade`.

    That module imports `datasets` (and `qwen_vl_prefill.datasets_eval`) at import time, which
    the served `appcorr-vllm-main` env does not have -- and the paths that only need the AXIS
    (`--tiny`, the composer/prompt gates) never call `degrade`. Importing it here at module level
    made `--tiny --family glm53` unrunnable in the ONLY env that has `transformers.models
    .glm5_next`. Same function, resolved on first call."""
    from analysis.experiments.qwen35_accuracy import degrade as _degrade
    return _degrade(*args, **kwargs)



FAMILIES = ("qwen25vl", "qwen35", "glm46v", "glm53")
# Pixel-AREA cap of each family's image processor (`size["longest_edge"]`), for the
# pyramid-direction rule in `degrade`: degrade relative to min(native, what the model samples).
# Qwen's 16,777,216 is what every qwen number in the table was measured with; GLM-4.6V's
# `Glm46VImageProcessor` caps at 9,633,792 (preprocessor_config.json), which is why the family
# has to say so rather than inherit the Qwen constant. Neither cap binds on the table's
# datasets -- it BINDS first on MME-RealWorld (36 Mpx).
# GLM-5.3-Flash has no `size` dict at all (`Glm5NextImageProcessor.size` is
# `{"longest_edge": 1}` with a `# TODO` saying it is unused): the budget is
# `max_image_tokens` (8000 in processor_config.json), which `smart_resize` turns into
# `tokens * temporal_patch_size * (patch*merge*patch_expand)**2` pixels and compares
# against `aligned_frames(=2 for a still) * H * W` -- so the SPATIAL area cap is
# 8000 * 28**2 = 6,272,000 px. Like the other two it does not bind on the table's
# datasets; it binds first on MME-RealWorld (36 Mpx).
FAMILY_MAX_PX = {"glm46v": 9_633_792, "glm53": 6_272_000}


# Datasets whose answer is a bounding box, not prose. `clean_text`'s markdown line-selection
# rule must not run on them: a GLM-5.3 grounding answer often opens with a preamble line
# ("The user wants to find the bounding box for "the truck" in the image") and the rule keeps
# that line and throws the box away. Measured 2026-09-15 on the stored rows -- box-less rate vs
# the "gen_tokens > 100 but pred < 150 chars" truncation signature: refcoco ceiling 43.2 / 40.6 %,
# refcoco floor 14.7 / 12.9 %, visdrone_det ceiling 52.9 / 51.3 %. They track each other, so the
# box-less rows are our truncation, not the model failing to answer; row i=0 of the visdrone_det
# ceiling has finish_reason=stop, gen_tokens=370 and a 62-character pred. It sank GLM-5.3's
# refcoco ceiling to 39.33 BELOW its own floor at 66.49, and inflated the adaptive arms over the
# ceiling on visdrone_det (on the 173 rows where every arm did emit a box the ordering is the
# ordinary floor 37.57 < adaptive 48.55 < ceiling 50.29). Qwen3.5 and GLM-4.6V are untouched:
# 0.0 % on all 36 of their box arms -- they do not open with a preamble on grounding.
GROUNDING_DATASETS = ("refcoco", "visdrone_det")


def clean_text(family: str, text: str, dataset: str = "") -> str:
    """Model-family answer normalisation applied before scoring. GLM-4.6V wraps its final answer
    in `<|begin_of_box|>...<|end_of_box|>`; every MCQ scorer takes the FIRST A-D letter of the
    upper-cased text, so the 'B' of `<|BEGIN_OF_BOX|>` scored every V*Bench answer as B
    (2026-09-12 22:37: floor == ceiling == 36.13% = the share of gold B). Free-text scorers would
    keep the sentinels inside `pred` and fail exact match. Strip them, keep everything else."""
    if family in ("glm46v", "glm53"):
        # GLM-5.3-Flash's chat template never emits the pair (0 hits in chat_template.jinja),
        # but its tokenizer still carries `<|begin_of_box|>` 154852 / `<|end_of_box|>` 154853 as
        # added tokens, i.e. the trained wrapping behaviour is still reachable. Stripping is a
        # no-op when they are absent and saves the whole family from the V*Bench failure mode
        # (the 'B' of `<|BEGIN_OF_BOX|>` scoring every MCQ answer as B).
        text = text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip()
        # GLM answers free-text questions as "The answer is Pinterest." on 52-63% of InfoVQA
        # rows (GLM-4.6V, 2026-09-13); ANLS / exact-match compare the WHOLE string, so the row
        # scored 37 where the bare answer scores 87. Keep what follows the answer prefix, drop
        # one trailing period.
        # GLM-5.3-Flash answers free-text questions in MARKDOWN and usually keeps talking:
        # "**Pinterest**\n\nAccording to the infographic, ..." or, just as often, the reverse
        # "Looking at the image, I can see one motorcycle ...\n\n**1**".  ANLS / exact match
        # compare the whole string, so a right answer scored 0.69 (bold markers) or 0.00
        # (surrounding prose): InfoVQA's CEILING read 58.59 where the answers are worth 87.52,
        # and the floor was hit harder than the ceiling (a degraded image makes the model more
        # verbose), which widens the floor-ceiling gap and flatters every technique in between.
        #
        # 20-70% of GLM-5.3's free-text rows carry markdown; 7-50% run to several lines.
        # When exactly ONE line is nothing but a bolded span, that line is the answer wherever it
        # sits -- first (InfoVQA) or last (VisDrone counting).  With no such line, only a SHORT
        # first line that is not a lead-in ("...I can count the bicycles:") can be the answer.
        # Verified on 2026-09-14 over every stored GLM row: InfoVQA +901/-0 rows, TextVQA
        # +202/-0, ChartQA +30/-2, VisDrone Count +50/-34 (the losses are counting narratives
        # whose old score came from matching a list index, not an answer), and GLM-4.6V moves by
        # one row in 9,000 -- it does not use this format, so the rule must not disturb it.
        if dataset in GROUNDING_DATASETS:
            # Two things the markdown rule must not touch on a grounding answer.
            # (a) The box may sit on any line, so never select one line.
            # (b) GLM-5.3 emits its OWN `</think>` on sharp frames even though build_inputs
            #     already appends one, and the answer follows it:
            #         "...x1: 735\ny1: 355\nx2: 810\ny2: 410</think>735, 355, 810, 410"
            #     `_parse_bbox` takes the FIRST four numbers, which are the reasoning's
            #     intermediate coordinates, so the row scored its scratch work. Keeping only
            #     what follows the LAST `</think>` moved the 448-row visdrone_det ceiling from
            #     22.32 to 30.13 and the floor from 20.31 to 21.21 (2026-09-15, re-run rows).
            return text.rsplit("</think>", 1)[-1].strip() if "</think>" in text else text
        lines = [l for l in text.split("\n") if l.strip()]
        solo = [m.group(1).strip() for m in (_ONLY_BOLD.fullmatch(l) for l in lines) if m]
        if len(lines) > 1 and len(solo) == 1:
            text = solo[0]
        elif len(lines) > 1:
            first = lines[0].strip()
            if not first.endswith(":") and len(first) <= 120:
                text = first
        text = _BOLD.sub(r"\1", text)
        m = _ANSWER_PREFIX.match(text)
        if m and m.group(1).strip():
            text = m.group(1).strip()
        text = text.rstrip(".").strip() if "\n" not in text else text
    return text


_ANSWER_PREFIX = re.compile(r"^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(.+?)\s*$",
                            re.IGNORECASE | re.DOTALL)
# markdown emphasis GLM-5.3 wraps its answers in, and a line that is NOTHING BUT one such span
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ONLY_BOLD = re.compile(r"\s*\*\*(.+?)\*\*\s*[.:]?\s*$")

def make_axis(family: str, model, proc):
    if family == "qwen25vl":
        from appcorr.models.qwen25vl.unified import Qwen25VLAxis
        return Qwen25VLAxis(model, proc)
    if family == "qwen35":
        from appcorr.models.qwen35.unified import Qwen35Axis
        return Qwen35Axis(model, proc)
    if family == "glm46v":
        from appcorr.models.glm46v.axis import Glm46VAxis
        return Glm46VAxis(model, proc)
    if family == "glm53":
        from appcorr.models.glm53.axis import Glm53Axis
        return Glm53Axis(model, proc)
    raise ValueError(family)


@torch.no_grad()
def greedy_tokens(axis, logits, cache, start_pos, n=24):
    """qwen35_accuracy.greedy, returning token ids (the HF twin of vLLM temperature-0)."""
    toks, cur, pos = [], logits.argmax(-1, keepdim=True), start_pos
    eos = axis.processor.tokenizer.eos_token_id
    for _ in range(n):
        t = int(cur)
        if t == eos:
            break
        toks.append(t)
        pid = torch.full((3, 1, 1), pos, device=cur.device, dtype=torch.long)
        out = axis.model(input_ids=cur, past_key_values=cache, position_ids=pid, use_cache=True)
        cache = out.past_key_values
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        pos += 1
    return toks


# Grounding coordinate convention per family. `GROUNDING_PROMPT` asks for "pixel coordinates in
# this image" and the families do not agree on what that means: the Qwen3 generation and GLM-4.6V
# answer in a 0-1000 relative frame, GLM-5.3-Flash answers in native pixels. Rescaling the wrong
# family destroys the row -- it is a multiply by W/1000, H/1000, so on a 1920x1080 VisDrone image
# it inflates x by 1.92 and y by 1.08 (and by 0.96 / 0.54 on the 960x540 half of the split, which
# is why the error does not look like a single scale factor).
#
# Measured on all 448 VisDrone Det rows of every arm on disk (2026-09-14), scoring each file both
# ways -- Acc@0.5 as scored / with the rescale undone:
#     Qwen3.5-35B     28.6-49.8  /  0.45-0.67      <- relative, rescale is correct
#     Qwen3.5-122B    30.1-51.3  /  0.45
#     GLM-4.6V        23.7-38.4  /  0.22-0.67
#     GLM-5.3-Flash    0.22      /  20.54          <- pixels, rescale was destroying it
# The split is total: no file is ambiguous. GLM-5.3's box cells were withheld from the table as a
# model defect ("emits boxes in a frame the scorer does not share"); they were this bug.
GROUNDING_COORDS = {"qwen25vl": "rel1000", "qwen35": "rel1000",
                    "glm46v": "rel1000", "glm53": "pixel"}


def resize_to_tokens(img, target_tokens: int, factor: int):
    """Force an image to ~`target_tokens` merged vision tokens, aspect ratio preserved.

    Every one of the four served families ingests images the same way: no fixed resolution, an
    AREA cap, and `smart_resize` rounding H and W to `factor = patch_size * merge_size` (Qwen3.5
    16*2 = 32 px per token side, both GLMs 14*2 = 28). One token therefore costs factor**2 pixels,
    and a token budget converts to an area budget exactly:

        area = target_tokens * factor**2

    Passing that as BOTH min_pixels and max_pixels makes smart_resize scale up or down onto it, so
    the resulting grid is target_tokens +- the rounding to whole patches. Specifying the target in
    TOKENS rather than pixels is what makes the four models comparable: the same pixel area gives
    Qwen 1024 px/token and the GLMs 784, a 31 % difference in sequence length -- and sequence
    length, not pixel count, is what Comp / Crit. Comp / Crit. Lat are functions of.
    """
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    area = int(target_tokens) * factor * factor
    h, w = smart_resize(img.height, img.width, factor=factor, min_pixels=area, max_pixels=area)
    return img.resize((w, h), Image.BICUBIC)


def token_factor(proc, family: str = "") -> int:
    """`patch_size * merge_size` of the served processor -- the pixel side one token covers.

    Measured 2026-09-15: Qwen3.5 16*2 = 32, both GLMs 14*2 = 28. GLM-5.3-Flash's processor does
    not load through AutoImageProcessor (`Unrecognized image processor`), so the family table is
    the fallback; its processor_config.json carries patch_size 14, merge_size 2, patch_expand 1.
    """
    ip = getattr(proc, "image_processor", None) or proc
    p, m = getattr(ip, "patch_size", None), getattr(ip, "merge_size", None)
    if p and m:
        return int(p) * int(m)
    fam = {"qwen25vl": 28, "qwen35": 32, "glm46v": 28, "glm53": 28}.get(family)
    if fam:
        return fam
    raise RuntimeError(f"cannot read patch/merge from {type(ip).__name__} and no family given")
    return int(p) * int(m)


def frame_fix_box(pred: str, size) -> str:
    """A box that cannot fit inside the image was not in pixels -- read it as 0-1000.

    GLM-5.3 complies with GROUNDING_PROMPT's "pixel coordinates" on a degraded image and drifts
    to its native 0-1000 frame on the full-resolution one, PER ROW. RefCOCO's images are small
    (640x480 typical), so a 0-1000 box lands outside them and is detectable without the label:
    x2 > w or y2 > h (1 % tolerance) is impossible for a genuine pixel box.

    Measured over all 8811 rows of every GLM-5.3 refcoco arm (2026-09-15) -- as scored / under
    this rule, with the out-of-frame share: floor 72.99 / 74.88 (5.7 %), ceiling 50.49 / 59.49
    (20.0 %), auto50 72.22 / 73.90 (6.8 %), auto25 74.01 / 75.62 (5.5 %). An oracle allowed to
    pick the better frame per row beats the rule by ~1.1 points, so it catches nearly all of it.

    Safe to run unconditionally: on every Qwen3.5 and GLM-4.6V refcoco arm the out-of-frame share
    is 0.0 % and accuracy is unchanged to two decimals, because those families answer in 0-1000
    and `rescale_box` has already mapped them into the frame. The rule only ever fires on a box
    that is already impossible as pixels.

    The residual it cannot see: on a large image (VisDrone 1920x1080) a 0-1000 box fits inside the
    frame, so the same drift would be undetectable. Checked there -- an oracle equals the pixel
    rule exactly on all 448 rows, i.e. GLM-5.3 does not use the normalised frame on VisDrone.
    """
    from qwen_vl_prefill.datasets_eval import _parse_bbox
    b = _parse_bbox(pred)
    if b is None or not size:
        return pred
    w, h = size
    if not (w and h) or (b[2] <= w * 1.01 and b[3] <= h * 1.01):
        return pred
    return (f"{b[0] * w / 1000:.1f},{b[1] * h / 1000:.1f},"
            f"{b[2] * w / 1000:.1f},{b[3] * h / 1000:.1f}")


def rescale_box(pred: str, size, family: str) -> str:
    """Map a grounding answer onto native pixels, per `GROUNDING_COORDS[family]`."""
    if GROUNDING_COORDS.get(family, "rel1000") == "pixel":
        return pred
    nums = re.findall(r"-?\d+\.?\d*", pred)[:4]
    if len(nums) != 4:
        return pred
    w_, h_ = size
    x1, y1, x2, y2 = (float(v) for v in nums)
    return f"{x1 * w_ / 1000:.1f},{y1 * h_ / 1000:.1f},{x2 * w_ / 1000:.1f},{y2 * h_ / 1000:.1f}"


# Row-file name of each LLM schedule. `unified_staged` is the flag the axis takes; its rows are
# named `interleaved_unified` so all three correct-op schedules sort together under --out and
# `make_eval_table` / `flops_report_qwen35` can glob the family.
SCHEDULE_TAG = {"unified_staged": "interleaved_unified"}


def target_suffix(args) -> str:
    """`_t<N>` when the run forces a token budget, so a ladder point cannot collide with the
    native-resolution rows of the same arm."""
    return f"_t{int(args.target_tokens)}" if getattr(args, "target_tokens", 0) else ""


def keep_suffix(args) -> str:
    """The `--keep` half of the progressive arm's row-file name.

    Fixed budget: `_k0.50` as always (and nothing at keep=1.0). Adaptive ("--keep auto"): the
    THRESHOLD, `_auto0.02` -- a per-image k is not a property of the arm, so naming the file after
    a k would be a lie and, worse, would collide with a fixed-k row file. `%g` so 0.02 is
    "0.02" and 2e-05 is "2e-05" (no trailing zeros to drift between runs).
    """
    if args.keep == "auto":
        # bucket 8 is the default lattice and keeps the bare tag; a coarser lattice (user
        # 2026-09-13: "1/4로 올려서도 해봐라") is a different arm and says so in the name
        b = int(args.pscore_bucket)
        lat = "_lat" if getattr(args, "pscore_lattice", None) else ""
        return (f"_auto{args.pscore_threshold:g}" + ("" if b == 8 else f"b{b}") + lat
                + target_suffix(args))
    return (f"_k{args.keep:.2f}" if args.keep < 1.0 else "") + target_suffix(args)


def arm_tag(arm: str, args) -> str:
    """Row-file name of an arm. The progressive arm is named after its LLM SCHEDULE
    (`streaming_g4_k0.50` / `interleaved_g4_k0.50`), so the two schedules' rows sit next to each
    other under --out instead of one resuming from the other's file."""
    if arm != "streaming":
        # bounds arms carry the token budget too, else a ladder's T=4096 ceiling resumes from
        # the T=2048 file and silently reuses it (caught 2026-09-15 after the first rung)
        return arm + target_suffix(args)
    return SCHEDULE_TAG.get(args.llm_schedule, args.llm_schedule)


@torch.no_grad()
def merge_shards(args) -> None:
    """`--merge N`: for every arm, fold `<row file>_sKofN.jsonl` (K = 0..N-1) and whatever the
    canonical row file already holds into the canonical file, one row per i (a later shard
    row never overrides an existing canonical row), sorted by i. The shard files are left in
    place; make_eval_table / vllm_vs_hf_preds read only the canonical file."""
    slug = args.model.split("/")[-1].lower()
    n_sh = int(args.merge)
    for arm in args.arms:
        suffix = ""
        if arm == "streaming":
            suffix = f"_g{args.groups}" + keep_suffix(args)
        if args.concurrency > 1:
            suffix += f"_c{args.concurrency}"
        if args.backend == "hf":
            suffix += "_hf"
        base = os.path.join(args.out, f"{args.dataset}_{slug}_{arm_tag(arm, args)}{suffix}")
        rows = {}
        srcs = [f"{base}.jsonl"] + [f"{base}_s{k}of{n_sh}.jsonl" for k in range(n_sh)]
        counts = {}
        for src in srcs:
            if not os.path.exists(src):
                counts[os.path.basename(src)] = None
                continue
            c = 0
            with open(src) as fh:
                for l in fh:
                    if l.strip():
                        r = json.loads(l)
                        c += 1
                        rows.setdefault(int(r["i"]), r)
            counts[os.path.basename(src)] = c
        if not rows:
            print(f"[{arm}] nothing to merge ({counts})")
            continue
        tmp = f"{base}.jsonl.merging"
        with open(tmp, "w") as fh:
            for i in sorted(rows):
                fh.write(json.dumps(rows[i]) + "\n")
        os.replace(tmp, f"{base}.jsonl")
        sc = [r for r in rows.values() if "skip" not in r]
        acc = 100 * sum(r["ok"] for r in sc) / max(1, len(sc))
        print(f"[{arm}] merged {len(rows)} rows ({len(sc)} scored, acc {acc:.2f}, "
              f"{len(rows) - len(sc)} skipped) from {counts} -> {base}.jsonl", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILIES), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5591)
    ap.add_argument("--arms", nargs="+", default=["floor", "streaming", "ceiling"])
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keep", default="1.0",
                    help="fraction of image tokens corrected (0.25 / 0.50 / 1.0), or 'auto' for "
                         "bucketized THRESHOLD selection: no budget, each band corrects the "
                         "groups whose score clears --pscore-threshold, ceilinged onto the "
                         "1/--pscore-bucket lattice of the band (rows "
                         "`..._g{g}_auto{theta}.jsonl`, each carrying its own `keep_realised`)")
    ap.add_argument("--pscore-threshold", type=float, default=None,
                    help="--keep auto: the GLOBAL score cut theta (one number for every image "
                         "and dataset). Calibrate it offline against a target mean k with "
                         "analysis/experiments/threshold_sim.py on a pscore_dump.py npz")
    ap.add_argument("--pscore-lattice", type=str, default=None,
                    help="opt-in: comma-separated captured CUDA-graph sizes; the last band's "
                         "adaptive count is lifted so its round (groups + text suffix) lands on "
                         "one of them (appcorr.models.qwen_vl_axis.bucket_quota_lattice). Arm "
                         "tag gains `_lat`. Thetas must be calibrated with the same lattice "
                         "(threshold_sim.py --lattice)")
    ap.add_argument("--target-tokens", type=int, default=0,
                    help="resize every image to ~N merged vision tokens before anything else "
                         "(aspect preserved; see resize_to_tokens). 0 = native, the default. "
                         "The row-file name gains `_tN` so a ladder point cannot overwrite "
                         "the native run of the same arm.")
    ap.add_argument("--pscore-bucket", type=int, default=8,
                    help="--keep auto: the 1/b lattice each band's corrected count is ceilinged "
                         "onto (and its floor: a band never corrects fewer than ceil(G_r/b))")
    ap.add_argument("--pscore-score", choices=["rms", "mse"], default="rms",
                    help="--keep auto: energy factor of the score. 'rms' = RMS residual in RAW "
                         "[0,1] pixel units (absolute -- what a global threshold needs) x mean-1 "
                         "received attention; 'mse' = the fixed-k arm's per-image mean-1 score "
                         "(thresholdable only in shape, kept for the A/B and the identity gate)")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--degrade-filter", choices=["bicubic", "box", "pyr"], default="box")
    ap.add_argument("--samples", type=int, default=0, help="0 = full split")
    ap.add_argument("--contiguous", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="requests kept in flight on the server (vllm backend only)")
    ap.add_argument("--think", action="store_true",
                    help="qwen35 / glm46v / glm53: thinking on (all three default OFF; "
                         "GLM's template emits `/nothink` plus an empty <think></think> when it "
                         "is off, so the answer starts at the first generated token)")
    ap.add_argument("--llm-schedule",
                    choices=["streaming", "interleaved", "interleaved_staged", "unified_staged"],
                    default="streaming",
                    help="how the LLM consumes the bands (docs/memo/vllm_interleaved_design.md). "
                         "'streaming' appends band r to the prompt and prefills it once; "
                         "'interleaved' pushes the WHOLE approximate prompt at t=0 and rewrites "
                         "each band's corrected rows in place (`correct` op), so the work after "
                         "the last arrival is keep/groups of the image rows plus the text suffix; "
                         "'interleaved_staged' is the depth-staged form of it (round r corrects "
                         "over the first b_r layers, then walks the image rows through the next "
                         "layer band -- memo §7.11); 'unified_staged' puts the VISION TOWER "
                         "inside that staging (one cost-split axis of tower + decoder layers, so "
                         "the early rounds never reach the LLM and the prompt opens at the "
                         "crossing -- memo §7.12; rows `interleaved_unified_g{g}[_k{keep}]`). "
                         "Names the progressive arm's rows "
                         "`interleaved[_staged]_g{g}[_k{keep}]`")
    ap.add_argument("--no-open-walk", action="store_true",
                    help="unified_staged: keep the engine's stock full-depth prefill at open "
                         "instead of the open walk (A/B gate of the two forms; same state)")
    ap.add_argument("--pscore", choices=["deferred", "eager"], default="deferred",
                    help="keep<1 selection score: 'deferred' computes the received-attention term "
                         "after the first band's push (band 0 ranks on energy alone; default since "
                         "2026-09-09), 'eager' is the pre-2026-09-09 arm (attention before band 0, "
                         "+60 ms on the first push at RWQA size). Towers without deferred support "
                         "run eager either way; rows record which one ran under `pscore`")
    ap.add_argument("--prefetch", type=int, default=0,
                    help="samples to preprocess ahead on a CPU thread (0 = inline); use for "
                         "throughput runs so the HF image processor is not the bottleneck")
    ap.add_argument("--workers", type=int, default=0,
                    help="CPU worker PROCESSES for degrade + image processing (DataLoader, "
                         "forked); unlike --prefetch they do not share the GIL with the vision "
                         "loop. 0 = inline. Recommended 4-8 for throughput runs")
    ap.add_argument("--load", choices=["full", "vision"], default=None,
                    help="HF-side model: 'vision' loads only the tower + embed_tokens "
                         "(appcorr.models.vision_only; ~3 GB for 122B) -- the default for the "
                         "vllm backend, where the decoder never runs; 'full' is forced for hf")
    ap.add_argument("--max-prompt-tokens", type=int, default=None,
                    help="skip (record a `skip` row for) any sample whose prompt + --max-tokens "
                         "exceeds this; default = the server's max_model_len. A 15.5k-token "
                         "TextVQA image crashed the ceiling arm and hung the streaming arm of the "
                         "122B campaign against an 8192 engine (2026-09-08); the user's call is "
                         "to keep the proven 8192 engine and skip past such rows, counted in the "
                         "row file (`skip`, `prompt_tokens`) and in the Final Summary (`skipped`)")
    ap.add_argument("--shard", default=None, metavar="K/N",
                    help="run only every N-th sample starting at K (0-based) and write to "
                         "`<row file>_sKofN.jsonl`: N driver processes, each with its own vision "
                         "tower, overlap their vision passes against the one server (the "
                         "streaming arm is driver-bound: ~3 samples/s vs the engine's 6-10). "
                         "Interleaved, not contiguous, so every shard sees the same image-size "
                         "mix. `--merge N` folds the shard files back into the canonical row file")
    ap.add_argument("--merge", type=int, default=None, metavar="N",
                    help="no runs: merge the N shard files of each arm (plus any rows already in "
                         "the canonical file) into the canonical row file, sorted by i, and exit")
    ap.add_argument("--out", default="analysis/results/qwen_vllm_accuracy")
    args = ap.parse_args()
    # `--keep` is str-typed so "auto" can share the flag with the numeric budgets; resolve it
    # once, here, and everything downstream sees either a float or the string "auto".
    args.keep = "auto" if str(args.keep).lower() == "auto" else float(args.keep)
    if args.keep == "auto" and args.pscore_threshold is None:
        raise SystemExit("--keep auto needs --pscore-threshold (the global score cut)")
    shard = None
    if args.shard is not None:
        k, n_sh = (int(v) for v in args.shard.split("/"))
        if not (0 <= k < n_sh):
            raise SystemExit(f"--shard {args.shard}: need 0 <= K < N")
        shard = (k, n_sh)
    if args.merge is not None:
        return merge_shards(args)
    if args.backend == "hf":
        if args.llm_schedule != "streaming":
            raise SystemExit(f"--llm-schedule {args.llm_schedule} needs the vllm backend: "
                             "rewriting rows of a KV cache the HF model already prefilled is the "
                             "engine step")
        args.concurrency = 1
        args.load = "full"
    elif args.load is None:
        args.load = "vision"
    if args.workers and args.prefetch:
        raise SystemExit("--workers and --prefetch are alternatives; pick one")

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_prefill.datasets_eval import get_spec
    from datasets import load_dataset

    bridge, info = None, {}
    if args.backend == "vllm":
        from appcorr.vllm_stream.bridge import LLMBridge
        bridge = LLMBridge(args.host, args.port)
        info = bridge.info()
        if info["model"] != args.model:
            raise SystemExit(f"server serves {info['model']!r}, driver asked for {args.model!r}")
        print(f"server: {info}", flush=True)
    max_prompt = args.max_prompt_tokens
    if max_prompt is None and info.get("max_model_len"):
        max_prompt = int(info["max_model_len"]) - args.max_tokens
    elif max_prompt is not None:
        max_prompt -= args.max_tokens
    print(f"prompt cap: {max_prompt} tokens (+{args.max_tokens} generated)", flush=True)

    proc = AutoProcessor.from_pretrained(args.model)
    t_load = time.perf_counter()
    if args.load == "vision":
        from appcorr.models.vision_only import load_vision_only
        model = load_vision_only(args.model, device="cuda:0")
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype="auto", device_map="cuda:0").eval()
    print(f"model ({args.load}) loaded in {time.perf_counter() - t_load:.1f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    axis = make_axis(args.family, model, proc)
    axis.pscore_defer = args.pscore == "deferred"
    if args.keep == "auto":
        axis.pscore_threshold = args.pscore_threshold
        axis.pscore_bucket = args.pscore_bucket
        axis.pscore_score = args.pscore_score
        if args.pscore_lattice:
            axis.pscore_lattice = tuple(sorted({int(v) for v in args.pscore_lattice.split(",") if v.strip()}))
    if args.no_open_walk:
        axis.engine_open_walk = False
    tmpl_kw = {"think": True} if (args.think and args.family in ("qwen35", "glm46v", "glm53")) else {}
    family_max_px = FAMILY_MAX_PX.get(args.family)
    spec = get_spec(args.dataset)
    ds = spec.load(load_dataset)
    n = len(ds) if args.samples == 0 else min(args.samples, len(ds))
    idxs = list(range(n)) if args.samples == 0 else \
        (list(range(n)) if args.contiguous else list(range(0, len(ds), max(1, len(ds) // n)))[:n])
    if shard is not None:
        idxs = idxs[shard[0]::shard[1]]
    os.makedirs(args.out, exist_ok=True)
    slug = args.model.split("/")[-1].lower()

    for arm in args.arms:
        suffix = ""
        if arm == "streaming":
            suffix = f"_g{args.groups}" + keep_suffix(args)
        if args.concurrency > 1:
            suffix += f"_c{args.concurrency}"
        if args.backend == "hf":
            suffix += "_hf"
        if shard is not None:
            suffix += f"_s{shard[0]}of{shard[1]}"
        tag = arm_tag(arm, args)
        path = os.path.join(args.out, f"{args.dataset}_{slug}_{tag}{suffix}.jsonl")
        done = set()
        if os.path.exists(path):
            with open(path) as fh:
                done = {json.loads(l)["i"] for l in fh if l.strip()}
        pending = [i for i in idxs if i not in done]
        correct, scored, skipped = 0, 0, {}
        # `correct` is exact-match; `val` is the dataset's own metric (InfoVQA ANLS,
        # TextVQA VQA soft score). Both TABLES read `val`, so the log has to print it too --
        # quoting `acc` for those two understates the ceiling and overstates the floor
        # (InfoVQA floor T=2048 reads 72.30 as exact match against 69.32 ANLS).
        val_sum, val_n = 0.0, 0
        fh = open(path, "a")
        t_arm0 = time.perf_counter()

        def record(i, pred, gold, size, extra):
            nonlocal correct, scored, val_sum, val_n
            if args.dataset in GROUNDING_DATASETS:
                pred = frame_fix_box(rescale_box(pred, size, args.family), size)
            try:
                ok, val = spec.score(pred, gold)
            except NotImplementedError:
                ok, val = 0, None
            correct += ok
            scored += 1
            if val is not None:
                val_sum += float(val); val_n += 1
            row = {"i": int(i), "pred": pred, "gold": gold, "ok": int(ok),
                   "val": (float(val) if val is not None else None)}
            row.update(extra)
            fh.write(json.dumps(row) + "\n")
            if scored % 50 == 0:
                fh.flush()
                el = time.perf_counter() - t_arm0
                print(f"[{arm}] {scored} scored, running {correct / scored * 100:.2f}%  "
                      f"({scored / el:.2f} samples/s)", flush=True)

        def build(i):
            """The sample at NATIVE resolution. --target-tokens is NOT applied here: the
            degraded level has to be built in native coordinates first (see prep)."""
            img, q, gold = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img, q, gold

        def prep(i):
            """CPU side of one sample: degrade + HF image processor (two images for the
            streaming arm). Runs on the prefetch thread when --prefetch > 0."""
            img, q, gold = build(i)
            t0 = time.perf_counter()
            # AGENTS.md "Approx/Correct Contracts", restated in
            # docs/memo/pyramid_degradation_native_vs_canvas.md: the degraded level is built from
            # the ORIGINAL in native coordinates, and only the selected level is scaled onto the
            # model input shape.  Degrading the resized canvas instead is what made COCO's floor
            # equal its ceiling to 1e-4, and it is what this driver did until 2026-09-16: above
            # native the resize is an upscale that carries no information, so degrading it removes
            # the upscale rather than real content and the floor comes out far too strong
            # (measured on Qwen3.5/pyr: 2.27x the high-frequency energy at InfoVQA T=6144).
            base = degrade(img, args.level, args.degrade_filter, max_px=family_max_px)
            if args.target_tokens:
                # degrade() preserves size, so the full and the degraded image enter
                # resize_to_tokens with identical dimensions and land on the same token grid --
                # which the band mixing requires.
                _f = token_factor(proc, args.family)
                img = resize_to_tokens(img, args.target_tokens, _f)
                base = resize_to_tokens(base, args.target_tokens, _f)
                assert img.size == base.size, (img.size, base.size)
            if arm == "streaming":
                inputs = axis.build_inputs(img, q, **tmpl_kw)
                px_base = axis.build_inputs(base, q, **tmpl_kw)["pixel_values"]
            else:
                inputs = axis.build_inputs(img if arm == "ceiling" else base, q, **tmpl_kw)
                px_base = None
            # Prompt layout read here, on the CPU tensors, so the vision pass never reads it
            # back from the GPU (`image_run`, `grid_thw` kwargs of the axis).
            ids = inputs["input_ids"][0]
            pos = (ids == axis.image_token_id).nonzero(as_tuple=True)[0]
            if pos.numel() == 0 or int(pos[-1] - pos[0]) + 1 != pos.numel():
                raise ValueError(f"sample {i}: image tokens are not one contiguous run")
            return {"size": img.size, "gold": gold, "inputs": inputs, "px_base": px_base,
                    "image_run": (int(pos[0]), int(pos.numel())),
                    "grid_thw": tuple(int(v) for v in inputs["image_grid_thw"][0].tolist()),
                    "t_prep_ms": (time.perf_counter() - t0) * 1e3}

        if args.prefetch > 0:
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prep")
            futs = {}

            def prepared(k):
                """prep(pending[k]) with the next --prefetch samples already submitted."""
                for j in range(k, min(k + 1 + args.prefetch, len(pending))):
                    if j not in futs:
                        futs[j] = pool.submit(prep, pending[j])
                return futs.pop(k).result()
        elif args.workers > 0:
            # Forked worker processes run `prep` (PIL degrade + HF image processor, CPU only;
            # the workers never touch CUDA, so inheriting the parent's model handle is fine);
            # the main process only pops ready samples in order. Batch size 1, no collation.
            from torch.utils.data import DataLoader, Dataset

            class _Prep(Dataset):
                def __len__(self):
                    return len(pending)

                def __getitem__(self, k):
                    return prep(pending[k])

            def _worker_init(_):
                # A killed driver used to leave its forked workers alive holding the parent's
                # CUDA context (27.8 GB on GPU0 until killed by PID, 2026-09-08): ask the kernel
                # to SIGTERM them when the parent goes.
                import ctypes, signal
                try:
                    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
                except OSError:
                    pass

            loader_it = iter(DataLoader(_Prep(), batch_size=None, shuffle=False,
                                        num_workers=args.workers, prefetch_factor=4,
                                        multiprocessing_context="fork", persistent_workers=False,
                                        worker_init_fn=_worker_init))
            next_k = 0

            def prepared(k):
                nonlocal next_k
                if k != next_k:
                    raise RuntimeError(f"--workers prep is sequential: asked {k}, next is {next_k}")
                next_k += 1
                return next(loader_it)
        else:
            def prepared(k):
                return prep(pending[k])

        def skip(i, reason, **fields):
            skipped[reason] = skipped.get(reason, 0) + 1
            fh.write(json.dumps({"i": int(i), "skip": reason, **fields}) + "\n")
            fh.flush()
            print(f"[{arm}] skip i={i}: {reason} {fields}", flush=True)

        def too_long(i, p):
            """Prompt cap (see --max-prompt-tokens): checked on the CPU-side input_ids BEFORE
            any GPU work or chunk push, so the engine never sees a prompt it cannot hold."""
            n_prompt = int(p["inputs"]["input_ids"].shape[1])
            if max_prompt is not None and n_prompt > max_prompt:
                skip(i, "prompt_too_long", prompt_tokens=n_prompt, cap=max_prompt)
                return True
            return False

        def vision(p, sink):
            """Run this arm's vision + chunk pushes for a prepared sample; returns (extra-fields,
            HF twin (logits, cache, start_pos) when backend=hf else None)."""
            size, gold = p["size"], p["gold"]
            layout = {"image_run": p["image_run"], "grid_thw": p["grid_thw"]}
            t_h = time.perf_counter()
            inputs = p["inputs"].to("cuda:0")
            hf = None
            if arm == "streaming":
                px_base = p["px_base"].to("cuda:0")
                torch.cuda.synchronize()
                t_loop_h2d = (time.perf_counter() - t_h) * 1e3
                t1 = time.perf_counter()
                lg, kv, st = axis.streaming_forward(inputs, px_base, args.groups, keep=args.keep,
                                                    sink=sink, llm_schedule=args.llm_schedule,
                                                    **layout)
                # `chunks`: the count for the streaming schedule (what every existing row file
                # holds), the RECORDS for the interleaved one -- ("approx", 0, N-1) then
                # ("correct", s, e, |P_r|) per round, which is what the closed-form cost script
                # replays (flops_analytic.interleaved_cost).
                extra = {"corrected_groups": int(st["corrected_groups"]),
                         "chunks": ([list(c) for c in st["chunks"]]
                                    if str(st.get("llm_schedule", "")) in
                                    ("interleaved", "interleaved_staged", "unified_staged")
                                    else len(st["chunks"])),
                         "llm_schedule": st.get("llm_schedule", "streaming")}
                if "pscore" in st:
                    extra["pscore"] = st["pscore"]
                # Adaptive arm: the per-sample k is a RESULT, not a setting -- without these two
                # a row file says nothing about what the arm spent (make_eval_table reads
                # `keep_realised` for the mean/p95 column).
                for f_ in ("keep_realised", "theta", "pscore_score", "pscore_bucket", "pscore_lattice",
                           "pscore_rms_units", "band_selected", "n_groups"):
                    if f_ in st:
                        extra[f_] = st[f_]
                if sink is None:
                    hf = (lg, kv, st["decode_start_pos"])
            else:
                torch.cuda.synchronize()
                t_loop_h2d = (time.perf_counter() - t_h) * 1e3
                t1 = time.perf_counter()
                emb, pos, delta = axis.oneshot_embeds(inputs, inputs["pixel_values"], **layout)
                extra = {"chunks": 1}
                if sink is not None:
                    sink.push(emb, pos, delta, final=True)
                else:
                    pos3 = pos.unsqueeze(1)
                    out = axis.model(inputs_embeds=emb.unsqueeze(0), position_ids=pos3, use_cache=True)
                    hf = (out.logits[:, -1], out.past_key_values, int(pos3.max().item()) + 1)
            torch.cuda.synchronize()
            # t_prep: CPU work that every arm pays and no serving path would put on the critical
            # path (degrade + HF image processor -- two images for the streaming arm);
            # t_vision: the GPU vision pass incl. the chunk pushes -- the number to set beside
            # the server's ttft_open_ms (which for the streaming arm already contains it).
            extra["t_prep_ms"] = p["t_prep_ms"]
            extra["t_loop_h2d_ms"] = t_loop_h2d       # pageable host -> GPU copies of the inputs
            extra["t_vision_ms"] = (time.perf_counter() - t1) * 1e3
            extra["prompt_tokens"] = int(inputs["input_ids"].shape[1])
            # the prompt layout the cost script needs to replay `chunks` (lo, number of merge
            # groups); free here, and not recoverable from the chunk records alone
            extra["image_run"] = [int(v) for v in p["image_run"]]
            return gold, size, extra, hf

        if args.backend == "hf":
            for k, i in enumerate(pending):
                p = prepared(k)
                if too_long(i, p):
                    continue
                try:
                    gold, size, extra, (lg, kv, dp) = vision(p, None)
                    toks = greedy_tokens(axis, lg, kv, dp, args.max_tokens)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    skip(i, "oom")
                    continue
                raw = proc.tokenizer.decode(toks, skip_special_tokens=True)
                extra["raw"] = raw   # the generated text BEFORE clean_text, so rows can be re-scored offline
                record(i, clean_text(args.family, raw, args.dataset), gold, size, extra)
        else:
            inflight = deque()

            def drain_one():
                i, sink, gold, size, extra = inflight.popleft()
                t_w = time.perf_counter()
                try:
                    res = sink.result()
                except BridgeError as e:
                    # The server rejected one of this request's chunks (its error rides the ack
                    # of that push and is raised by this sink alone); the request is already
                    # gone on the server. Record and move on -- one row must not end an arm.
                    skip(i, "bridge_error", error=str(e)[:300])
                    return
                t = res["timing"]
                # Main-loop accounting (the driver is serial per sample): t_loop_prep = blocked
                # on the prep workers, t_loop_wait = blocked in result() for THIS row (its
                # request was already the oldest in flight), t_loop_push = bridge encode+queue.
                extra["t_loop_wait_ms"] = (time.perf_counter() - t_w) * 1e3
                extra["t_loop_push_ms"] = sum(
                    (r["t_queued"] - r["t_send"]) * 1e3 for r in res["pushes"])
                t_start = extra.pop("_t_start")
                # Latency view from the driver's clock (t_start = inputs on the GPU, vision pass
                # about to begin): when the first / last chunk left, and first token from there.
                # ttft_from_open is a server-side duration, so adding it to t_open needs no
                # clock alignment.
                t_open_ms = (res["pushes"][0]["t_send"] - t_start) * 1e3
                extra.update({"ttft_last_chunk_ms": t["ttft_from_last_chunk_ms"],
                              "ttft_open_ms": t["ttft_from_open_ms"], "total_ms": t["total_ms"],
                              "t_open_ms": t_open_ms,
                              "t_last_push_ms": (res["pushes"][-1]["t_send"] - t_start) * 1e3,
                              # every push's send time: push r is band r (band 0 with the leading
                              # text, the last band with the trailing text -- `groups` pushes in
                              # all); band r's correction starts right after push r-1 (push()
                              # syncs the GPU for its D2H copy)
                              "t_pushes_ms": [round((q["t_send"] - t_start) * 1e3, 2)
                                              for q in res["pushes"]],
                              # start of each band's processing on the driver clock (after the
                              # band-spacing sleep = its pixels' arrival); the anchor for
                              # schedules whose bands are not one message each (unified axis)
                              "t_bands_ms": [round((b - t_start) * 1e3, 2)
                                             for b in res.get("bands", [])],
                              # server-side handling time of each push (server perf_counter,
                              # same host clock) and the driver's ack receipt: the gap
                              # t_recv - t_send is transport + the server's in-flight step
                              "t_recv_ms": [None if q["t_recv_server"] is None else
                                            round((q["t_recv_server"] - t_start) * 1e3, 2)
                                            for q in res["pushes"]],
                              # correct messages only (interleaved): when the server finished
                              # the drain + correct step, and the step's own wall time
                              "t_done_ms": [None if q.get("t_done_server") is None else
                                            round((q["t_done_server"] - t_start) * 1e3, 2)
                                            for q in res["pushes"]],
                              "t_step_ms": [q.get("t_step_ms") for q in res["pushes"]],
                              # staged/unified: the engine's per-round walk/correct split
                              "stage_steps": [q.get("stage_steps") for q in res["pushes"]],
                              "t_ack_ms": [None if q["t_ack"] is None else
                                           round((q["t_ack"] - t_start) * 1e3, 2)
                                           for q in res["pushes"]],
                              # when each push actually LEFT (sender thread, after its D2H
                              # copy completed = the GPU finished that band's correction):
                              # the honest start of the next band's window. t_send is only
                              # the CPU issue time -- the sync-free tower runs 20+ ms ahead
                              # of the GPU, so t_pushes_ms is not a GPU timeline.
                              "t_sent_ms": [None if q.get("t_sent") is None else
                                            round((q["t_sent"] - t_start) * 1e3, 2)
                                            for q in res["pushes"]],
                              # first token on the driver clock: the server's open time
                              # (same host clock) + its ttft_from_open. The former
                              # t_open + ttft_from_open started the server-side span at the
                              # CPU issue time of push 0, i.e. before the chunk had left (GPU
                              # backlog + transport, 6 ms one-shot / ~20 ms streaming, was
                              # dropped); that value is kept as ttft_start_issue_ms.
                              "ttft_start_ms": (t_open_ms + t["ttft_from_open_ms"]
                                                if res["pushes"][0]["t_recv_server"] is None else
                                                (res["pushes"][0]["t_recv_server"] - t_start) * 1e3
                                                + t["ttft_from_open_ms"]),
                              "ttft_start_issue_ms": t_open_ms + t["ttft_from_open_ms"],
                              "gen_tokens": len(res["token_ids"]),
                              "finish_reason": res["finish_reason"],
                              "t_client_done_ms": (time.perf_counter() - t_start) * 1e3})
                extra["raw"] = res["text"]   # raw generated text, kept so a scorer fix can re-score rows offline
                record(i, clean_text(args.family, res["text"], args.dataset), gold, size, extra)

            t_iter = None
            for k, i in enumerate(pending):
                t_p = time.perf_counter()
                t_loop_iter = None if t_iter is None else (t_p - t_iter) * 1e3   # whole previous iteration
                t_iter = t_p
                p = prepared(k)
                t_loop_prep = (time.perf_counter() - t_p) * 1e3
                if too_long(i, p):
                    continue
                sink = bridge.sink(f"{tag}-{i}", max_tokens=args.max_tokens)
                t_start = time.perf_counter()
                try:
                    gold, size, extra, _ = vision(p, sink)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if sink.opened and not sink.closed and sink.error is None:
                        bridge.abort(sink.rid)
                    skip(i, "oom")
                    continue
                except BridgeError as e:
                    skip(i, "bridge_error", error=str(e)[:300])
                    continue
                extra["_t_start"] = t_start
                extra["t_loop_prep_ms"] = t_loop_prep
                extra["t_loop_iter_ms"] = t_loop_iter     # of the iteration before this sample's
                inflight.append((i, sink, gold, size, extra))
                while len(inflight) >= args.concurrency:
                    drain_one()
            while inflight:
                drain_one()
        fh.close()
        if args.prefetch > 0:
            pool.shutdown(wait=True)
        el = time.perf_counter() - t_arm0
        if scored:
            print(f"Final Summary: {{\"dataset\": \"{args.dataset}\", \"model\": \"{slug}\", "
                  f"\"arm\": \"{tag}{suffix}\", \"scored\": {scored}, "
                  f"\"acc\": {correct / scored * 100:.4f}, "
                  + (f"\"val\": {val_sum / val_n * 100:.4f}, " if val_n else "") +
                  f"\"elapsed_s\": {el:.1f}, "
                  f"\"samples_per_s\": {scored / el:.3f}, "
                  f"\"skipped\": {json.dumps(skipped)}}}", flush=True)
    print("QWEN_VLLM_ACCURACY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
