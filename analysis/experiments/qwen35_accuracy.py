"""Accuracy for the Qwen3.5-35B streaming arm: floor / streaming(g) / ceiling, one driver.

Every arm decodes through the SAME mechanism -- prefill produces (last-position logits, cache),
then one shared greedy loop steps with explicit positions. This is not pedantry: on Qwen2.5-VL a
baseline that decoded via one continuous `generate()` while the corrected arms decoded via a
two-stage path disagreed with the corrected mechanism on 30-35% of RefCOCO samples with ZERO
correction involved (+2.25pp on 32B) -- the decode mechanism is a real confound and the fix is to
never vary it across arms.

Positions are explicit everywhere. The model silently falls back to cached `rope_deltas` from
whatever ran last on the module when position_ids are omitted in decode -- correct by coincidence,
wrong on reorder (see the axis gate's history).

Scoring is `spec.score()` from the shared registry -- the same normalization every other model's
numbers went through.

Throughput policy (user decision 2026-08-31, after the GQA n=240 equivalence gates):
  * ADOPTED for new campaigns: `--batch 8` on the bounds arms (identical 95-98%, dacc +0.00pp)
    and `--fast-processor` (identical 97-99%, dacc +0.00pp; GPU preprocessing, 10.5x).
  * bs=16 REJECTED (floor dacc -1.67pp, identical 91.7%) -- keep 8 as the ceiling.
  * --group-by-image REJECTED for this model (floor dacc -3.33pp in both padded and length-sorted
    forms; the hybrid linear-attention conv/recurrent state diverges at the prefix boundary).
    Worth retrying only on pure-softmax models.
  * Per-dataset consistency: every arm of one dataset must share one preprocessing path. The
    streaming arm cannot batch but CAN take --fast-processor: GATED 2026-09-01 (gqa contiguous
    n=240: identical 97.1%, dacc +0.42pp) -- ADOPTED, so new campaigns run fastproc on all arms.
  * Rows measured before this date (refcoco/textvqa pyr campaign) are slow-processor bs=1
    throughout -- internally consistent; do not re-run.
"""
import argparse, json, os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from qwen_vl_prefill.datasets_eval import get_spec
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from appcorr.models.qwen35.unified import Qwen35Axis, MODEL_ID_35B
from PIL import Image


def degrade(img: Image.Image, level: int = 2, filt: str = "bicubic") -> Image.Image:
    """The transmission's level-`level` base: 2^level x down, back up. Content degrades, geometry
    does not -- the token grid must match the full-res image or the band mixing is meaningless.

    `filt` selects the DOWNSAMPLING filter: 'bicubic' is what every qwen35 number in the table was
    measured with; 'box' (area average) matches the gemma3/ov2 oracles and approximates the
    canonical cv2.pyrDown pyramid more closely. The 2026-08-28 convention audit flagged the
    divergence; the BOX-vs-BICUBIC sensitivity probe decides whether the table needs re-measuring."""
    # Pyramid-direction cap (both branches of the rule): degrade relative to
    # min(native, what the model samples). Qwen's smart_resize tops out at
    # max_pixels=12.8M, so every bench measured so far sat below it and the two
    # branches coincided -- MME-RealWorld (36M px) is where this first BINDS.
    QWEN_MAX_PX = 16_777_216  # measured 2026-08-31: processor longest_edge cap; images stay native below it
    f = 2 ** level
    w, h = img.size
    s = min(1.0, (QWEN_MAX_PX / (w * h)) ** 0.5)
    if s < 1.0:
        w2, h2 = max(1, int(w * s)), max(1, int(h * s))
    else:
        w2, h2 = w, h
    if filt == "pyr":
        # The protocol archetype itself: cv2.pyrDown chain, cv2.pyrUp back with
        # per-step dstsize (odd dims round up on pyrDown; the stored size chain
        # restores them exactly, mirroring laplacian.py's _iterative_upsample_native).
        # The cap branch fits to the sampled resolution FIRST, then walks the pyramid.
        import cv2
        import numpy as np
        arr = np.asarray(img if s == 1.0 else img.resize((w2, h2), Image.BILINEAR))
        sizes = [(arr.shape[1], arr.shape[0])]
        for _ in range(level):
            arr = cv2.pyrDown(arr)
            sizes.append((arr.shape[1], arr.shape[0]))
        for i in range(level - 1, -1, -1):
            arr = cv2.pyrUp(arr, dstsize=sizes[i])
        out = Image.fromarray(arr)
        return out if s == 1.0 else out.resize((w, h), Image.BICUBIC)
    down = Image.BOX if filt == "box" else Image.BICUBIC
    return img.resize((max(1, w2 // f), max(1, h2 // f)), down).resize((w, h), Image.BICUBIC)


@torch.no_grad()
def greedy(axis, logits, cache, start_pos, n=24):
    """The one decode mechanism, shared by every arm."""
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
    return axis.processor.tokenizer.decode(toks, skip_special_tokens=True)


def _left_pad(seqs, pad_val):
    """Stack (1, L_i) rows into (B, Lmax), left-padded with `pad_val`."""
    L = max(s.shape[-1] for s in seqs)
    out = torch.full((len(seqs), L), pad_val, dtype=seqs[0].dtype)
    for b, s in enumerate(seqs):
        out[b, L - s.shape[-1]:] = s[0]
    return out


@torch.no_grad()
def prefill_stock_batched(axis, per_sample):
    """Batched stock prefill, BOUNDS arms only (streaming decode rides the fork K/V and cannot
    batch). Preprocessing stays the per-sample `build_inputs` the bs=1 path makes; this only
    collates: left padding, per-row M-RoPE positions computed at bs=1 and shifted into the padded
    frame, attention mask over the pad. Each row therefore sees exactly the tensors its bs=1 run
    would -- the only degree of freedom left is batched bf16 reduction order, which the
    equivalence gate (bs=1 vs bs=N on the same subset) measures empirically."""
    dev = "cuda:0"
    ids_l, mm_l, pos_l, dps = [], [], [], []
    for inp in per_sample:
        pos3, _ = axis.model.model.get_rope_index(inp["input_ids"], inp["mm_token_type_ids"],
                                                  image_grid_thw=inp["image_grid_thw"])
        ids_l.append(inp["input_ids"])
        mm_l.append(inp["mm_token_type_ids"])
        pos_l.append(pos3)
        dps.append(int(pos3.max().item()) + 1)
    tok = axis.processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = _left_pad(ids_l, pad_id).to(dev)
    mm = _left_pad(mm_l, 0).to(dev)
    B, L = ids.shape
    attn = torch.zeros(B, L, dtype=torch.long, device=dev)
    pos = torch.ones(3, B, L, dtype=torch.long, device=dev)
    for b, p in enumerate(pos_l):
        n = p.shape[-1]
        attn[b, L - n:] = 1
        pos[:, b, L - n:] = p[:, 0].to(dev)
    px = torch.cat([inp["pixel_values"] for inp in per_sample]).to(dev)
    thw = torch.cat([inp["image_grid_thw"] for inp in per_sample]).to(dev)
    out = axis.model(input_ids=ids, attention_mask=attn, pixel_values=px.to(axis.model.dtype),
                     image_grid_thw=thw, mm_token_type_ids=mm, position_ids=pos, use_cache=True)
    return out.logits[:, -1], out.past_key_values, dps, attn


@torch.no_grad()
def greedy_batched(axis, logits, cache, start_pos, attn, n=24):
    """`greedy`, vectorised over rows: same argmax loop, per-row positions, per-row EOS. A
    finished row keeps stepping (its trailing tokens are discarded) so the cache stays rectangular."""
    tok = axis.processor.tokenizer
    eos = tok.eos_token_id
    B = logits.shape[0]
    cur = logits.argmax(-1, keepdim=True)
    pos = torch.tensor(start_pos, device=cur.device, dtype=torch.long)
    toks = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=cur.device)
    for _ in range(n):
        done |= cur[:, 0] == eos
        for b in range(B):
            if not done[b]:
                toks[b].append(int(cur[b, 0]))
        if bool(done.all()):
            break
        attn = torch.cat([attn, torch.ones(B, 1, dtype=attn.dtype, device=attn.device)], 1)
        pid = pos.view(1, B, 1).expand(3, B, 1).contiguous()
        out = axis.model(input_ids=cur, attention_mask=attn, past_key_values=cache,
                         position_ids=pid, use_cache=True)
        cache = out.past_key_values
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        pos = pos + 1
    return [tok.decode(t, skip_special_tokens=True) for t in toks]


# Datasets whose rows share images (GQA: ~31.6 questions/image over 398 images; VisDrone: one row
# per (image, category)). The value is the row field identifying the image.
IMAGE_GROUP_KEYS = {"gqa": "imageId", "visdrone_count": "path", "visdrone_det": "path"}


def _common_prefix_len(per_sample):
    """Longest shared leading run of input_ids across the group (token-exact, no assumptions
    about the chat template's layout)."""
    ref = per_sample[0]["input_ids"][0]
    P = ref.shape[0]
    for inp in per_sample[1:]:
        o = inp["input_ids"][0]
        n = min(P, o.shape[0])
        neq = (ref[:n] != o[:n]).nonzero()
        P = min(P, int(neq[0]) if len(neq) else n)
    return P


def _expand_cache(prefix_cache, B):
    """Batch-expand a bs=1 prefix cache for the suffix forward, layer-type-aware (Qwen3.5 is a
    HYBRID: standard attention layers hold appended KV, linear-attention layers hold conv +
    recurrent state). Standard layers get expanded VIEWS -- their update() rebinds via cat, so
    the shared prefix is read, never written (verified on DynamicCache). Linear layers get
    CLONES -- their update_*() writes in place via copy_(), so a view would corrupt the prefix
    for every later chunk."""
    import copy as _copy
    out = _copy.copy(prefix_cache)
    out.layers = []
    for layer in prefix_cache.layers:
        l2 = _copy.copy(layer)
        if hasattr(layer, "keys"):
            l2.keys = layer.keys.expand(B, *layer.keys.shape[1:])
            l2.values = layer.values.expand(B, *layer.values.shape[1:])
        else:
            l2.conv_states = layer.conv_states.expand(B, *layer.conv_states.shape[1:]).clone()
            l2.recurrent_states = layer.recurrent_states.expand(
                B, *layer.recurrent_states.shape[1:]).clone()
            l2.batch_size = B
        out.layers.append(l2)
    return out


@torch.no_grad()
def prefill_prefix(axis, inp, P):
    """bs=1 forward over the shared [template + image tokens] prefix; returns its KV cache."""
    dev = "cuda:0"
    ids = inp["input_ids"]
    mm = inp["mm_token_type_ids"]
    pos3, _ = axis.model.model.get_rope_index(ids, mm, image_grid_thw=inp["image_grid_thw"])
    out = axis.model(input_ids=ids[:, :P].to(dev),
                     pixel_values=inp["pixel_values"].to(dev).to(axis.model.dtype),
                     image_grid_thw=inp["image_grid_thw"].to(dev),
                     mm_token_type_ids=mm[:, :P].to(dev),
                     position_ids=pos3[:, :, :P].to(dev), use_cache=True)
    return out.past_key_values


@torch.no_grad()
def prefill_suffix_batched(axis, per_sample, P, prefix_cache):
    """Batched forward of the per-question suffixes on top of one shared image prefix. Left pad
    between prefix and suffix, masked out; per-row M-RoPE positions from each row's own full
    get_rope_index, sliced past P -- identical values to what a full prefill would assign."""
    dev = "cuda:0"
    tok = axis.processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids_l, pos_l, dps = [], [], []
    for inp in per_sample:
        pos3, _ = axis.model.model.get_rope_index(inp["input_ids"], inp["mm_token_type_ids"],
                                                  image_grid_thw=inp["image_grid_thw"])
        ids_l.append(inp["input_ids"][:, P:])
        pos_l.append(pos3[:, :, P:])
        dps.append(int(pos3.max().item()) + 1)
    ids = _left_pad(ids_l, pad_id).to(dev)
    B, Ls = ids.shape
    attn = torch.zeros(B, P + Ls, dtype=torch.long, device=dev)
    attn[:, :P] = 1
    pos = torch.ones(3, B, Ls, dtype=torch.long, device=dev)
    for b, p in enumerate(pos_l):
        n = p.shape[-1]
        attn[b, P + Ls - n:] = 1
        pos[:, b, Ls - n:] = p[:, 0].to(dev)
    cache = _expand_cache(prefix_cache, B)
    out = axis.model(input_ids=ids, attention_mask=attn, past_key_values=cache,
                     position_ids=pos, use_cache=True)
    return out.logits[:, -1], out.past_key_values, dps, attn


@torch.no_grad()
def prefill_stock(axis, inputs):
    """Stock prefill (floor and ceiling), returning the same (logits, cache, decode_pos) the
    streaming arm returns, so `greedy` cannot tell the arms apart."""
    ids = inputs["input_ids"]
    mm = inputs["mm_token_type_ids"]
    pos3, _ = axis.model.model.get_rope_index(ids, mm, image_grid_thw=inputs["image_grid_thw"])
    out = axis.model(input_ids=ids, pixel_values=inputs["pixel_values"].to(axis.model.dtype),
                     image_grid_thw=inputs["image_grid_thw"], mm_token_type_ids=mm,
                     position_ids=pos3, use_cache=True)
    return out.logits[:, -1], out.past_key_values, int(pos3.max().item()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default=MODEL_ID_35B,
                    help="checkpoint id; e.g. Qwen/Qwen3.5-122B-A10B-FP8")
    ap.add_argument("--level", type=int, default=2, help="pyramid level of the degraded base")
    # Default flipped to box 2026-08-28 after the paired probe (4/50 flips, 3:1 toward box):
    # box matches the convention's reference (area average ~ pyramid level). Every degraded-arm
    # number measured before this date used bicubic and lives in analysis/results/qwen35_accuracy/;
    # box re-measurements go to qwen35_accuracy_box/ -- NEVER append across the boundary, the
    # jsonl resume would silently mix filters.
    ap.add_argument("--degrade-filter", choices=["bicubic", "box", "pyr"], default="box")
    ap.add_argument("--arms", nargs="+", default=["floor", "streaming", "ceiling"])
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keep", type=float, default=1.0,
                    help="fraction of image tokens corrected (streaming arm); 1.0 = correct all")
    ap.add_argument("--samples", type=int, default=0, help="0 = full split")
    ap.add_argument("--batch", type=int, default=1,
                    help="batch size for the BOUNDS arms (floor/ceiling) only; streaming stays "
                         "bs=1 (its decode rides the fork K/V). Gate bs=N against bs=1 on the "
                         "same subset before trusting a batched campaign.")
    ap.add_argument("--group-by-image", action="store_true",
                    help="bounds only, image-shared datasets (gqa/visdrone): prefill each image's "
                         "shared prefix once and batch the question suffixes on it. Gate against "
                         "the plain path before a campaign.")
    ap.add_argument("--contiguous", action="store_true",
                    help="with --samples N, take the FIRST N rows instead of a stride -- needed "
                         "for gating --group-by-image (a strided subset breaks every image group)")
    ap.add_argument("--fast-processor", action="store_true",
                    help="load the torchvision-based fast image processor (use_fast=True). "
                         "FASTER BUT NOT PIXEL-IDENTICAL to the slow PIL path -- never mix with "
                         "slow-processor results in one table column; gate before adopting.")
    ap.add_argument("--out", default="analysis/results/qwen35_accuracy")
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model, use_fast=bool(args.fast_processor))
    if args.fast_processor:
        # The fast (torchvision) image processor only pays off on GPU: measured 14 ms/build on
        # cuda vs 148 ms PIL / 242 ms fast-on-cpu for a 1400x1800 image. Route every
        # apply_chat_template through the device without touching the shared Axis code.
        _orig_act = proc.apply_chat_template

        def _act(*a, **kw):
            kw.setdefault("device", "cuda:0")
            return _orig_act(*a, **kw)

        proc.apply_chat_template = _act
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="cuda:0").eval()
    axis = Qwen35Axis(model, proc)
    spec = get_spec(args.dataset)
    ds = spec.load(load_dataset)
    n = len(ds) if args.samples == 0 else min(args.samples, len(ds))
    idxs = list(range(n)) if args.samples == 0 else \
        (list(range(n)) if args.contiguous else list(range(0, len(ds), max(1, len(ds) // n)))[:n])
    os.makedirs(args.out, exist_ok=True)

    for arm in args.arms:
        slug = "" if args.model == MODEL_ID_35B else "_" + args.model.split("/")[-1].lower()
        arm_suffix = ""
        if arm == "streaming":
            arm_suffix = f"_g{args.groups}"
            if args.keep < 1.0:
                arm_suffix += f"_k{args.keep:.2f}"
        path = os.path.join(args.out, f"{args.dataset}{slug}_{arm}{arm_suffix}.jsonl")
        done = set()
        if os.path.exists(path):
            with open(path) as f:
                done = {json.loads(l)["i"] for l in f if l.strip()}
        correct, scored = 0, 0
        f = open(path, "a")

        def record(i, pred, gold, size):
            """Score + append one row -- the same transform/score/write the bs=1 loop performs."""
            nonlocal correct, scored
            w_, h_ = size
            if args.dataset in ("refcoco", "visdrone_det"):
                import re as _re
                nums = _re.findall(r"-?\d+\.?\d*", pred)[:4]
                if len(nums) == 4:
                    x1, y1, x2, y2 = (float(v) for v in nums)
                    pred = (f"{x1 * w_ / 1000:.1f},{y1 * h_ / 1000:.1f},"
                            f"{x2 * w_ / 1000:.1f},{y2 * h_ / 1000:.1f}")
            try:
                ok, val = spec.score(pred, gold)
            except NotImplementedError:
                ok, val = 0, None
            correct += ok
            scored += 1
            done.add(i)
            f.write(json.dumps({"i": int(i), "pred": pred, "gold": gold, "ok": int(ok),
                                "val": (float(val) if val is not None else None)}) + "\n")
            if scored % 50 == 0:
                f.flush()
                print(f"[{arm}] {scored} scored, running {correct / scored * 100:.2f}%",
                      flush=True)

        # Image-grouped prefix reuse (--group-by-image, bounds only): the shared [template +
        # image tokens] prefix prefills ONCE per image, and the question suffixes ride it in
        # batches on an expanded (read-only) cache. GQA amortises the image work ~31x. Any group
        # that fails a structural check falls through untouched to the exact paths below.
        if (args.group_by_image and arm in ("floor", "ceiling")
                and args.dataset in IMAGE_GROUP_KEYS):
            from concurrent.futures import ThreadPoolExecutor as _TPE
            keyf = IMAGE_GROUP_KEYS[args.dataset]
            groups = {}
            for i in idxs:
                if i not in done:
                    groups.setdefault(str(ds[int(i)][keyf]), []).append(i)
            gb_pool = _TPE(max_workers=8)

            def _gb_build(i):
                img, q, gold = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                use = img if arm == "ceiling" else degrade(img, args.level, args.degrade_filter)
                return axis.build_inputs(use, q), gold, img.size

            for rows in groups.values():
                if len(rows) < 2:
                    continue  # nothing to share; the batch / bs=1 paths below take it
                built = list(gb_pool.map(_gb_build, rows))
                per = [b[0] for b in built]
                P = _common_prefix_len(per)
                last_img = int(per[0]["mm_token_type_ids"][0].nonzero().max())
                if (P <= last_img
                        or any(not torch.equal(inp["image_grid_thw"], per[0]["image_grid_thw"])
                               for inp in per[1:])
                        or any(bool(inp["mm_token_type_ids"][0, P:].any()) for inp in per)):
                    continue
                # Chunk by ascending suffix length: padding feeds the linear-attention layers'
                # recurrent state (it cannot be masked out the way softmax attention masks it),
                # and in this path the pads sit right between the prefix state and the question.
                # The unsorted gate measured the damage at -3.33pp on the floor arm; sorting
                # makes most chunks pad-free since GQA questions are near-uniform length.
                order = sorted(range(len(rows)), key=lambda j: per[j]["input_ids"].shape[-1])
                rows = [rows[j] for j in order]
                built = [built[j] for j in order]
                per = [per[j] for j in order]
                try:
                    pc = prefill_prefix(axis, per[0], P)
                    bs = max(1, args.batch)
                    for c0 in range(0, len(rows), bs):
                        sl = slice(c0, c0 + bs)
                        lg, kv, dps, attn = prefill_suffix_batched(axis, per[sl], P, pc)
                        preds = greedy_batched(axis, lg, kv, dps, attn)
                        for i, pred, b in zip(rows[sl], preds, built[sl]):
                            record(i, pred, b[1], b[2])
                    del pc
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue
            gb_pool.shutdown(wait=False)

        # Bounds batching (--batch N): chunks of per-sample-built inputs through the batched
        # prefill+greedy. Every row scored here lands in `done`, so the per-sample loop below only
        # picks up what batching skipped -- an OOM'd chunk simply falls back to bs=1 there.
        if args.batch > 1 and arm in ("floor", "ceiling"):
            pending = [i for i in idxs if i not in done]

            from concurrent.futures import ThreadPoolExecutor
            build_pool = ThreadPoolExecutor(max_workers=max(1, min(8, args.batch)))

            def _build_one(i):
                img, q, gold = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                use = img if arm == "ceiling" else degrade(img, args.level, args.degrade_filter)
                return axis.build_inputs(use, q), gold, img.size

            def _build_chunk(chunk):
                """CPU side of a chunk, samples built in PARALLEL (same calls as bs=1; fast
                tokenizers and PIL/numpy are thread-safe for this and release the GIL)."""
                out = list(build_pool.map(_build_one, chunk))
                return [o[0] for o in out], [o[1] for o in out], [o[2] for o in out]

            def _forward_chunk(per):
                """Batched prefill+greedy with OOM backoff: halve rather than give the chunk up,
                so a huge-image chunk degrades to smaller batches instead of to bs=1."""
                try:
                    lg, kv, dps, attn = prefill_stock_batched(axis, per)
                    return greedy_batched(axis, lg, kv, dps, attn)
                except torch.cuda.OutOfMemoryError:
                    if len(per) == 1:
                        raise
                    torch.cuda.empty_cache()
                    h = len(per) // 2
                    return _forward_chunk(per[:h]) + _forward_chunk(per[h:])

            # One-chunk-ahead prefetch: chunk c+1 builds while the main thread waits on the GPU
            # for chunk c. Numerics untouched -- the same calls run, just earlier.
            pre_pool = ThreadPoolExecutor(max_workers=1)
            chunks = [pending[c0:c0 + args.batch] for c0 in range(0, len(pending), args.batch)]
            fut = pre_pool.submit(_build_chunk, chunks[0]) if chunks else None
            for ci, chunk in enumerate(chunks):
                per, golds, sizes = fut.result()
                fut = pre_pool.submit(_build_chunk, chunks[ci + 1]) if ci + 1 < len(chunks) else None
                try:
                    preds = _forward_chunk(per)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue
                for i, pred, gold, (w_, h_) in zip(chunk, preds, golds, sizes):
                    if args.dataset in ("refcoco", "visdrone_det"):
                        # Same 0-1000 -> pixel rescale as the bs=1 path below.
                        import re as _re
                        nums = _re.findall(r"-?\d+\.?\d*", pred)[:4]
                        if len(nums) == 4:
                            x1, y1, x2, y2 = (float(v) for v in nums)
                            pred = (f"{x1 * w_ / 1000:.1f},{y1 * h_ / 1000:.1f},"
                                    f"{x2 * w_ / 1000:.1f},{y2 * h_ / 1000:.1f}")
                    try:
                        ok, val = spec.score(pred, gold)
                    except NotImplementedError:
                        ok, val = 0, None
                    correct += ok
                    scored += 1
                    done.add(i)
                    f.write(json.dumps({"i": int(i), "pred": pred, "gold": gold, "ok": int(ok),
                                        "val": (float(val) if val is not None else None)}) + "\n")
                    if scored % 50 == 0:
                        f.flush()
                        print(f"[{arm}] {scored} scored, running {correct / scored * 100:.2f}%",
                              flush=True)
            pre_pool.shutdown(wait=False)
            build_pool.shutdown(wait=False)
        for i in idxs:
            if i in done:
                continue
            img, q, gold = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
            if img.mode != "RGB":
                img = img.convert("RGB")
            try:
                # Full-image inputs only where an arm consumes them -- the floor arm never does,
                # and the redundant build was measured at 375 ms/sample on textvqa (2026-08-31).
                if arm == "ceiling":
                    inputs = axis.build_inputs(img, q).to("cuda:0")
                    lg, kv, dp = prefill_stock(axis, inputs)
                elif arm == "floor":
                    base_inputs = axis.build_inputs(degrade(img, args.level, args.degrade_filter), q).to("cuda:0")
                    lg, kv, dp = prefill_stock(axis, base_inputs)
                else:
                    inputs = axis.build_inputs(img, q).to("cuda:0")
                    base_px = axis.build_inputs(degrade(img, args.level, args.degrade_filter), q)["pixel_values"].to("cuda:0")
                    lg, kv, st = axis.streaming_forward(inputs, base_px, args.groups,
                                                        keep=args.keep)
                    dp = st["decode_start_pos"]
                pred = greedy(axis, lg, kv, dp)
                if args.dataset in ("refcoco", "visdrone_det"):
                    # Qwen3-generation grounding emits 0-1000 RELATIVE coords for refcoco AND
                    # visdrone_det (same convention; VisDrone gold is native pixels). (probe
                    # 2026-08-28: boxes cap at 1000 regardless of the pixel-coord
                    # instruction). Deterministic rescale to this image's pixels.
                    import re as _re
                    nums = _re.findall(r"-?\d+\.?\d*", pred)[:4]
                    if len(nums) == 4:
                        w_, h_ = img.size
                        x1, y1, x2, y2 = (float(v) for v in nums)
                        pred = (f"{x1 * w_ / 1000:.1f},{y1 * h_ / 1000:.1f},"
                                f"{x2 * w_ / 1000:.1f},{y2 * h_ / 1000:.1f}")
                try:
                    ok, val = spec.score(pred, gold)
                except NotImplementedError:  # wildvision: judge-only prediction dump
                    ok, val = 0, None
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                f.write(json.dumps({"i": int(i), "skip": "oom"}) + "\n")
                f.flush()
                continue
            correct += ok
            scored += 1
            f.write(json.dumps({"i": int(i), "pred": pred, "gold": gold, "ok": int(ok),
                                "val": (float(val) if val is not None else None)}) + "\n")
            if scored % 50 == 0:
                f.flush()
                print(f"[{arm}] {scored} scored, running {correct / scored * 100:.2f}%", flush=True)
        f.close()
        if scored:
            print(f"Final Summary: {{\"dataset\": \"{args.dataset}\", \"arm\": \"{arm}\", "
                  f"\"scored\": {scored}, \"acc\": {correct / scored * 100:.4f}}}", flush=True)
    print("QWEN35_ACCURACY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
