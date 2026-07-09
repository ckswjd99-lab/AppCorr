"""
refcoco_offload_eval.py

RefCOCO (val split) referring-expression-comprehension driver for the Qwen2.5-VL AppCorr pipeline --
structurally identical to realworldqa_offload_eval.py (same offload pipeline mechanics, native/
dynamic per-image resolution, --grouping-strategy/--num-groups/--keep-rate CLI overrides), but a
grounding (text -> bounding box) task instead of VQA.

`lmms-lab/RefCOCO` ships each (image, bbox) pair with a *reverse*-direction prompt template ("write
a caption for the circled region") plus a list of human `answer` captions for that region -- the
standard REC benchmark task is the other direction (given a referring expression, localize the
region), so this driver uses `answer[0]` as the input referring expression and predicts+scores a
bounding box against the dataset's own `bbox` (COCO [x,y,w,h] format, original-image pixel space).
`bbox` is rescaled to the smart_resize'd image's pixel space before scoring (since that's the pixel
space the model actually sees and is asked to output coordinates in). Metric: Acc@0.5 (IoU >= 0.5).

Run (appcorr env):
    python analysis/experiments/refcoco_offload_eval.py \\
        --config offload/config/realworldqa_qwen25vl_32b_interleaved_g4.json --num-samples 20
"""

import argparse
import json
import multiprocessing
import queue as queue_mod
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GROUNDING_PROMPT_TMPL = (
    "Locate the region described by: \"{expr}\". "
    "Output ONLY the bounding box as four numbers x1,y1,x2,y2 (top-left and bottom-right pixel "
    "coordinates in this image), with no other text."
)

NUM_RE = re.compile(r"(?<![a-zA-Z0-9])-?\d+(?:\.\d+)?")


def parse_bbox(text: str):
    nums = NUM_RE.findall(text)
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = (float(n) for n in nums[:4])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def score_answer(pred_text: str, gt_box_xyxy) -> tuple:
    """Returns (correct: bool, iou: float)."""
    pred_box = parse_bbox(pred_text)
    if pred_box is None:
        return False, 0.0
    iou = iou_xyxy(pred_box, gt_box_xyxy)
    return iou >= 0.5, iou


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--grouping-strategy", type=str, default=None)
    p.add_argument("--num-groups", type=int, default=None)
    p.add_argument("--keep-rate", type=float, default=None)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--full", action="store_true", help="Run all RefCOCO val examples (ignores --num-samples).")
    p.add_argument("--result-timeout", type=float, default=600.0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--label", type=str, default=None)
    return p.parse_args()


def load_base_config_dict(args):
    with open(args.config, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["batch_size"] = 1
    raw["device"] = args.device
    if args.grouping_strategy is not None:
        raw.setdefault("transmission_kwargs", {})["grouping_strategy"] = args.grouping_strategy
    if args.num_groups is not None:
        raw.setdefault("transmission_kwargs", {})["num_groups"] = args.num_groups
    if args.keep_rate is not None:
        raw.setdefault("transmission_kwargs", {})["keep_rate"] = args.keep_rate
    return raw


def main():
    args = parse_args()
    raw_config = load_base_config_dict(args)
    label = args.label or Path(args.config).stem
    model_path = raw_config["dataset_kwargs"]["model_path"]

    from offload.common import ExperimentConfig
    from offload.policies import get_transmission
    from offload.server.scheduler import SchedulerModule
    from offload.server.worker import WorkerModule
    from transformers import AutoProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset
    from PIL import Image

    print(f"[driver] === Run: {label} ===")
    print(f"[driver] config={args.config}  transmission_kwargs={raw_config.get('transmission_kwargs', {})}")

    print(f"[driver] loading processor for {model_path} (cheap -- tokenizer/image_processor config only)...")
    processor = AutoProcessor.from_pretrained(model_path)
    ip = processor.image_processor
    min_pixels = ip.size["shortest_edge"]
    max_pixels = ip.size["longest_edge"]
    base_factor = ip.patch_size * ip.merge_size
    factor = base_factor * 4
    print(f"[driver] smart_resize params: factor={factor} (base={base_factor}, aligned for pyramid scale 4) "
          f"min_pixels={min_pixels} max_pixels={max_pixels}")

    print("[driver] loading lmms-lab/RefCOCO (val split) ...")
    ds = load_dataset("lmms-lab/RefCOCO", split="val")
    n_total = len(ds)
    if args.full:
        indices = list(range(n_total))
        print(f"[driver] running FULL val split: {n_total} examples")
    else:
        n_samples = min(args.num_samples, n_total)
        stride = max(n_total // n_samples, 1)
        indices = list(range(0, n_total, stride))[:n_samples]
        print(f"[driver] sampling {len(indices)} examples strided across {n_total} (stride={stride})")

    sched_q = multiprocessing.Queue()
    worker_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    control_q = multiprocessing.Queue()
    feedback_q = multiprocessing.Queue()
    scheduler = SchedulerModule(sched_q, worker_q, control_q, feedback_q)
    worker = WorkerModule(worker_q, result_q, feedback_q)
    scheduler.start()
    worker.start()

    encoder = get_transmission(raw_config["transmission_policy_name"])
    op_times = defaultdict(list)
    per_sample = []
    t_start = time.time()
    correct = 0
    processed = 0
    iou_sum = 0.0
    print_every = 1 if len(indices) <= 20 else max(len(indices) // 100, 5)

    try:
        for idx in indices:
            ex = ds[idx]
            image = ex["image"].convert("RGB")
            expr = ex["answer"][0] if isinstance(ex["answer"], list) and ex["answer"] else str(ex["answer"])
            bx, by, bw, bh = ex["bbox"]
            gt_box_orig = (bx, by, bx + bw, by + bh)  # xyxy, original image pixel space
            prompt = GROUNDING_PROMPT_TMPL.format(expr=expr)

            orig_w, orig_h = image.width, image.height
            target_h, target_w = smart_resize(orig_h, orig_w, factor=factor,
                                               min_pixels=min_pixels, max_pixels=max_pixels)
            resized = image.resize((target_w, target_h), Image.BILINEAR)
            image_np = np.array(resized, dtype=np.uint8)  # [H,W,3]

            sx, sy = target_w / orig_w, target_h / orig_h
            gt_box_resized = (gt_box_orig[0] * sx, gt_box_orig[1] * sy, gt_box_orig[2] * sx, gt_box_orig[3] * sy)

            image_config = dict(raw_config)
            image_config["image_shape"] = [target_h, target_w, 3]
            config = ExperimentConfig(**image_config)

            control_q.put(("CONFIG", config))
            time.sleep(2.0)  # let CONFIG propagate before this image's patches arrive

            t0 = time.time()
            for group_patches in encoder.encode(image_np[None], config):
                now = time.time()
                for p in group_patches:
                    p.arrival_time = now
                    p.text_payload = prompt
                for p in group_patches:
                    sched_q.put(p)
            try:
                result = result_q.get(timeout=args.result_timeout)
            except queue_mod.Empty:
                print(f"    sample {processed}: TIMEOUT waiting for InferenceResult -- aborting")
                raise
            wall = time.time() - t0

            pred_text = result.output[0] if result.output else ""
            for ev in result.server_events:
                op_times[ev["type"]].append(ev["end"] - ev["start"])

            ok, iou = score_answer(pred_text, gt_box_resized)
            correct += int(ok)
            iou_sum += iou
            processed += 1
            per_sample.append({"idx": idx, "expr": expr, "gt_box": gt_box_resized, "pred": pred_text,
                                "iou": iou, "correct": ok, "wall": wall, "grid_hw": (target_h, target_w)})
            if processed % print_every == 0 or processed == len(indices):
                acc = 100.0 * correct / processed
                mean_iou = iou_sum / processed
                print(f"    [{processed}/{len(indices)}] idx={idx} grid={target_h}x{target_w} "
                      f"expr={expr[:40]!r} pred={pred_text[:60]!r} iou={iou:.2f} correct={ok} "
                      f"running_acc@0.5={acc:.2f}% mean_iou={mean_iou:.3f} "
                      f"wall={wall:.2f}s elapsed={time.time()-t_start:.0f}s")
                sys.stdout.flush()
    finally:
        control_q.put(("STOP", None))
        result_q.cancel_join_thread()
        for proc in (scheduler, worker):
            proc.join(timeout=15)
            if proc.is_alive():
                proc.terminate()

    total_wall = time.time() - t_start
    acc = 100.0 * correct / max(processed, 1)
    mean_iou = iou_sum / max(processed, 1)

    print(f"\n[driver] === Summary: {label} ===")
    print(f"    samples: {processed}")
    print(f"    accuracy@0.5: {acc:.2f}%  ({correct}/{processed})")
    print(f"    mean IoU: {mean_iou:.4f}")
    print(f"    total wall time: {total_wall:.1f}s ({total_wall / max(processed, 1):.2f}s/sample avg)")

    print(f"\n[driver] === Mean per-op GPU time (ms), {label} ===")
    for op_name in sorted(op_times):
        vals = op_times[op_name]
        print(f"    {op_name:20s} mean={np.mean(vals) * 1000:8.3f}ms  n={len(vals)}")

    return {
        "label": label,
        "config": args.config,
        "accuracy": acc,
        "mean_iou": mean_iou,
        "op_times_ms": {k: float(np.mean(v) * 1000) for k, v in op_times.items()},
        "total_wall_sec": total_wall,
        "per_sample": per_sample,
    }


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
