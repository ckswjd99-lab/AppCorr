"""
clip_coco_retrieval_offload_eval.py

Drives the REAL offload pipeline (SchedulerModule + WorkerModule, no TCP -- same convention as
clip_zeroshot_offload_eval.py) against MS-COCO Captions val2017 images, using the CLIP-ViT-bigG/14
retrieval executor (OpenCLIPExecutor with dataset_kwargs.clip_task="retrieval"). Computes
image-to-text and text-to-image Recall@{1,5,10}.

Unlike ImageNet top1/top5 (batch-local, additive), retrieval recall@k is GLOBAL: it requires the
full image-embeds x caption-embeds similarity matrix. So this driver:
  1. Samples N images (deterministic stride, same as the zero-shot driver) from val2017 whose
     image_ids have captions in captions_val2017.json.
  2. Precomputes ALL captions belonging to those N sampled images (not the full 25014 -- recall@k
     is computed within the sampled set, so a small --num-images run stays fast) via the text
     tower, full precision, no approx/correct (captions aren't progressively streamed).
  3. Runs the N images through the SAME offload pipeline used for zero-shot (approx-only / full
     baseline / interleaved correction, selected by --config) to get corrected image embeddings.
  4. Computes the [N_images, N_captions] cosine similarity matrix and recall@{1,5,10} both
     directions.

Run (appcorr env):
    python analysis/experiments/clip_coco_retrieval_offload_eval.py \\
        --config offload/config/coco_retrieval_clip_bigg_interleaved_g4.json --num-images 10
"""

import argparse
import json
import multiprocessing
import queue as queue_mod
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

COCO_CAPTIONS_JSON = "/home/nxclab/fiftyone/coco-2017/raw/captions_val2017.json"
COCO_IMAGES_ROOT = "/home/nxclab/fiftyone/coco-2017/validation/data"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--grouping-strategy", type=str, default=None)
    parser.add_argument("--num-groups", type=int, default=None)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--result-timeout", type=float, default=300.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--label", type=str, default=None)
    return parser.parse_args()


def load_config(args):
    from offload.common import ExperimentConfig

    with open(args.config, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["batch_size"] = 1
    raw["device"] = args.device
    if args.grouping_strategy is not None:
        raw.setdefault("transmission_kwargs", {})["grouping_strategy"] = args.grouping_strategy
    if args.num_groups is not None:
        raw.setdefault("transmission_kwargs", {})["num_groups"] = args.num_groups
    return ExperimentConfig(**raw), raw


def load_coco_annotations():
    with open(COCO_CAPTIONS_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_captions = defaultdict(list)
    for ann in coco["annotations"]:
        id_to_captions[ann["image_id"]].append(ann["caption"])
    return id_to_file, id_to_captions


def sample_image_ids(id_to_file, id_to_captions, num_images):
    all_ids = sorted([iid for iid in id_to_file if id_to_captions.get(iid)])
    n_total = len(all_ids)
    n_samples = min(num_images, n_total)
    stride = max(n_total // n_samples, 1)
    return [all_ids[i] for i in range(0, n_total, stride)][:n_samples]


def compute_recall_at_k(sim: np.ndarray, image_ids: list, caption_owner: list, ks=(1, 5, 10)):
    """
    sim: [N_images, N_captions] cosine similarity.
    image_ids: [N_images] -- image id for row i.
    caption_owner: [N_captions] -- image id that caption j belongs to.
    """
    n_img, n_cap = sim.shape
    caption_owner = np.array(caption_owner)

    # i2t: for each image, rank captions, check if ANY ground-truth caption is in top-k.
    i2t_ranks = np.argsort(-sim, axis=1)  # [N_images, N_captions], descending
    i2t_recall = {k: 0 for k in ks}
    for i in range(n_img):
        gt_mask = caption_owner == image_ids[i]
        for k in ks:
            topk_caps = i2t_ranks[i, :k]
            if gt_mask[topk_caps].any():
                i2t_recall[k] += 1
    i2t_recall = {k: 100.0 * v / n_img for k, v in i2t_recall.items()}

    # t2i: for each caption, rank images, check if its ground-truth image is in top-k.
    t2i_ranks = np.argsort(-sim.T, axis=1)  # [N_captions, N_images], descending
    image_ids_arr = np.array(image_ids)
    t2i_recall = {k: 0 for k in ks}
    for j in range(n_cap):
        gt_image_id = caption_owner[j]
        for k in ks:
            topk_imgs = t2i_ranks[j, :k]
            if (image_ids_arr[topk_imgs] == gt_image_id).any():
                t2i_recall[k] += 1
    t2i_recall = {k: 100.0 * v / n_cap for k, v in t2i_recall.items()}

    return i2t_recall, t2i_recall


def main():
    args = parse_args()
    config, raw_config = load_config(args)
    label = args.label or Path(args.config).stem

    from offload.policies import get_transmission
    from offload.server.scheduler import SchedulerModule
    from offload.server.worker import WorkerModule

    print(f"[driver] === Run: {label} ===")
    print(f"[driver] config={args.config}  transmission_kwargs={raw_config.get('transmission_kwargs', {})}")

    id_to_file, id_to_captions = load_coco_annotations()
    sampled_ids = sample_image_ids(id_to_file, id_to_captions, args.num_images)
    print(f"[driver] sampling {len(sampled_ids)} COCO val2017 images "
          f"(of {len(id_to_file)} total with captions)")

    captions_flat = []
    caption_owner = []
    for iid in sampled_ids:
        for cap in id_to_captions[iid]:
            captions_flat.append(cap)
            caption_owner.append(iid)
    print(f"[driver] {len(captions_flat)} captions across {len(sampled_ids)} images "
          f"({len(captions_flat) / len(sampled_ids):.1f} captions/image)")

    print("[driver] loading CLIP text tower to precompute caption embeddings...")
    from transformers import CLIPModel, CLIPProcessor
    MODEL_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    text_model = CLIPModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    caption_embeds = []
    with torch.no_grad():
        bs = 64
        for i in range(0, len(captions_flat), bs):
            batch = captions_flat[i : i + bs]
            inputs = processor(text=batch, return_tensors="pt", padding=True).to(args.device)
            feats = text_model.get_text_features(**inputs).pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            caption_embeds.append(feats.float().cpu())
    caption_embeds = torch.cat(caption_embeds, dim=0).numpy()  # [N_captions, proj_dim]
    print(f"[driver] caption_embeds shape={caption_embeds.shape}")
    del text_model
    torch.cuda.empty_cache()

    transform = transforms.Compose([
        transforms.Resize(config.image_shape[0]),
        transforms.CenterCrop(config.image_shape[0]),
    ])

    sched_q = multiprocessing.Queue()
    worker_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    control_q = multiprocessing.Queue()
    feedback_q = multiprocessing.Queue()
    scheduler = SchedulerModule(sched_q, worker_q, control_q, feedback_q)
    worker = WorkerModule(worker_q, result_q, feedback_q)
    scheduler.start()
    worker.start()

    encoder = get_transmission(config.transmission_policy_name)
    op_times = defaultdict(list)
    image_embeds_list = []
    t_start = time.time()

    try:
        control_q.put(("CONFIG", config))
        time.sleep(3.0)  # generous buffer -- this driver loads a second CLIPModel copy in the
                          # main process just beforehand (for caption embeddings), which has been
                          # observed to make the worker subprocess's own config-propagation race
                          # more likely to lose with only 1.0s (same transient class of issue
                          # documented for the DINOv3 driver earlier this session).

        for processed, iid in enumerate(sampled_ids):
            img_path = Path(COCO_IMAGES_ROOT) / id_to_file[iid]
            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            image_np = np.array(image, dtype=np.uint8)  # [H,W,3]

            t0 = time.time()
            for group_patches in encoder.encode(image_np[None], config):
                now = time.time()
                for p in group_patches:
                    p.arrival_time = now
                for p in group_patches:
                    sched_q.put(p)
            try:
                result = result_q.get(timeout=args.result_timeout)
            except queue_mod.Empty:
                print(f"    image {processed}: TIMEOUT waiting for InferenceResult -- aborting")
                raise
            wall = time.time() - t0

            embed = np.array(result.output[0], dtype=np.float32) if result.output else None
            image_embeds_list.append(embed)
            for ev in result.server_events:
                op_times[ev["type"]].append(ev["end"] - ev["start"])

            print(f"    image {processed + 1}/{len(sampled_ids)}: id={iid} wall={wall:.2f}s")
            sys.stdout.flush()
    finally:
        control_q.put(("STOP", None))
        result_q.cancel_join_thread()
        for proc in (scheduler, worker):
            proc.join(timeout=15)
            if proc.is_alive():
                proc.terminate()

    total_wall = time.time() - t_start
    image_embeds = np.stack(image_embeds_list, axis=0)  # [N_images, proj_dim]
    sim = image_embeds @ caption_embeds.T  # [N_images, N_captions]

    i2t_recall, t2i_recall = compute_recall_at_k(sim, sampled_ids, caption_owner)

    print(f"\n[driver] === Summary: {label} ===")
    print(f"    images: {len(sampled_ids)}  captions: {len(captions_flat)}")
    print(f"    i2t (image->text): R@1={i2t_recall[1]:.2f}%  R@5={i2t_recall[5]:.2f}%  R@10={i2t_recall[10]:.2f}%")
    print(f"    t2i (text->image): R@1={t2i_recall[1]:.2f}%  R@5={t2i_recall[5]:.2f}%  R@10={t2i_recall[10]:.2f}%")
    print(f"    total wall time: {total_wall:.1f}s ({total_wall / max(len(sampled_ids), 1):.2f}s/image avg)")

    print(f"\n[driver] === Mean per-op GPU time (ms), {label} ===")
    for op_name in sorted(op_times):
        vals = op_times[op_name]
        print(f"    {op_name:20s} mean={np.mean(vals) * 1000:8.3f}ms  n={len(vals)}")

    return {
        "label": label,
        "config": args.config,
        "grouping_strategy": raw_config.get("transmission_kwargs", {}).get("grouping_strategy"),
        "i2t_recall": i2t_recall,
        "t2i_recall": t2i_recall,
        "op_times_ms": {k: float(np.mean(v) * 1000) for k, v in op_times.items()},
        "total_wall_sec": total_wall,
    }


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
