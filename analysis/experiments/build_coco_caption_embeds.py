"""Precompute CLIP text embeddings for COCO val2017 captions, once, for retrieval eval.

Why a separate step rather than doing it inside the loader: the loader runs on the MOBILE client,
which by design holds no model -- it ships pixels and scores what comes back. Loading a 2.5B CLIP on
the client to encode 25k captions on every arm would put a second copy of the model on the GPU
beside the server's, four times over (ceiling / floor / ours 25 / ours 50), to recompute a tensor
that never changes. The text side is entirely independent of the approximation being studied, so it
is computed once here and cached.

The cache pins the image order too. Recall is computed against a fixed image list, so the loader and
this script must agree on which image is row `i` -- storing the ids together with the embeddings
makes that an assertion rather than a convention.

    python analysis/experiments/build_coco_caption_embeds.py
    python analysis/experiments/build_coco_caption_embeds.py --device cuda:0 --out <path>

Output: a torch .pt holding
    image_ids       [N]      COCO image ids, the row order recall is computed in
    text_embeds     [M, D]   L2-normalised, one row per caption
    text_to_image   [M]      index into image_ids for each caption's image
"""

from __future__ import annotations

import argparse
import json
import os

import torch

MODEL_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
DEFAULT_ANN = os.path.expanduser("~/fiftyone/coco-2017/raw/captions_val2017.json")
DEFAULT_OUT = "analysis/results/coco_retrieval/caption_embeds_bigg.pt"


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=DEFAULT_ANN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()

    if not os.path.exists(a.annotations):
        raise SystemExit(f"captions file not found: {a.annotations}")

    from transformers import CLIPModel, CLIPProcessor

    ann = json.load(open(a.annotations))
    # Fixed, explicit order: sorted image ids. The loader sorts its file list the same way, and
    # asserts the ids match, so a silent misalignment cannot turn into a plausible-looking recall.
    image_ids = sorted({int(im["id"]) for im in ann["images"]})
    id_to_row = {iid: r for r, iid in enumerate(image_ids)}

    captions, text_to_image = [], []
    for c in ann["annotations"]:
        row = id_to_row.get(int(c["image_id"]))
        if row is None:
            continue
        captions.append(str(c["caption"]).strip())
        text_to_image.append(row)
    print(f"[build] {len(image_ids)} images, {len(captions)} captions")

    model = CLIPModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(a.device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_ID)

    embeds = []
    for i in range(0, len(captions), a.batch_size):
        chunk = captions[i:i + a.batch_size]
        inputs = proc(text=chunk, return_tensors="pt", padding=True, truncation=True).to(a.device)
        feats = model.get_text_features(**inputs).pooler_output.float()
        embeds.append((feats / feats.norm(dim=-1, keepdim=True)).cpu())
        if (i // a.batch_size) % 20 == 0:
            print(f"[build] {i + len(chunk)}/{len(captions)}", flush=True)

    text_embeds = torch.cat(embeds, dim=0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({
        "image_ids": torch.tensor(image_ids, dtype=torch.long),
        "text_embeds": text_embeds,
        "text_to_image": torch.tensor(text_to_image, dtype=torch.long),
        "model_id": MODEL_ID,
        "annotations": a.annotations,
    }, a.out)
    print(f"[build] wrote {a.out}  text_embeds={tuple(text_embeds.shape)}")


if __name__ == "__main__":
    main()
