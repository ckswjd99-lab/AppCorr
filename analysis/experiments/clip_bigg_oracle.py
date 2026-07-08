"""
clip_bigg_oracle.py

Standalone correctness oracle for the CLIP-ViT-bigG/14 AppCorr fork (Phase 0).

Loads the stock `transformers.CLIPModel` (laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) and dumps:
  - per-layer vision CLS hidden states for a handful of real ImageNet val images (48 layers)
  - final normalized image embeddings for those images
  - all 1000 ImageNet zero-shot class text embeddings (80-template ensemble, averaged+normalized,
    using open_clip's IMAGENET_CLASSNAMES/OPENAI_IMAGENET_TEMPLATES)
  - a handful of real COCO caption text embeddings

Everything is saved to a .pt file. Phase 1's vision-tower fork unit tests compare against this.

Run (appcorr env):
    python analysis/experiments/clip_bigg_oracle.py --out /tmp/clip_bigg_oracle.pt --num-images 4
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
IMAGENET_VAL_ROOT = "/NHNHOME/share/cjpark/data/imagenet_val"
COCO_CAPTIONS_JSON = "/home/nxclab/fiftyone/coco-2017/raw/captions_val2017.json"
COCO_IMAGES_ROOT = "/home/nxclab/fiftyone/coco-2017/validation/data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="/tmp/clip_bigg_oracle.pt")
    p.add_argument("--num-images", type=int, default=4)
    p.add_argument("--num-captions", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def build_zeroshot_classifier(model, processor, device):
    from open_clip import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

    all_embeds = []
    with torch.no_grad():
        for classname in IMAGENET_CLASSNAMES:
            texts = [tmpl(classname) for tmpl in OPENAI_IMAGENET_TEMPLATES]
            inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
            text_feats = model.get_text_features(**inputs).pooler_output
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            class_embed = text_feats.mean(dim=0)
            class_embed = class_embed / class_embed.norm()
            all_embeds.append(class_embed)
    return torch.stack(all_embeds, dim=0)  # [1000, proj_dim]


def main():
    args = parse_args()
    device = args.device

    print(f"[oracle] loading {MODEL_ID} ...")
    model = CLIPModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    val_root = Path(IMAGENET_VAL_ROOT)
    class_dirs = sorted([d for d in val_root.iterdir() if d.is_dir()])[: args.num_images]
    image_paths = []
    for d in class_dirs:
        imgs = sorted(d.glob("*"))
        if imgs:
            image_paths.append(imgs[0])
    print(f"[oracle] using {len(image_paths)} ImageNet val images: {[p.name for p in image_paths]}")

    images = [Image.open(p).convert("RGB") for p in image_paths]
    pixel_inputs = processor(images=images, return_tensors="pt").to(device)
    pixel_values = pixel_inputs["pixel_values"].to(dtype=torch.bfloat16)

    with torch.no_grad():
        vision_out = model.vision_model(
            pixel_values=pixel_values, output_hidden_states=True, return_dict=True
        )
        per_layer_cls = torch.stack(
            [h[:, 0, :].float().cpu() for h in vision_out.hidden_states], dim=0
        )  # [num_layers+1, B, hidden] (includes embedding layer output at idx 0)
        image_embeds = model.get_image_features(pixel_values=pixel_values).pooler_output
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    print(f"[oracle] per_layer_cls shape={tuple(per_layer_cls.shape)}")
    print(f"[oracle] image_embeds shape={tuple(image_embeds.shape)}")

    print("[oracle] building 1000-class zero-shot classifier (80-template ensemble)...")
    zeroshot_weights = build_zeroshot_classifier(model, processor, device)
    print(f"[oracle] zeroshot_weights shape={tuple(zeroshot_weights.shape)}")

    logits = 100.0 * image_embeds.float() @ zeroshot_weights.float().T
    top5 = logits.topk(5, dim=-1).indices.cpu()
    print(f"[oracle] sanity top5 preds per image:\n{top5}")

    print(f"[oracle] loading {args.num_captions} real COCO captions...")
    with open(COCO_CAPTIONS_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)
    captions = [ann["caption"] for ann in coco["annotations"][: args.num_captions]]
    with torch.no_grad():
        cap_inputs = processor(text=captions, return_tensors="pt", padding=True).to(device)
        cap_embeds = model.get_text_features(**cap_inputs).pooler_output
        cap_embeds = cap_embeds / cap_embeds.norm(dim=-1, keepdim=True)
    print(f"[oracle] caption_embeds shape={tuple(cap_embeds.shape)}")

    torch.save(
        {
            "model_id": MODEL_ID,
            "image_paths": [str(p) for p in image_paths],
            "per_layer_cls": per_layer_cls,
            "image_embeds": image_embeds.float().cpu(),
            "zeroshot_weights": zeroshot_weights.float().cpu(),
            "zeroshot_top5": top5,
            "captions": captions,
            "caption_embeds": cap_embeds.float().cpu(),
            "logit_scale": model.logit_scale.exp().item(),
        },
        args.out,
    )
    print(f"[oracle] saved to {args.out}")


if __name__ == "__main__":
    main()
