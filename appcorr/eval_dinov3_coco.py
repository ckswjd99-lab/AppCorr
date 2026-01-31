import torch
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
from torchvision.transforms import v2
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Import your model definition
from models.dinov3.hub.detectors import dinov3_vit7b16_de

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

@torch.inference_mode()
def run_evaluation():
    # 1. Setup Dataset
    # Using small max_samples for testing
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        # max_samples=10,
    )
    
    # Path to COCO annotations
    ann_file = os.path.expanduser("~/fiftyone/coco-2017/raw/instances_val2017.json")
    print(f"Using annotation file: {ann_file}")

    # 2. Load Model (DINOv3 7B)
    print("Loading model...")
    model_detector = dinov3_vit7b16_de(
        pretrained=True,
        weights="~/cjpark/weights/dinov3/dinov3_vit7b16_coco_detr_head-b0235ff7.pth",
        backbone_weights="~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    ).to(torch.bfloat16).to("cuda").eval()

    img_size = 1024
    transform = make_transform(img_size)
    
    # 3. Optimized Inference Loop
    coco_results = []
    processed_ids = []
    
    print("Starting inference...")
    with torch.autocast('cuda', dtype=torch.bfloat16):
        with fo.ProgressBar() as pb:
            for sample in pb(dataset):
                # Load and transform image
                img = Image.open(sample.filepath).convert("RGB")
                w, h = img.size
                batch_img = transform(img)[None].to("cuda").to(torch.bfloat16)
                
                # Inference
                outputs = model_detector(batch_img)
                preds = outputs[0]
                
                scores = preds['scores'].cpu()
                labels = preds['labels'].cpu()
                boxes = preds['boxes'].cpu()
                
                # Extract image ID from filename (standard COCO naming)
                try:
                    image_id = int(os.path.basename(sample.filepath).split('.')[0])
                    processed_ids.append(image_id)
                except ValueError:
                    print(f"Warning: Could not extract image info from {sample.filepath}")
                    continue

                for score, label, box in zip(scores, labels, boxes):
                    # Convert [x1, y1, x2, y2] (@img_size) -> absolute [x, y, w, h] (@orig_size)
                    x1, y1, x2, y2 = box.tolist()
                    
                    # Normalize to 0-1
                    nx1, ny1 = x1 / img_size, y1 / img_size
                    nx2, ny2 = x2 / img_size, y2 / img_size
                    
                    # Scale to original image size
                    abs_x1 = nx1 * w
                    abs_y1 = ny1 * h
                    abs_w = (nx2 - nx1) * w
                    abs_h = (ny2 - ny1) * h
                    
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": label.item(), # logic: indices match COCO IDs
                        "bbox": [abs_x1, abs_y1, abs_w, abs_h],
                        "score": float(score)
                    })
                
                # Manual memory cleanup
                del batch_img, outputs, preds

    # 5. Run COCO Evaluation
    print("Running COCO evaluation...")
    
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(coco_results)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Filter to only the images we processed
    coco_eval.params.imgIds = sorted(list(set(processed_ids)))
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    print(f"Overall mAP: {coco_eval.stats[0]}")
    
if __name__ == "__main__":
    run_evaluation()