from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import time
import torch
from torchvision import datasets, transforms
import numpy as np

class DatasetLoader(ABC):
    def __init__(self, root: str, batch_size: int, **kwargs):
        self.root = root
        self.batch_size = batch_size
        self.kwargs = kwargs

    @abstractmethod
    def get_loader(self) -> torch.utils.data.DataLoader:
        pass

    @abstractmethod
    def evaluate_batch(self, preds: List[Any], labels: List[Any], **kwargs) -> Dict[str, Any]:
        """
        Evaluate a single batch of predictions against labels.
        Returns a dictionary of metrics for this batch (e.g., {'top1': 5, 'total': 32}).
        """
        pass

    @abstractmethod
    def get_pbar_desc(self) -> str:
        """
        Returns a short description for the progress bar (e.g., 'Processed: 50').
        Should be lightweight to call repeatedly.
        """
        pass

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """
        Returns a summary of the evaluation (e.g., {'top1_acc': 75.0}).
        """
        pass

class ImageNetLoader(DatasetLoader):
    def __init__(self, root: str, batch_size: int, image_size: int = 256, num_workers: int = 4, **kwargs):
        super().__init__(root, batch_size, **kwargs)
        self.image_size = image_size
        self.num_workers = num_workers
        self.total_top1 = 0
        self.total_top5 = 0
        self.total_samples = 0

    def get_loader(self) -> torch.utils.data.DataLoader:
        val_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.PILToTensor(),
        ])

        val_dataset = datasets.ImageFolder(root=self.root, transform=val_transforms)
        
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False 
        )

    def evaluate_batch(self, preds: List[Any], labels: List[Any], **kwargs) -> Dict[str, Any]:
        batch_top1 = 0
        batch_top5 = 0
        curr_bs = len(labels)
        
        # Ensure preds matches labels length (handle partial batches if necessary, though logic usually aligns them)
        # preds is list of lists (top-k)
        
        for i in range(curr_bs):
            label = labels[i]
            if i >= len(preds): break # Should not happen if aligned
            p_list = preds[i]
            
            if not p_list: continue
            
            if p_list[0] == label:
                batch_top1 += 1
            if label in p_list:
                batch_top5 += 1

        self.total_top1 += batch_top1
        self.total_top5 += batch_top5
        self.total_samples += curr_bs
        
        return {
            'top1': batch_top1,
            'top5': batch_top5,
            'total': curr_bs,
            'acc1': batch_top1 / curr_bs * 100 if curr_bs > 0 else 0.0,
            'acc5': batch_top5 / curr_bs * 100 if curr_bs > 0 else 0.0
        }

    def get_pbar_desc(self) -> str:
        # For ImageNet, calculating per-step accuracy is cheap, so we can show it.
        acc1 = self.total_top1 / self.total_samples * 100 if self.total_samples > 0 else 0.0
        acc5 = self.total_top5 / self.total_samples * 100 if self.total_samples > 0 else 0.0
        return f"Acc@1: {acc1:.2f} | Acc@5: {acc5:.2f}"

    def get_summary(self) -> Dict[str, Any]:
        acc1 = self.total_top1 / self.total_samples * 100 if self.total_samples > 0 else 0.0
        acc5 = self.total_top5 / self.total_samples * 100 if self.total_samples > 0 else 0.0
        return {
            'total_samples': self.total_samples,
            'top1_acc': acc1,
            'top5_acc': acc5
        }

class COCO2017Loader(DatasetLoader):
    def __init__(self, root: str, batch_size: int, image_size: int = 1024, num_workers: int = 4, **kwargs):
        super().__init__(root, batch_size, **kwargs)
        self.image_size = image_size
        self.num_workers = num_workers
        self.coco_results = []
        self.processed_ids = []
        
        # Lazy load heavy dependencies
        import fiftyone.zoo as foz
        from pycocotools.coco import COCO
        import os

        print("[COCOLoader] Loading FiftyOne COCO-2017 Validation Split...")
        # Note: 'root' argument is ignored as FiftyOne manages its own dataset path.
        self.fo_dataset = foz.load_zoo_dataset("coco-2017", split="validation")
        
        self.ann_file = os.path.expanduser("~/fiftyone/coco-2017/raw/instances_val2017.json")
        if not os.path.exists(self.ann_file):
            print(f"!!! [COCOLoader] Annotation file not found at {self.ann_file}. Evaluation might fail.")
        else:
             self.coco_gt = COCO(self.ann_file)

    def get_loader(self) -> torch.utils.data.DataLoader:
        from torch.utils.data import Dataset
        from PIL import Image
        from torchvision.transforms import v2
        import os

        class FiftyOneTorchDataset(Dataset):
            def __init__(self, fo_dataset, transform=None):
                self.sample_ids = fo_dataset.values("id")
                self.filepaths = fo_dataset.values("filepath")
                self.transform = transform
            
            def __len__(self):
                return len(self.filepaths)
            
            def __getitem__(self, idx):
                filepath = self.filepaths[idx]
                img = Image.open(filepath).convert("RGB")
                w, h = img.size
                
                if self.transform:
                    img_t = self.transform(img)
                else:
                    img_t = transforms.ToTensor()(img)
                
                # Extract proper COCO image ID from filename
                try:
                    image_id = int(os.path.basename(filepath).split('.')[0])
                except ValueError:
                    image_id = -1
                
                # Return (image, (image_id, w, h))
                # The label here is metadata needed for post-processing
                return img_t, torch.tensor([image_id, w, h], dtype=torch.long)

        # Keep COCO at native resolution. Transmission policies decide how to
        # project it onto the model input grid.
        to_tensor = v2.ToImage()
        to_uint8 = v2.ToDtype(torch.uint8, scale=False)
        tfm = v2.Compose([to_tensor, to_uint8])

        ds = FiftyOneTorchDataset(self.fo_dataset, transform=tfm)

        def collate_native(batch):
            images, labels = zip(*batch)
            return list(images), torch.stack(labels, dim=0)
        
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_native,
        )

    def evaluate_batch(self, preds: List[Any], labels: List[Any], **kwargs) -> Dict[str, Any]:
        count = 0
        input_size = self.image_size

        for i, pred in enumerate(preds):
            # labels[i] is [image_id, w, h] from our custom dataset
            meta = labels[i]
            if isinstance(meta, torch.Tensor):
                meta = meta.tolist()
            
            image_id, w, h = meta
            if image_id == -1: continue
            
            count += 1
            if image_id not in self.processed_ids:
                self.processed_ids.append(image_id)

            # Expecting pred to have 'scores', 'labels', 'boxes' keys
            if not isinstance(pred, dict) or 'boxes' not in pred:
                 continue
                 
            scores = pred['scores']
            cat_ids = pred['labels']
            boxes = pred['boxes']
            
            for score, cat_id, box in zip(scores, cat_ids, boxes):
                x1, y1, x2, y2 = box 
                
                # Normalize (assuming box is in pixels of input_size)
                nx1, ny1 = x1 / input_size, y1 / input_size
                nx2, ny2 = x2 / input_size, y2 / input_size
                
                # Scale to original
                abs_x1 = nx1 * w
                abs_y1 = ny1 * h
                abs_w = (nx2 - nx1) * w
                abs_h = (ny2 - ny1) * h
                
                self.coco_results.append({
                    "image_id": int(image_id),
                    "category_id": int(cat_id),
                    "bbox": [float(abs_x1), float(abs_y1), float(abs_w), float(abs_h)],
                    "score": float(score)
                })

        return {"detected_images": count}

    def get_pbar_desc(self) -> str:
        return f"Processed: {len(self.processed_ids)}"

    def get_summary(self) -> Dict[str, Any]:
        from pycocotools.cocoeval import COCOeval
        import io
        import contextlib

        if not self.coco_results:
            return {"mAP": 0.0, "status": "no_results"}
            
        print("[COCOLoader] Running Evaluation...")
        
        try:
            coco_dt = self.coco_gt.loadRes(self.coco_results)
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
            coco_eval.params.imgIds = sorted(list(set(self.processed_ids)))
            
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            return {
                "mAP": coco_eval.stats[0].item(),
                "mAP_50": coco_eval.stats[1].item(),
                "mAP_75": coco_eval.stats[2].item(),
                "total_images": len(self.processed_ids)
            }
        except Exception as e:
            print(f"!!! [COCOLoader] Eval Failed: {e}")
            return {"error": str(e)}


def get_dataset_loader(name: str, root: str, batch_size: int, **kwargs) -> DatasetLoader:
    if name == 'imagenet-1k':
        return ImageNetLoader(root, batch_size, **kwargs)
    elif name == 'coco2017':
        return COCO2017Loader(root, batch_size, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
