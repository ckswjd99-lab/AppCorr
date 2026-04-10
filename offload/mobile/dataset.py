from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import os
import time
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

class DatasetLoader(ABC):
    def __init__(self, root: str, batch_size: int, **kwargs):
        self.root = os.path.abspath(os.path.expanduser(root))
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

        # Transform: To Image -> Resize -> To Uint8 Tensor
        to_tensor = v2.ToImage()
        resize = v2.Resize((self.image_size, self.image_size), antialias=True)
        to_uint8 = v2.ToDtype(torch.uint8, scale=False)
        tfm = v2.Compose([to_tensor, resize, to_uint8])

        ds = FiftyOneTorchDataset(self.fo_dataset, transform=tfm)
        
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
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
        

class ADE20KLoader(DatasetLoader):
    """
    Expected directory layout:

    root/
      images/
        validation/
          ADE_val_00000001.jpg
          ...
      annotations/
        validation/
          ADE_val_00000001.png
          ...

    Annotation PNG convention:
      - class ids stored as 1..150
      - 0 is background / ignore
    This loader converts labels to 0..149 and keeps ignore_index=255.
    """

    def __init__(
        self,
        root: str,
        batch_size: int,
        image_size: int = 512,
        num_workers: int = 4,
        ignore_index: int = 255,
        **kwargs,
    ):
        super().__init__(root, batch_size, **kwargs)
        self.image_size = image_size
        self.num_workers = num_workers
        self.ignore_index = ignore_index

        self.total_intersection = None
        self.total_union = None
        self.total_correct = 0
        self.total_labeled = 0
        self.total_samples = 0

    def get_loader(self) -> torch.utils.data.DataLoader:
        import os
        from PIL import Image
        from torch.utils.data import Dataset
        from torchvision.transforms import v2

        image_root = os.path.join(self.root, "images", "validation")
        label_root = os.path.join(self.root, "annotations", "validation")

        class ADE20KTorchDataset(Dataset):
            def __init__(self, image_root, label_root, image_size, ignore_index):
                self.image_root = image_root
                self.label_root = label_root
                self.image_size = image_size
                self.ignore_index = ignore_index

                image_files = sorted(
                    [
                        f for f in os.listdir(image_root)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )

                self.samples = []
                for img_name in image_files:
                    stem = os.path.splitext(img_name)[0]
                    label_name = f"{stem}.png"
                    label_path = os.path.join(label_root, label_name)
                    if os.path.exists(label_path):
                        self.samples.append(
                            (
                                os.path.join(image_root, img_name),
                                label_path,
                            )
                        )

                if len(self.samples) == 0:
                    raise RuntimeError(
                        f"No ADE20K validation samples found under {image_root} and {label_root}"
                    )

                self.img_transform = v2.Compose([
                    v2.ToImage(),
                    v2.Resize((image_size, image_size), antialias=True),
                    v2.ToDtype(torch.uint8, scale=False),
                ])

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_path, label_path = self.samples[idx]

                img = Image.open(img_path).convert("RGB")
                label = Image.open(label_path)

                img_t = self.img_transform(img)

                # segmentation mask: nearest resize, keep integer ids
                label_np = np.array(label, dtype=np.uint8)
                label_t = torch.from_numpy(label_np).unsqueeze(0)  # [1, H, W]
                label_t = F.interpolate(
                    label_t.unsqueeze(0).float(),
                    size=(self.image_size, self.image_size),
                    mode="nearest",
                ).squeeze(0).squeeze(0).to(torch.uint8)

                # ADE20K GT is commonly 1..150, with 0 as background/void.
                # Convert valid classes to 0..149, and background/void -> ignore_index.
                label_t = label_t.clone()
                void_mask = label_t == 0
                label_t = label_t.long() - 1
                label_t[void_mask] = self.ignore_index

                return img_t, label_t

        ds = ADE20KTorchDataset(
            image_root=image_root,
            label_root=label_root,
            image_size=self.image_size,
            ignore_index=self.ignore_index,
        )

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def _fast_hist(self, pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
        valid = (target >= 0) & (target < num_classes)
        hist = torch.bincount(
            num_classes * target[valid].to(torch.int64) + pred[valid].to(torch.int64),
            minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes)
        return hist

    def evaluate_batch(self, preds: List[Any], labels: List[Any], **kwargs) -> Dict[str, Any]:
        """
        preds:
          either list of HxW prediction maps
          or dict payloads containing "segmentation"
        labels:
          batch of GT masks
        """
        if isinstance(labels, torch.Tensor):
            gt_batch = labels
        else:
            gt_batch = torch.as_tensor(labels)

        num_classes = kwargs.get("num_classes", 150)

        if self.total_intersection is None:
            self.total_intersection = torch.zeros(num_classes, dtype=torch.float64)
            self.total_union = torch.zeros(num_classes, dtype=torch.float64)

        batch_correct = 0
        batch_labeled = 0
        batch_size = gt_batch.shape[0]

        for i in range(batch_size):
            gt = gt_batch[i]
            if i >= len(preds):
                break

            pred_item = preds[i]
            if isinstance(pred_item, dict) and "segmentation" in pred_item:
                pred = pred_item["segmentation"]
                pred = torch.from_numpy(pred) if isinstance(pred, np.ndarray) else pred
            else:
                pred = pred_item
                pred = torch.from_numpy(pred) if isinstance(pred, np.ndarray) else pred

            pred = pred.to(torch.long)
            gt = gt.to(torch.long)

            if pred.shape != gt.shape:
                pred = F.interpolate(
                    pred.unsqueeze(0).unsqueeze(0).float(),
                    size=gt.shape[-2:],
                    mode="nearest",
                ).squeeze(0).squeeze(0).long()

            valid = gt != self.ignore_index
            batch_correct += (pred[valid] == gt[valid]).sum().item()
            batch_labeled += valid.sum().item()

            hist = self._fast_hist(pred[valid], gt[valid], num_classes)
            inter = torch.diag(hist).to(torch.float64)
            union = (hist.sum(1) + hist.sum(0) - torch.diag(hist)).to(torch.float64)

            self.total_intersection += inter
            self.total_union += union

        self.total_correct += batch_correct
        self.total_labeled += batch_labeled
        self.total_samples += batch_size

        pix_acc = 100.0 * batch_correct / batch_labeled if batch_labeled > 0 else 0.0
        iou = self.total_intersection / torch.clamp(self.total_union, min=1.0)
        miou = 100.0 * iou.mean().item()

        return {
            "total": batch_size,
            "pix_acc": pix_acc,
            "miou": miou,
        }

    def get_pbar_desc(self) -> str:
        pix_acc = 100.0 * self.total_correct / self.total_labeled if self.total_labeled > 0 else 0.0
        if self.total_intersection is None:
            miou = 0.0
        else:
            iou = self.total_intersection / torch.clamp(self.total_union, min=1.0)
            miou = 100.0 * iou.mean().item()
        return f"mIoU: {miou:.2f} | PixAcc: {pix_acc:.2f}"

    def get_summary(self) -> Dict[str, Any]:
        pix_acc = 100.0 * self.total_correct / self.total_labeled if self.total_labeled > 0 else 0.0
        if self.total_intersection is None:
            miou = 0.0
            per_class_iou = []
        else:
            iou = self.total_intersection / torch.clamp(self.total_union, min=1.0)
            miou = 100.0 * iou.mean().item()
            per_class_iou = (100.0 * iou).cpu().tolist()

        return {
            "total_samples": self.total_samples,
            "pixel_acc": pix_acc,
            "mIoU": miou,
            "per_class_iou": per_class_iou,
        }


def get_dataset_loader(name: str, root: str, batch_size: int, **kwargs) -> DatasetLoader:
    if name == 'imagenet-1k':
        return ImageNetLoader(root, batch_size, **kwargs)
    elif name == 'coco2017':
        return COCO2017Loader(root, batch_size, **kwargs)
    elif name == 'ade20k':
        return ADE20KLoader(root, batch_size, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
