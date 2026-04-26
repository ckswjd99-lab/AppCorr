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


class ADE20KLoader(DatasetLoader):
    def __init__(self, root: str, batch_size: int, image_size: int = 896, num_workers: int = 4, **kwargs):
        super().__init__(root, batch_size, **kwargs)
        self.image_size = image_size
        self.num_workers = num_workers
        self.dataset_name = kwargs.get('hf_dataset_name', 'merve/scene_parse_150')
        self.dataset_config = kwargs.get('hf_dataset_config', None)
        self.split = kwargs.get('split', 'validation')
        self.num_classes = int(kwargs.get('num_classes', 150))
        self.ignore_index = int(kwargs.get('ignore_index', 255))
        self.reduce_zero_label = bool(kwargs.get('reduce_zero_label', True))
        self.total_samples = 0
        self.total_area_intersect = torch.zeros(self.num_classes, dtype=torch.float64)
        self.total_area_union = torch.zeros(self.num_classes, dtype=torch.float64)
        self.total_area_pred_label = torch.zeros(self.num_classes, dtype=torch.float64)
        self.total_area_label = torch.zeros(self.num_classes, dtype=torch.float64)

    @staticmethod
    def _resize_short_side(image, short_side: int):
        from PIL import Image

        width, height = image.size
        if height > width:
            new_width = short_side
            new_height = int(short_side * height / width + 0.5)
        else:
            new_height = short_side
            new_width = int(short_side * width / height + 0.5)
        return image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    @staticmethod
    def _resize_mask(mask, size):
        from PIL import Image

        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.asarray(mask))
        return mask.resize(size, Image.Resampling.NEAREST)

    def get_loader(self) -> torch.utils.data.DataLoader:
        from datasets import load_dataset
        from torchvision.transforms import v2
        import os

        cache_dir = os.path.expanduser(self.root) if self.root else None
        load_kwargs = {
            'split': self.split,
            'cache_dir': cache_dir,
        }
        if self.dataset_config is not None:
            load_kwargs['name'] = self.dataset_config
        hf_dataset = load_dataset(self.dataset_name, **load_kwargs)

        class HFADE20KDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, image_size: int, resize_fn):
                self.dataset = dataset
                self.image_size = image_size
                self.resize_fn = resize_fn
                self.to_image = v2.ToImage()
                self.to_uint8 = v2.ToDtype(torch.uint8, scale=False)

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                sample = self.dataset[idx]
                image = sample['image'].convert('RGB')
                annotation = sample['annotation']
                orig_w, orig_h = image.size
                orig_annotation_np = np.asarray(annotation, dtype=np.uint8)
                resized = self.resize_fn(image, self.image_size)
                resized_w, resized_h = resized.size
                annotation = ADE20KLoader._resize_mask(annotation, resized.size)
                annotation_np = np.asarray(annotation, dtype=np.uint8)
                image_t = self.to_uint8(self.to_image(resized))
                label = {
                    'idx': int(idx),
                    'orig_width': int(orig_w),
                    'orig_height': int(orig_h),
                    'input_width': int(resized_w),
                    'input_height': int(resized_h),
                    'scene_category': sample.get('scene_category', ''),
                    'mask': annotation_np,
                    'orig_mask': orig_annotation_np,
                }
                return image_t, label

        dataset = HFADE20KDataset(hf_dataset, self.image_size, self._resize_short_side)

        def collate_variable(batch):
            images, labels = zip(*batch)
            return list(images), list(labels)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_variable,
        )

    def evaluate_batch(self, preds: List[Any], labels: List[Any], **kwargs) -> Dict[str, Any]:
        count = len(labels)
        self.total_samples += count
        pred_summaries = []
        batch_intersect = torch.zeros(self.num_classes, dtype=torch.float64)
        batch_union = torch.zeros(self.num_classes, dtype=torch.float64)
        batch_pred_label = torch.zeros(self.num_classes, dtype=torch.float64)
        batch_label = torch.zeros(self.num_classes, dtype=torch.float64)
        for pred, label in zip(preds, labels):
            if not isinstance(pred, dict):
                continue
            pred_mask = pred.get('mask')
            gt_mask = None
            if isinstance(label, dict):
                pred_shape = tuple(np.asarray(pred_mask).shape) if pred_mask is not None else None
                orig_mask = label.get('orig_mask')
                if orig_mask is not None and pred_shape == tuple(np.asarray(orig_mask).shape):
                    gt_mask = orig_mask
                else:
                    gt_mask = label.get('mask')
            if pred_mask is not None and gt_mask is not None:
                areas = self._intersect_and_union(pred_mask, gt_mask)
                batch_intersect += areas[0]
                batch_union += areas[1]
                batch_pred_label += areas[2]
                batch_label += areas[3]
            sample_idx = label.get('idx') if isinstance(label, dict) else None
            pred_summaries.append({
                'idx': sample_idx,
                'shape': pred.get('shape'),
                'sha256': pred.get('sha256'),
            })
        self.total_area_intersect += batch_intersect
        self.total_area_union += batch_union
        self.total_area_pred_label += batch_pred_label
        self.total_area_label += batch_label
        metric_values = self._metrics_from_areas(
            batch_intersect,
            batch_union,
            batch_pred_label,
            batch_label,
        )
        return {
            'segmented_images': count,
            **metric_values,
            'predictions': pred_summaries,
        }

    def get_pbar_desc(self) -> str:
        metrics = self._metrics_from_areas(
            self.total_area_intersect,
            self.total_area_union,
            self.total_area_pred_label,
            self.total_area_label,
        )
        if metrics:
            return (
                f"mIoU: {metrics['mIoU']:.2f} | "
                f"aAcc: {metrics['aAcc']:.2f} | "
                f"Processed: {self.total_samples}"
            )
        return f"Processed: {self.total_samples}"

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_samples': self.total_samples,
            **self._metrics_from_areas(
                self.total_area_intersect,
                self.total_area_union,
                self.total_area_pred_label,
                self.total_area_label,
            ),
        }

    def _intersect_and_union(self, pred_mask, gt_mask) -> torch.Tensor:
        pred = torch.as_tensor(np.asarray(pred_mask), dtype=torch.long)
        gt = torch.as_tensor(np.asarray(gt_mask), dtype=torch.long)

        if tuple(pred.shape) != tuple(gt.shape):
            raise ValueError(f"Prediction/GT shape mismatch: {tuple(pred.shape)} != {tuple(gt.shape)}")

        if self.reduce_zero_label:
            gt = gt.clone()
            gt[gt == self.ignore_index] += 1
            gt -= 1
            gt[gt == -1] = self.ignore_index

        valid = gt != self.ignore_index
        pred = pred[valid]
        gt = gt[valid]
        intersect = pred[pred == gt]

        area_intersect = torch.bincount(intersect, minlength=self.num_classes)[:self.num_classes].double()
        area_pred_label = torch.bincount(pred, minlength=self.num_classes)[:self.num_classes].double()
        area_label = torch.bincount(gt, minlength=self.num_classes)[:self.num_classes].double()
        area_union = area_pred_label + area_label - area_intersect
        return torch.stack([area_intersect, area_union, area_pred_label, area_label])

    @staticmethod
    def _nanmean_percent(values: torch.Tensor) -> float:
        if values.numel() == 0:
            return 0.0
        valid = values[~torch.isnan(values)]
        if valid.numel() == 0:
            return 0.0
        return float(valid.mean().item() * 100.0)

    def _metrics_from_areas(
        self,
        area_intersect: torch.Tensor,
        area_union: torch.Tensor,
        area_pred_label: torch.Tensor,
        area_label: torch.Tensor,
    ) -> Dict[str, float]:
        if area_label.sum().item() == 0:
            return {}

        eps = torch.finfo(torch.float64).eps
        iou = area_intersect / area_union.clamp_min(eps)
        acc = area_intersect / area_label.clamp_min(eps)
        dice = 2 * area_intersect / (area_pred_label + area_label).clamp_min(eps)
        precision = area_intersect / area_pred_label.clamp_min(eps)
        recall = area_intersect / area_label.clamp_min(eps)
        fscore = 2 * precision * recall / (precision + recall).clamp_min(eps)
        aacc = area_intersect.sum() / area_label.sum().clamp_min(eps)

        fscore_valid = (area_pred_label + area_label) != 0
        iou[area_union == 0] = torch.nan
        acc[area_label == 0] = torch.nan
        dice[(area_pred_label + area_label) == 0] = torch.nan
        precision[area_pred_label == 0] = torch.nan
        recall[area_label == 0] = torch.nan
        fscore[~fscore_valid] = torch.nan

        return {
            'mIoU': self._nanmean_percent(iou),
            'acc': self._nanmean_percent(acc),
            'aAcc': float(aacc.item() * 100.0),
            'dice': self._nanmean_percent(dice),
            'fscore': self._nanmean_percent(fscore),
            'precision': self._nanmean_percent(precision),
            'recall': self._nanmean_percent(recall),
        }


def get_dataset_loader(name: str, root: str, batch_size: int, **kwargs) -> DatasetLoader:
    if name == 'imagenet-1k':
        return ImageNetLoader(root, batch_size, **kwargs)
    elif name == 'coco2017':
        return COCO2017Loader(root, batch_size, **kwargs)
    elif name in {'ade20k', 'scene_parse_150'}:
        return ADE20KLoader(root, batch_size, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
