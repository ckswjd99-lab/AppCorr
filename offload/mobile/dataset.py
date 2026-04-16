from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import time
import torch
from torchvision import datasets, transforms
import numpy as np
import json
import os


def _infer_fiftyone_label_field(dataset: Any, fallback: str = "ground_truth") -> str:
    field_name = getattr(dataset, "default_label_field", None)
    if isinstance(field_name, str) and field_name:
        return field_name

    try:
        import fiftyone as fo

        schema = dataset.get_field_schema()
        for name, field in schema.items():
            document_type = getattr(field, "document_type", None)
            if document_type in (fo.Detections, fo.Polylines, fo.Keypoints):
                return name
    except Exception:
        pass

    return fallback


def _infer_fiftyone_classes(dataset: Any, label_field: str) -> List[str]:
    classes = getattr(dataset, "default_classes", None)
    if classes:
        return list(classes)

    try:
        info = dataset.info or {}
        field_classes = info.get("classes", {}).get(label_field)
        if field_classes:
            return list(field_classes)
    except Exception:
        pass

    return []


class DatasetLoader(ABC):
    def __init__(self, root: Optional[str], batch_size: int, **kwargs):
        self.root = root or ""
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.log_dir: Optional[str] = None

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

    def set_log_dir(self, log_dir: str) -> None:
        self.log_dir = log_dir

class ImageNetLoader(DatasetLoader):
    def __init__(self, root: Optional[str], batch_size: int, image_size: int = 256, num_workers: int = 4, **kwargs):
        super().__init__(root, batch_size, **kwargs)
        self.image_size = image_size
        self.num_workers = num_workers
        self.total_top1 = 0
        self.total_top5 = 0
        self.total_samples = 0

    def get_loader(self) -> torch.utils.data.DataLoader:
        if not self.root:
            raise ValueError("ImageNetLoader requires a dataset root.")
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
    def __init__(self, root: Optional[str], batch_size: int, image_size: int = 1024, num_workers: int = 4, **kwargs):
        super().__init__(root, batch_size, **kwargs)
        self.image_size = image_size
        self.num_workers = num_workers
        self.preserve_original_resolution = bool(kwargs.get("preserve_original_resolution", False))
        self.coco_results = []
        self.processed_ids = []
        self.sample_records: Dict[int, Dict[str, Any]] = {}
        self.predictions_by_image: Dict[int, List[Dict[str, Any]]] = {}
        self.export_info: Dict[str, str] = {}
        raw_split = str(kwargs.get("split", "validation")).strip().lower()
        split_aliases = {
            "val": "validation",
            "validation": "validation",
            "train": "train",
        }
        if raw_split not in split_aliases:
            raise ValueError(
                f"Unsupported COCO split '{raw_split}'. Available splits: train, validation"
            )
        self.split = split_aliases[raw_split]
        self.shuffle = bool(kwargs.get("shuffle", self.split == "train"))
        
        # Lazy load heavy dependencies
        import fiftyone.zoo as foz
        from pycocotools.coco import COCO

        print(f"[COCOLoader] Loading FiftyOne COCO-2017 {self.split} split...")
        if self.root:
            print(f"[COCOLoader] Ignoring dataset root '{self.root}' and using FiftyOne dataset storage.")
        # Note: root is intentionally ignored; FiftyOne manages the dataset path.
        self.fo_dataset = foz.load_zoo_dataset("coco-2017", split=self.split)
        self.fo_dataset_name = self.fo_dataset.name
        self.default_label_field = _infer_fiftyone_label_field(self.fo_dataset)
        self.coco_classes = _infer_fiftyone_classes(self.fo_dataset, self.default_label_field)
        self.fo_sample_map = {
            int(os.path.splitext(os.path.basename(filepath))[0]): {
                "filepath": filepath,
                "width": metadata.width if metadata is not None else None,
                "height": metadata.height if metadata is not None else None,
            }
            for filepath, metadata in zip(
                self.fo_dataset.values("filepath"),
                self.fo_dataset.values("metadata"),
            )
        }
        
        ann_suffix = "train2017" if self.split == "train" else "val2017"
        self.ann_file = os.path.expanduser(f"~/fiftyone/coco-2017/raw/instances_{ann_suffix}.json")
        if not os.path.exists(self.ann_file):
            print(f"!!! [COCOLoader] Annotation file not found at {self.ann_file}. Evaluation might fail.")
        else:
            self.coco_gt = COCO(self.ann_file)
            self.category_id_to_name = {
                int(cat["id"]): str(cat["name"])
                for cat in self.coco_gt.loadCats(self.coco_gt.getCatIds())
            }

    def _category_name(self, category_id: int) -> str:
        if hasattr(self, "category_id_to_name") and category_id in self.category_id_to_name:
            return self.category_id_to_name[category_id]
        if 0 <= category_id < len(self.coco_classes):
            return str(self.coco_classes[category_id])
        return str(category_id)

    def _record_sample(self, image_id: int, width: int, height: int) -> None:
        sample_info = self.fo_sample_map.get(image_id, {})
        self.sample_records[image_id] = {
            "image_id": int(image_id),
            "filepath": sample_info.get("filepath"),
            "width": int(sample_info.get("width") or width),
            "height": int(sample_info.get("height") or height),
        }

    def _append_prediction(
        self,
        image_id: int,
        width: int,
        height: int,
        category_id: int,
        score: float,
        bbox_xywh: List[float],
    ) -> None:
        norm_bbox = [
            bbox_xywh[0] / max(width, 1),
            bbox_xywh[1] / max(height, 1),
            bbox_xywh[2] / max(width, 1),
            bbox_xywh[3] / max(height, 1),
        ]
        self.predictions_by_image.setdefault(image_id, []).append({
            "label": self._category_name(category_id),
            "category_id": int(category_id),
            "confidence": float(score),
            "bounding_box": [float(v) for v in norm_bbox],
            "bbox_abs_xywh": [float(v) for v in bbox_xywh],
        })

    def _write_detection_exports(self) -> None:
        if not self.log_dir:
            return

        export_dir = os.path.join(self.log_dir, "detections")
        os.makedirs(export_dir, exist_ok=True)

        coco_results_path = os.path.join(export_dir, "coco_results.json")
        with open(coco_results_path, "w") as f:
            json.dump(self.coco_results, f, indent=2)

        per_image_predictions = [
            {
                **self.sample_records[image_id],
                "predictions": self.predictions_by_image.get(image_id, []),
            }
            for image_id in sorted(self.sample_records)
        ]
        detections_payload = {
            "dataset_type": "coco2017",
            "split": self.split,
            "fiftyone_dataset_name": self.fo_dataset_name,
            "fiftyone_label_field": self.default_label_field,
            "annotation_file": self.ann_file,
            "processed_image_ids": sorted(self.sample_records),
            "num_images": len(self.sample_records),
            "num_detections": len(self.coco_results),
            "classes": self.coco_classes,
            "samples": per_image_predictions,
        }
        detections_path = os.path.join(export_dir, "fiftyone_predictions.json")
        with open(detections_path, "w") as f:
            json.dump(detections_payload, f, indent=2)

        self.export_info = {
            "coco_results_path": coco_results_path,
            "fiftyone_predictions_path": detections_path,
        }

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

        to_tensor = v2.ToImage()
        to_uint8 = v2.ToDtype(torch.uint8, scale=False)
        transform_steps = [to_tensor]
        if not self.preserve_original_resolution:
            transform_steps.append(v2.Resize((self.image_size, self.image_size), antialias=True))
        tfm = v2.Compose([*transform_steps, to_uint8])

        ds = FiftyOneTorchDataset(self.fo_dataset, transform=tfm)

        collate_fn = None
        if self.preserve_original_resolution:
            def collate_fn(batch):
                images, labels = zip(*batch)
                return list(images), torch.stack(labels, dim=0)

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
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
            self._record_sample(int(image_id), int(w), int(h))

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
                bbox_xywh = [float(abs_x1), float(abs_y1), float(abs_w), float(abs_h)]
                
                self.coco_results.append({
                    "image_id": int(image_id),
                    "category_id": int(cat_id),
                    "bbox": bbox_xywh,
                    "score": float(score)
                })
                self._append_prediction(
                    image_id=int(image_id),
                    width=int(w),
                    height=int(h),
                    category_id=int(cat_id),
                    score=float(score),
                    bbox_xywh=bbox_xywh,
                )

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
            self._write_detection_exports()
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
                "total_images": len(self.processed_ids),
                "detection_exports": dict(self.export_info),
            }
        except Exception as e:
            print(f"!!! [COCOLoader] Eval Failed: {e}")
            return {"error": str(e)}


def get_dataset_loader(name: str, root: Optional[str], batch_size: int, **kwargs) -> DatasetLoader:
    if name == 'imagenet-1k':
        return ImageNetLoader(root, batch_size, **kwargs)
    elif name == 'coco2017':
        return COCO2017Loader(root, batch_size, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
