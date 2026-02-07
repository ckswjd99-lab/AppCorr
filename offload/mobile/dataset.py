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
        if self.image_size == 256:
            val_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ])
        else:
            val_transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
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

    def get_summary(self) -> Dict[str, Any]:
        acc1 = self.total_top1 / self.total_samples * 100 if self.total_samples > 0 else 0.0
        acc5 = self.total_top5 / self.total_samples * 100 if self.total_samples > 0 else 0.0
        return {
            'total_samples': self.total_samples,
            'top1_acc': acc1,
            'top5_acc': acc5
        }

def get_dataset_loader(name: str, root: str, batch_size: int, **kwargs) -> DatasetLoader:
    if name == 'imagenet':
        return ImageNetLoader(root, batch_size, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
