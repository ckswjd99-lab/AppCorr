from typing import Any
import torch
from .base import ModelExecutor
from .dinov3_classifier import DINOv3ClassifierExecutor
from .dinov3_detector import DINOv3DetectorExecutor
from .dinov3_segmentor import DINOv3SegmentorExecutor

def get_model_executor(name: str, device: torch.device) -> ModelExecutor:
    if "dinov3_classifier" in name:
        return DINOv3ClassifierExecutor(device)
    elif "dinov3_detector" in name:
        return DINOv3DetectorExecutor(device)
    elif "dinov3_segmentor" in name:
        return DINOv3SegmentorExecutor(device)
    else:
        if "dinov3" in name:
             return DINOv3ClassifierExecutor(device)
        raise ValueError(f"Unknown model executor for: {name}")
