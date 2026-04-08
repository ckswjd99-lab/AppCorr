import torch
from .base import ModelExecutor

def get_model_executor(name: str, device: torch.device) -> ModelExecutor:
    if "dinov3_classifier" in name:
        from .dinov3_classifier import DINOv3ClassifierExecutor
        return DINOv3ClassifierExecutor(device)
    elif "dinov3_detector" in name:
        from .dinov3_detector import DINOv3DetectorExecutor
        return DINOv3DetectorExecutor(device)
    else:
        if "dinov3" in name:
             from .dinov3_classifier import DINOv3ClassifierExecutor
             return DINOv3ClassifierExecutor(device)
        raise ValueError(f"Unknown model executor for: {name}")
