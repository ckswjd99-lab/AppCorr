import torch
from .base import ModelExecutor

def get_model_executor(name: str, device: torch.device) -> ModelExecutor:
    if "dinov3_classifier" in name:
        from .dinov3_classifier import DINOv3ClassifierExecutor
        return DINOv3ClassifierExecutor(device)
    elif "dinov3_detector" in name:
        from .dinov3_detector import DINOv3DetectorExecutor
        return DINOv3DetectorExecutor(device)
    elif "dinov3_segmentor_linhead" in name:
        from .dinov3_segmentor_linhead import DINOv3SegmentorLinheadExecutor
        return DINOv3SegmentorLinheadExecutor(device)
    elif "dinov3_segmentor" in name:
        from .dinov3_segmentor import DINOv3SegmentorExecutor
        return DINOv3SegmentorExecutor(device)
    elif "dinov3_depther" in name:
        from .dinov3_depther import DINOv3DeptherExecutor
        return DINOv3DeptherExecutor(device)
    else:
        if "dinov3" in name:
             from .dinov3_classifier import DINOv3ClassifierExecutor
             return DINOv3ClassifierExecutor(device)
        raise ValueError(f"Unknown model executor for: {name}")
