import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Any

@dataclass
class ExperimentConfig:
    """Experiment settings."""
    # Batch Settings
    batch_size: int = 32        # Number of images per request (e.g., 32)
    patches_per_image: int = 256 # Total patches per image (16x16 grid)
    
    # Image Settings
    image_shape: Tuple[int, int, int] = (256, 256, 3)
    patch_size: Tuple[int, int] = (16, 16)
    
    # Policy Names
    scheduler_policy_name: str = "BatchCountBased"
    transmission_policy_name: str = "BatchRaw"

@dataclass
class Patch:
    """
    Unit of data.
    - image_idx: Index of the image within the batch (0 ~ batch_size-1)
    - spatial_idx: Index of the patch within the image (0 ~ patches_per_image-1)
    """
    image_idx: int
    spatial_idx: int
    data: bytes

@dataclass
class Task:
    """
    Instruction + Payload for the Worker.
    Now carries the actual patch objects to allow sparse processing.
    """
    task_id: int
    mode: str               # 'APPROX' or 'CORRECT'
    payload: List[Patch]    # The actual patches to process
    layer_range: Tuple[int, int]

@dataclass
class InferenceResult:
    """Final result sent from Server to Mobile."""
    task_id: int
    timestamp: float
    output: Any             # Expecting (Batch_Size, ...) format