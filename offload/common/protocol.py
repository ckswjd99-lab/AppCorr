from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict
from enum import Enum, auto


@dataclass
class ExperimentConfig:
    """Experiment settings."""
    exp_id: str = "exp"
    
    # Model Settings
    model_name: str = "resnet18"  # "resnet18", "dinov3_custom", etc.
    
    # Batch Settings
    batch_size: int = 32
    
    # Image/Patch Specs
    image_shape: Tuple[int, int, int] = (256, 256, 3)
    patch_size: Tuple[int, int] = (16, 16)
    
    # Policies
    scheduler_policy_name: str = "BatchCountBased"
    transmission_policy_name: str = "Raw"
    
    # Dynamic arguments
    transmission_kwargs: Dict[str, Any] = field(default_factory=dict)
    appcorr_kwargs: Dict[str, Any] = field(default_factory=dict)
    early_exit_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Patch:
    image_idx: int
    spatial_idx: int
    data: bytes
    
    res_level: int = 0
    group_id: int = 0
    batch_group_total: int = 0

class OpType(Enum):
    # --- Computation Ops ---
    FULL_INFERENCE = auto()   
    APPROX_FORWARD = auto()   
    CORRECT_FORWARD = auto()  
    HEAD_INFERENCE = auto()   
    
    # --- Control Ops ---
    LOAD_INPUT = auto()
    PREPARE_TOKENS = auto()
    SEND_RESPONSE = auto()
    FREE_SESSION = auto()
    TIME_SYNC = auto()
    DECIDE_EXIT = auto()
    EXIT_ALL = auto()

@dataclass
class Instruction:
    op_type: OpType
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Task:
    task_id: int
    request_id: int
    payload: List['Patch']
    instructions: List[Instruction]

@dataclass
class InferenceResult:
    """Final result sent from Server to Mobile."""
    task_id: int
    timestamp: float
    output: Any
    server_events: List[Dict[str, Any]] = field(default_factory=list)