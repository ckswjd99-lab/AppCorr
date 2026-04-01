from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict
from enum import Enum, auto


def default_appcorr_kwargs() -> Dict[str, Any]:
    return {
        'enabled': False,
        'generated_from_client': False,
        'global_source_mode': 'final_correct',
        'update_attn': False,
        'pyramid_levels': [0],
        'token_res': [1.0],
        'plan': [],
        'num_groups': 1,
        'group_strategy': 'uniform',
        'cls_alive_ratio': 0.2,
        'attn_col_alive_ratio': 1.0,
        'token_prune_enabled': False,
        'token_prune_threshold': 0.0,
        'token_prune_min_keep': 1,
        'method': 'partial_token',
        'debug': False,
    }


def normalize_appcorr_kwargs(appcorr_kwargs: Dict[str, Any] | None = None) -> Dict[str, Any]:
    defaults = default_appcorr_kwargs()
    raw = dict(appcorr_kwargs or {})
    explicit_enabled = raw.pop('enabled', None)

    options = default_appcorr_kwargs()
    options.update(raw)
    options['enabled'] = bool(raw) if explicit_enabled is None else bool(explicit_enabled)
    options['generated_from_client'] = bool(options.get('generated_from_client', defaults['generated_from_client']))
    options['global_source_mode'] = str(options.get('global_source_mode', defaults['global_source_mode']))
    if options['global_source_mode'] not in {'final_correct', 'approx'}:
        options['global_source_mode'] = defaults['global_source_mode']
    options['update_attn'] = bool(options.get('update_attn', defaults['update_attn']))
    options['pyramid_levels'] = list(options.get('pyramid_levels', defaults['pyramid_levels']))
    options['token_res'] = list(options.get('token_res', defaults['token_res']))
    options['plan'] = list(options.get('plan', defaults['plan']))
    options['num_groups'] = max(int(options.get('num_groups', defaults['num_groups'])), 1)
    options['group_strategy'] = str(options.get('group_strategy', defaults['group_strategy']))
    options['cls_alive_ratio'] = float(options.get('cls_alive_ratio', defaults['cls_alive_ratio']))
    options['attn_col_alive_ratio'] = float(options.get('attn_col_alive_ratio', defaults['attn_col_alive_ratio']))
    options['token_prune_enabled'] = bool(options.get('token_prune_enabled', defaults['token_prune_enabled']))
    options['token_prune_threshold'] = float(options.get('token_prune_threshold', defaults['token_prune_threshold']))
    options['token_prune_min_keep'] = max(int(options.get('token_prune_min_keep', defaults['token_prune_min_keep'])), 1)
    options['method'] = str(options.get('method', defaults['method']))
    options['debug'] = bool(options.get('debug', defaults['debug']))
    return options


@dataclass
class ExperimentConfig:
    """Experiment settings."""
    exp_id: str = "exp"
    
    # Model Settings
    model_name: str = "dinov3_classifier"  # "dinov3_classifier", etc.
    device: str = None  # User can specify "cuda:0", "cpu", etc. Default is None (auto-detect)
    
    # Dataset Settings
    dataset_name: str = "imagenet-1k"
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Batch Settings
    batch_size: int = 32
    
    # Image/Patch Specs
    image_shape: Tuple[int, int, int] = (256, 256, 3)
    patch_size: Tuple[int, int] = (16, 16)
    
    # Policies
    scheduler_policy_name: str = "BatchCountBased"
    transmission_policy_name: str = "Raw"
    
    # Dynamic arguments
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    transmission_kwargs: Dict[str, Any] = field(default_factory=dict)
    appcorr_kwargs: Dict[str, Any] = field(default_factory=default_appcorr_kwargs)

    def early_exit_enabled(self) -> bool:
        return bool(self.scheduler_kwargs.get('early_exit', False))

    def get_early_exit_config(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self.scheduler_kwargs.items()
            if key in {'metric', 'threshold'}
        }

    def lowres_sr_enabled(self) -> bool:
        return bool(self.scheduler_kwargs.get('lowres_sr', False))

    def get_lowres_sr_config(self) -> Dict[str, Any]:
        return {
            'model': self.scheduler_kwargs.get('lowres_sr_model', 'realesrgan_x4plus'),
            'dtype': self.scheduler_kwargs.get('lowres_sr_dtype', 'fp16'),
            'weights_dir': self.scheduler_kwargs.get('lowres_sr_weights_dir', '~/cjpark/weights/realesrgan'),
            'tile': self.scheduler_kwargs.get('lowres_sr_tile', 0),
            'tile_pad': self.scheduler_kwargs.get('lowres_sr_tile_pad', 10),
            'pre_pad': self.scheduler_kwargs.get('lowres_sr_pre_pad', 0),
        }

@dataclass
class Patch:
    image_idx: int
    spatial_idx: int
    data: bytes
    
    res_level: int = 0
    group_id: int = 0
    batch_group_total: int = 0
    arrival_time: float = 0.0

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
    cache_size_bytes: int = 0
    cache_breakdown_bytes: Dict[str, int] = field(default_factory=dict)
    attn_prob_mass_used: float = 0.0
    attn_prob_mass_full: float = 0.0
    token_prune_kept_patch: float = 0.0
    token_prune_full_patch: float = 0.0
    token_prune_kept_residual_mass: float = 0.0
    token_prune_full_residual_mass: float = 0.0
