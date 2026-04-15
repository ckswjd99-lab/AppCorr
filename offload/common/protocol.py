from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict
from enum import Enum, auto


def default_appcorr_kwargs() -> Dict[str, Any]:
    return {
        'enabled': False,
        'generated_from_client': False,
        'global_source_mode': 'final_correct',
        'correction_mode': 'exact',
        'update_attn': False,
        'pyramid_levels': [0],
        'token_res': [1.0],
        'plan': [],
        'num_groups': 1,
        'group_strategy': 'uniform',
        'token_keep_ratio': 0.2,
        'attn_col_alive_ratio': 1.0,
        'mobile_pscore': 'none',
        'mobile_pscore_weight': 0.0,
        'server_pscore': 'cls_attn_prob',
        'server_pscore_weight': 1.0,
        'token_prune_enabled': False,
        'token_prune_threshold': 0.0,
        'token_prune_min_keep': 1,
        'method': 'partial_token',
        'learned_correction_layers': [0],
        'learned_hidden_size': 512,
        'learned_bottleneck_size': 256,
        'learned_attn_mixer_layers': 1,
        'learned_dropout': 0.0,
        'learned_checkpoint_path': '',
        'learned_checkpoint_load_path': '',
        'learned_checkpoint_save_path': '',
        'learned_train': False,
        'learned_train_epochs': 1,
        'learned_train_steps_per_epoch': 0,
        'learned_train_lr': 1e-4,
        'learned_train_weight_decay': 1e-4,
        'learned_log_interval': 10,
        'learned_save_every': 1,
        'learned_loss_weight_dx': 1.0,
        'learned_loss_weight_attn': 0.1,
        'learned_loss_weight_ffn': 0.1,
        'learned_loss_weight_cosine': 0.0,
        'debug': False,
    }


def normalize_appcorr_kwargs(
    appcorr_kwargs: Dict[str, Any] | None = None,
    transmission_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    defaults = default_appcorr_kwargs()
    raw = dict(appcorr_kwargs or {})
    transmission = dict(transmission_kwargs or {})
    explicit_enabled = raw.pop('enabled', None)

    options = default_appcorr_kwargs()
    options.update(raw)
    for key in ('pyramid_levels', 'num_groups'):
        if key not in raw and transmission.get(key) is not None:
            options[key] = transmission[key]
    options['enabled'] = bool(raw) if explicit_enabled is None else bool(explicit_enabled)
    options['generated_from_client'] = bool(options.get('generated_from_client', defaults['generated_from_client']))
    options['global_source_mode'] = str(options.get('global_source_mode', defaults['global_source_mode']))
    if options['global_source_mode'] not in {'final_correct', 'approx'}:
        options['global_source_mode'] = defaults['global_source_mode']
    correction_mode = str(options.get('correction_mode', defaults['correction_mode']))
    if correction_mode not in {'exact', 'learned_block', 'none'}:
        correction_mode = defaults['correction_mode']
    options['correction_mode'] = correction_mode
    options['update_attn'] = bool(options.get('update_attn', defaults['update_attn']))
    options['pyramid_levels'] = list(options.get('pyramid_levels', defaults['pyramid_levels']))
    options['token_res'] = list(options.get('token_res', defaults['token_res']))
    options['plan'] = list(options.get('plan', defaults['plan']))
    options['num_groups'] = max(int(options.get('num_groups', defaults['num_groups'])), 1)
    options['group_strategy'] = str(options.get('group_strategy', defaults['group_strategy']))
    token_keep_ratio = options.get('token_keep_ratio', defaults['token_keep_ratio'])
    if 'token_keep_ratio' not in raw and 'cls_alive_ratio' in raw:
        token_keep_ratio = raw['cls_alive_ratio']
    options['token_keep_ratio'] = float(token_keep_ratio)
    options['attn_col_alive_ratio'] = float(options.get('attn_col_alive_ratio', defaults['attn_col_alive_ratio']))
    mobile_pscore = str(options.get('mobile_pscore', defaults['mobile_pscore']))
    if mobile_pscore in {'', 'null', 'None'}:
        mobile_pscore = defaults['mobile_pscore']
    options['mobile_pscore'] = mobile_pscore
    options['mobile_pscore_weight'] = float(options.get('mobile_pscore_weight', defaults['mobile_pscore_weight']))

    server_pscore = str(options.get('server_pscore', defaults['server_pscore']))
    legacy_token_prune_score = raw.get('token_prune_score')
    if legacy_token_prune_score is not None and 'server_pscore' not in raw:
        server_pscore = str(legacy_token_prune_score)
    if bool(raw.get('patch_attn_prune', False)) and 'server_pscore' not in raw and legacy_token_prune_score is None:
        server_pscore = 'patch_attn_prob'
    if server_pscore == 'patch_attn_prune':
        server_pscore = 'patch_attn_prob'
    if server_pscore not in {'cls_attn_prob', 'patch_attn_prob'}:
        server_pscore = defaults['server_pscore']
    options['server_pscore'] = server_pscore
    options['server_pscore_weight'] = float(options.get('server_pscore_weight', defaults['server_pscore_weight']))
    options['token_prune_enabled'] = bool(options.get('token_prune_enabled', defaults['token_prune_enabled']))
    options['token_prune_threshold'] = float(options.get('token_prune_threshold', defaults['token_prune_threshold']))
    options['token_prune_min_keep'] = max(int(options.get('token_prune_min_keep', defaults['token_prune_min_keep'])), 1)
    options['method'] = str(options.get('method', defaults['method']))
    options['learned_correction_layers'] = [
        int(layer_idx) for layer_idx in options.get('learned_correction_layers', defaults['learned_correction_layers'])
    ]
    options['learned_hidden_size'] = max(int(options.get('learned_hidden_size', defaults['learned_hidden_size'])), 1)
    options['learned_bottleneck_size'] = max(
        int(options.get('learned_bottleneck_size', defaults['learned_bottleneck_size'])),
        1,
    )
    options['learned_attn_mixer_layers'] = max(
        int(options.get('learned_attn_mixer_layers', defaults['learned_attn_mixer_layers'])),
        1,
    )
    options['learned_dropout'] = float(options.get('learned_dropout', defaults['learned_dropout']))
    options['learned_checkpoint_path'] = str(
        options.get('learned_checkpoint_path', defaults['learned_checkpoint_path'])
    )
    options['learned_checkpoint_load_path'] = str(
        options.get('learned_checkpoint_load_path', defaults['learned_checkpoint_load_path'])
    )
    options['learned_checkpoint_save_path'] = str(
        options.get('learned_checkpoint_save_path', defaults['learned_checkpoint_save_path'])
    )
    options['learned_train'] = bool(options.get('learned_train', defaults['learned_train']))
    options['learned_train_epochs'] = max(
        int(options.get('learned_train_epochs', defaults['learned_train_epochs'])),
        1,
    )
    options['learned_train_steps_per_epoch'] = max(
        int(options.get('learned_train_steps_per_epoch', defaults['learned_train_steps_per_epoch'])),
        0,
    )
    options['learned_train_lr'] = float(options.get('learned_train_lr', defaults['learned_train_lr']))
    options['learned_train_weight_decay'] = float(
        options.get('learned_train_weight_decay', defaults['learned_train_weight_decay'])
    )
    options['learned_log_interval'] = max(int(options.get('learned_log_interval', defaults['learned_log_interval'])), 1)
    options['learned_save_every'] = max(int(options.get('learned_save_every', defaults['learned_save_every'])), 1)
    options['learned_loss_weight_dx'] = float(
        options.get('learned_loss_weight_dx', defaults['learned_loss_weight_dx'])
    )
    options['learned_loss_weight_attn'] = float(
        options.get('learned_loss_weight_attn', defaults['learned_loss_weight_attn'])
    )
    options['learned_loss_weight_ffn'] = float(
        options.get('learned_loss_weight_ffn', defaults['learned_loss_weight_ffn'])
    )
    options['learned_loss_weight_cosine'] = float(
        options.get('learned_loss_weight_cosine', defaults['learned_loss_weight_cosine'])
    )
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

    def get_pyramid_resize_order(self) -> str:
        raw_order = self.transmission_kwargs.get('pyramid_resize_order')
        if raw_order is None and 'build_pyramid_before_resize' in self.transmission_kwargs:
            return 'pyramid_then_resize' if bool(self.transmission_kwargs['build_pyramid_before_resize']) else 'resize_then_pyramid'

        normalized = str(raw_order or 'resize_then_pyramid').strip().lower()
        aliases = {
            'resize_then_pyramid': 'resize_then_pyramid',
            'resize_first': 'resize_then_pyramid',
            'pyramid_then_resize': 'pyramid_then_resize',
            'pyramid_first': 'pyramid_then_resize',
            'raw_then_resize': 'pyramid_then_resize',
        }
        if normalized not in aliases:
            raise ValueError(
                "Unsupported transmission_kwargs.pyramid_resize_order="
                f"{raw_order!r}. Expected one of: resize_then_pyramid, pyramid_then_resize."
            )
        return aliases[normalized]

    def get_appcorr_options(self) -> Dict[str, Any]:
        return normalize_appcorr_kwargs(self.appcorr_kwargs, self.transmission_kwargs)

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
