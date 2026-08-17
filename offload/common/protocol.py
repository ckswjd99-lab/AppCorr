from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict
from enum import Enum, auto


def default_appcorr_kwargs() -> Dict[str, Any]:
    return {
        'enabled': False,
        'generated_from_client': False,
        'global_source_mode': 'global_first',
        'update_attn': False,
        'pyramid_levels': [0],
        'token_res': [1.0],
        'plan': [],
        'num_groups': 1,
        'group_strategy': 'uniform',
        'token_keep_ratio': 0.2,
        'token_keep_thres': None,
        'token_keep_cap': 0,
        'attn_col_alive_ratio': 1.0,
        'mobile_pscore': 'none',
        'mobile_pscore_weight': 0.0,
        'server_pscore': 'cls_attn_prob',
        'server_pscore_weight': 1.0,
        'pscore_fusion': 'add',
        'sdpa_query_bucket_size': 0,
        'sdpa_warmup': True,
        'sdpa_warmup_runs': 2,
        'correct_warmup_runs': 1,
        'token_prune_enabled': False,
        'token_prune_threshold': 0.0,
        'token_prune_min_keep': 1,
        'method': 'partial_token',
        'debug': False,
    }


def _inherit_shared_appcorr_kwargs(
    raw: Dict[str, Any],
    transmission_kwargs: Dict[str, Any] | None,
) -> None:
    if not transmission_kwargs:
        return

    if 'pyramid_levels' in transmission_kwargs:
        pyramid_levels = list(transmission_kwargs['pyramid_levels'])
        if 'pyramid_levels' in raw and list(raw['pyramid_levels']) != pyramid_levels:
            raise ValueError(
                "appcorr_kwargs.pyramid_levels must match transmission_kwargs.pyramid_levels"
            )
        raw['pyramid_levels'] = pyramid_levels

    if 'num_groups' in transmission_kwargs:
        num_groups = max(int(transmission_kwargs['num_groups']), 1)
        if 'num_groups' in raw and max(int(raw['num_groups']), 1) != num_groups:
            raise ValueError(
                "appcorr_kwargs.num_groups must match transmission_kwargs.num_groups"
            )
        raw['num_groups'] = num_groups


def normalize_appcorr_kwargs(
    appcorr_kwargs: Dict[str, Any] | None = None,
    transmission_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    defaults = default_appcorr_kwargs()
    raw = dict(appcorr_kwargs or {})
    explicit_enabled = raw.pop('enabled', None)
    enabled_from_appcorr = bool(raw)
    _inherit_shared_appcorr_kwargs(raw, transmission_kwargs)

    # Removed 2026-08-16: persisting the corrected increment into `blocks_out_sum` is
    # unconditional, because not doing it is the bug rather than a setting -- interleaved
    # correction then discards every round but the last. Raise instead of ignoring the key: a
    # stale `--set ...=false` would otherwise produce an "off" arm that is really an on arm, and
    # an A/B whose two halves are secretly the same condition is worse than one that fails.
    if 'persist_correction_residual' in raw:
        raise ValueError(
            "appcorr_kwargs.persist_correction_residual no longer exists; the corrected "
            "increment is always persisted. Drop the setting -- the pre-fix behaviour is not "
            "reproducible, and its measurements are recorded in "
            "docs/memo/dinov3_correct_low_precision_status.md."
        )

    options = default_appcorr_kwargs()
    options.update(raw)
    options['enabled'] = enabled_from_appcorr if explicit_enabled is None else bool(explicit_enabled)
    options['generated_from_client'] = bool(options.get('generated_from_client', defaults['generated_from_client']))
    options['global_source_mode'] = str(options.get('global_source_mode', defaults['global_source_mode']))
    if options['global_source_mode'] not in {'global_first', 'final_correct', 'approx'}:
        options['global_source_mode'] = defaults['global_source_mode']
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
    token_keep_thres = options.get('token_keep_thres', defaults['token_keep_thres'])
    if token_keep_thres in {'', 'null', 'None'}:
        token_keep_thres = None
    options['token_keep_thres'] = None if token_keep_thres is None else float(token_keep_thres)
    # >0 routes threshold selection through the sync-free fixed-width builder. The .item()/nonzero()
    # in the general builder stall the launch pipeline: 17.84 ms of a 77.5 ms FP4 correction pass is
    # GPU idle on that path (23%), against ~0% on the sync-free builders.
    options['token_keep_cap'] = max(0, int(options.get('token_keep_cap', 0) or 0))
    options['attn_col_alive_ratio'] = float(options.get('attn_col_alive_ratio', defaults['attn_col_alive_ratio']))
    mobile_pscore = str(options.get('mobile_pscore', defaults['mobile_pscore']))
    if mobile_pscore in {'', 'null', 'None'}:
        mobile_pscore = defaults['mobile_pscore']
    mobile_pscore_aliases = {
        'residual_rms': 'residual_rms',
        'patch_residual_rms': 'residual_rms',
        'residual_l2': 'residual_energy',
        'residual_l2_energy': 'residual_energy',
        'residual_energy': 'residual_energy',
        'patch_residual_l2': 'residual_energy',
        'patch_residual_energy': 'residual_energy',
    }
    mobile_pscore = mobile_pscore_aliases.get(mobile_pscore, mobile_pscore)
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
    server_pscore_aliases = {
        'pseudo_patch_attn_prob': 'patch_pseudo_attn_prob',
        'pseudo_patch_attn_prob_layermean': 'patch_pseudo_attn_prob_layermean',
    }
    server_pscore = server_pscore_aliases.get(server_pscore, server_pscore)
    legacy_server_pscore_layer_fusion = str(raw.get('server_pscore_layer_fusion', '')).lower()
    if legacy_server_pscore_layer_fusion in {'mean', 'avg', 'all_layer_mean', 'layer_mean', 'mean_all_layers'}:
        if server_pscore == 'patch_attn_prob':
            server_pscore = 'patch_attn_prob_layermean'
        elif server_pscore == 'patch_pseudo_attn_prob':
            server_pscore = 'patch_pseudo_attn_prob_layermean'
        elif server_pscore == 'cls_attn_prob':
            server_pscore = 'cls_attn_prob_layermean'
    # Single source of truth: the block owns the set, because it is the code that has to implement
    # each value. Duplicating the list here meant adding a score in one place and having the
    # scheduler reject it in the other -- which surfaces as `decide()` raising, no Task ever being
    # built, and the client sitting on a full patch buffer until its timeout. Nothing in either log
    # says why.
    from appcorr.models.dinov3.layers.block import SelfAttentionBlock

    valid_server_pscores = set(SelfAttentionBlock._VALID_SERVER_PSCORES)
    if server_pscore not in valid_server_pscores:
        raise ValueError(
            f"Unknown server_pscore '{server_pscore}'. "
            f"Available values: {sorted(valid_server_pscores)}"
        )
    options['server_pscore'] = server_pscore
    options['server_pscore_weight'] = float(options.get('server_pscore_weight', defaults['server_pscore_weight']))
    pscore_fusion = str(options.get('pscore_fusion', defaults['pscore_fusion'])).lower()
    if pscore_fusion in {'mul', 'product'}:
        pscore_fusion = 'multiply'
    elif pscore_fusion in {'geomean', 'geometric_mean'}:
        pscore_fusion = 'geo_mean'
    elif pscore_fusion not in {'add', 'multiply', 'geo_mean'}:
        pscore_fusion = defaults['pscore_fusion']
    options['pscore_fusion'] = pscore_fusion
    sdpa_query_bucket_size = int(options.get('sdpa_query_bucket_size', defaults['sdpa_query_bucket_size']) or 0)
    options['sdpa_query_bucket_size'] = max(sdpa_query_bucket_size, 0)
    options['sdpa_warmup'] = bool(options.get('sdpa_warmup', defaults['sdpa_warmup']))
    options['sdpa_warmup_runs'] = max(int(options.get('sdpa_warmup_runs', defaults['sdpa_warmup_runs']) or 0), 0)
    options['correct_warmup_runs'] = max(int(options.get('correct_warmup_runs', defaults['correct_warmup_runs']) or 0), 0)
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
    model_name: str = "dinov3_classifier"  # e.g. "dinov3_segmentor_m2f", "dinov3_segmentor_linhead"
    device: str = None  # User can specify "cuda:0", "cpu", etc. Default is None (auto-detect)
    precision: str = "bf16"
    fp8_auto_min_rows: int = 3072
    correct_precision: str = "bf16"
    correct_compile: bool = False
    # correct_precision=fp4 only. >0 runs that many correction events through torchao's observer
    # (numerically exact BF16, recording activation amax) and then bakes a static per-tensor scale,
    # removing the per-call amax scan. 0 keeps the dynamic per-call scale.
    correct_fp4_calib_events: int = 1
    # correct_precision=fp4 only. attn.proj is the worst FP4 candidate of the five correction
    # Linears -- its input is the attention-core output (least-compressible delta) and it is the one
    # input no producer fusion can reach -- so it runs FP8 by default. Set "fp4" to force it.
    correct_fp4_proj_precision: str = "fp8"
    # correct_precision=fp4 only. NVFP4 carries a mandatory 16-element block scale and an optional
    # second, per-tensor one. The per-tensor scale halves the weight-reconstruction error (rel-L2
    # 0.0951 vs 0.1305 on a 4096x4096 draw) but costs an [M, N] epilogue: with an output scale
    # pending, `_scaled_mm` cannot take the bias, so scale and bias need their own kernel. Dropping
    # it lets the bias ride along in the GEMM and the epilogue disappear.
    #
    # Default off, from ADE20K m2f full-2000: 61.4208 mIoU without it against 61.4243 with -- a
    # 0.0035 difference on a 6.223 floor-to-ceiling gap -- while CORRECT_FORWARD went 98.25 -> 89.20
    # ms (-9.2%, with APPROX_FORWARD steady at 184.26 -> 184.08 as the control). The weight error is
    # genuinely larger; it just does not reach the task metric.
    correct_fp4_per_tensor_scale: bool = False
    # Round the correction GEMMs' row count M up to a multiple of this, zero-padding. M changes every
    # correction round, so without it every shape-specialised consumer -- torch.compile graphs, and
    # CUDA graph capture in particular -- sees an unbounded set of shapes. Bucketing is a *cost* on
    # its own (~19% more rows at M=1027, bucket 256); it only pays once something consumes the fixed
    # shapes. 0 disables.
    correct_bucket_rows: int = 0

    # Dataset Settings
    dataset_name: str = "imagenet-1k"
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Batch Settings
    batch_size: int = 32
    
    # Image/Patch Specs
    image_shape: Tuple[int, int, int] = (256, 256, 3)
    patch_size: Tuple[int, int] = (16, 16)
    input_profile_name: str = "fixed_image_shape"
    input_profile_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Policies
    scheduler_policy_name: str = "BatchCountBased"
    transmission_policy_name: str = "Raw"
    
    # Dynamic arguments
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    transmission_kwargs: Dict[str, Any] = field(default_factory=dict)
    appcorr_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.precision = str(self.precision).lower()
        if self.precision not in {"bf16", "fp8", "fp4", "auto"}:
            raise ValueError(
                "precision must be one of 'bf16', 'fp8', 'fp4', or 'auto', "
                f"got {self.precision!r}"
            )
        self.fp8_auto_min_rows = int(self.fp8_auto_min_rows)
        if self.fp8_auto_min_rows <= 0:
            raise ValueError(
                "fp8_auto_min_rows must be positive, "
                f"got {self.fp8_auto_min_rows}"
            )
        self.correct_precision = str(self.correct_precision).lower()
        if self.correct_precision not in {"bf16", "fp8", "fp4"}:
            raise ValueError(
                "correct_precision must be one of 'bf16', 'fp8', or 'fp4', "
                f"got {self.correct_precision!r}"
            )
        self.correct_compile = bool(self.correct_compile)
        self.correct_fp4_calib_events = max(0, int(self.correct_fp4_calib_events))
        self.correct_bucket_rows = max(0, int(self.correct_bucket_rows))
        self.correct_fp4_per_tensor_scale = bool(self.correct_fp4_per_tensor_scale)
        if self.correct_fp4_proj_precision not in {"fp4", "fp8"}:
            raise ValueError(
                "correct_fp4_proj_precision must be 'fp4' or 'fp8', "
                f"got {self.correct_fp4_proj_precision!r}"
            )

    def get_input_profile_config(self) -> Dict[str, Any]:
        name = self.input_profile_name or "fixed_image_shape"
        if name == "fixed_image_shape":
            return {"name": name}
        if name == "dinov3_nyu_synthmix_dpt":
            options = {
                "name": name,
                "depther_eval_size": 768,
                "depther_use_tta": True,
                "autocast_dtype": "bfloat16",
            }
            options.update(self.input_profile_kwargs)
            return options
        if name == "dinov3_ade20k_m2f_official":
            options = {
                "name": name,
                "mobile_resize_short_side": 896,
                "server_inference_mode": "slide",
                "server_crop_size": 896,
                "server_stride": 596,
                "server_eval_mode": "tta",
                "server_rescale_to": "input",
                "server_use_tta": True,
                "server_tta_ratios": [0.9, 0.95, 1.0, 1.05, 1.1],
                "decoder_head_type": "m2f",
                "num_classes": 150,
                "autocast_dtype": "bfloat16",
                "reduce_zero_label": True,
            }
            options.update(self.input_profile_kwargs)
            return options
        if name == "dinov3_ade20k_linhead_official":
            options = {
                "name": name,
                "mobile_resize_short_side": 512,
                "server_inference_mode": "slide",
                "server_crop_size": 512,
                "server_stride": 341,
                "server_eval_mode": "single",
                "server_rescale_to": "input",
                "decoder_head_type": "linear",
                "num_classes": 150,
                "autocast_dtype": "float32",
                "reduce_zero_label": True,
            }
            options.update(self.input_profile_kwargs)
            return options
        if name == "vggt_omega_512":
            options = {
                "name": name,
                # VGGT derives each frame's canvas from that frame's own aspect ratio instead of
                # using one fixed shape, so there is deliberately no `mobile_resize_short_side`:
                # `vggt_resolution` is a token *budget* ((res/patch)**2 tokens), not a side length.
                "vggt_resolution": 512,
                "vggt_patch_size": 16,
                "vggt_resize_mode": "balanced",
                "vggt_weights_path": "~/cjpark/weights/vggt/vggt_omega_1b_512.pt",
                "autocast_dtype": "bfloat16",
            }
            options.update(self.input_profile_kwargs)
            return options
        raise ValueError(f"Unknown input_profile_name: {name}")

    def use_official_ade20k_m2f_profile(self) -> bool:
        return (self.input_profile_name or "fixed_image_shape") == "dinov3_ade20k_m2f_official"

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
    pscore_hint: float = 0.0
    target_shape: tuple = ()
    # Number of correction groups for this image (crop-cover policy: = sliding-crop count,
    # which varies per image). 0 = unset / not applicable.
    num_correction_groups: int = 0

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
    token_pscore_kept_mass: float = 0.0
    token_pscore_full_mass: float = 0.0
    partial_token_kept_patch: float = 0.0
    partial_token_full_patch: float = 0.0
    partial_token_sample_count: float = 0.0
