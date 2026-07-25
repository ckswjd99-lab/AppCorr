"""
pi0fast_libero_partial_token_eval.py

DINOv3-style progressive pi0-FAST paths evaluated in the official lerobot-eval harness. This patches
PI0FastPolicy.predict_action_chunk and supports:

* `partial`: partial-token correct on both SigLIP and Gemma:
  - vision: SigLIP base approx, rank by hidden residual, L2-to-L0 visual residual energy times
            SigLIP or language-query Gemma attention, or an optional Gemma-attention fusion, then
            correct top-`keep` patches/image;
  - LLM:    bidirectional approx on the base vision features + text -> correct ONLY the selected
            vision tokens' K/V + the text permanent group (non-selected vision keep base LLM K/V) ->
            FAST-decode. At keep=1.0 this reproduces stock lerobot exactly (verified |diff|=0).
* `block_causal`: no token pruning. SigLIP cumulatively corrects four spatial arrival groups. Gemma
  prefills each new vision group exactly once with bidirectional intra-group attention and only
  earlier/current vision keys, then prefills language as the final bidirectional block. This is an
  intentional approximation of stock PaliGemma's globally bidirectional prefix.

Stock, exact-capable partial-token, and approximate block-causal modes therefore run in one harness.

Env:
  PTC_MODE: partial (default), block_causal, or stock.
  PTC_PRECISION: float32 (default parity path) or amp_bf16 (official mixed-precision diagnostic).
  PTC_COMPARE_STOCK: 1 compares stock and partial action chunks on every partial rollout input.
  PTC_DUMP_MISMATCH: optional .pt path for the first mismatching preprocessed batch.
  PTC_STOP_ON_MISMATCH: 1 stops immediately after writing/printing the first mismatch.
  PTC_SCORE_MODE: vit (default), visual_residual_attn, visual_residual_llm_language,
                  vit_llm_vision, or vit_llm_language.
  PTC_LLM_VISION_WEIGHT: geometric-fusion exponent for LLM attention to vision keys (default 1.0).
  PTC_DUMP_PSCORE: optional .pt path for first-call score components and selected indices.
  PTC_SELECTION_VIDEO: 1 (default) writes per-episode MP4s with selected patches highlighted.
  PTC_NUM_GROUPS: number of block-causal spatial arrival groups (default 4).
  TASK_ID (0), N_EP (5), PTC_KEEP (0.5), PTC_BASE (4), PTC_CORRECT_TEXT (1).
  arg1 = output dir.
Run:
    TASK_ID=0 N_EP=10 PTC_MODE=partial PTC_PRECISION=float32 PTC_KEEP=0.5 \
        python analysis/experiments/pi0fast_libero_partial_token_eval.py /tmp/out_ptc50

The shared pi0fast_libero_runtime bootstrap supplies this host's LIBERO path, EGL settings, and
TORCHDYNAMO_DISABLE default. Set LIBERO_ROOT or the individual environment variables to override.
"""

import os
import sys
from pathlib import Path

APPCORR_ROOT = Path(__file__).resolve().parents[2]
if str(APPCORR_ROOT) not in sys.path:
    sys.path.insert(0, str(APPCORR_ROOT))

from analysis.experiments.pi0fast_libero_runtime import (
    configure_pi0fast_libero_runtime,
)
from analysis.experiments.pi0fast_selection_video import (
    write_selection_videos,
)

configure_pi0fast_libero_runtime()

import torch

from appcorr.models.pi0fast.progressive_model import (
    configure_policy_precision,
    install_gemma_scaling_fix,
)

install_gemma_scaling_fix()

MODE = os.environ.get("PTC_MODE", "partial")
PRECISION = os.environ.get("PTC_PRECISION", "float32")
KEEP = float(os.environ.get("PTC_KEEP", "0.5"))
BASE_FACTOR = int(os.environ.get("PTC_BASE", "4"))
NUM_GROUPS = int(os.environ.get("PTC_NUM_GROUPS", "4"))
CORRECT_TEXT = os.environ.get("PTC_CORRECT_TEXT", "1") not in {"0", "false", "False"}
COMPARE_STOCK = os.environ.get("PTC_COMPARE_STOCK", "0") in {"1", "true", "True"}
DUMP_MISMATCH = os.environ.get("PTC_DUMP_MISMATCH")
STOP_ON_MISMATCH = os.environ.get("PTC_STOP_ON_MISMATCH", "0") in {"1", "true", "True"}
SCORE_MODE = os.environ.get("PTC_SCORE_MODE", "vit")
LLM_VISION_WEIGHT = float(os.environ.get("PTC_LLM_VISION_WEIGHT", "1.0"))
DUMP_PSCORE = os.environ.get("PTC_DUMP_PSCORE")
SELECTION_VIDEO = os.environ.get("PTC_SELECTION_VIDEO", "1") not in {
    "0",
    "false",
    "False",
}
_SELECTION_TRACE = []
_SELECTION_ACTION_HORIZON = None

if MODE not in {"stock", "partial", "block_causal"}:
    raise ValueError(
        "PTC_MODE must be 'stock', 'partial', or "
        f"'block_causal', got {MODE!r}"
    )
if PRECISION not in {"float32", "amp_bf16"}:
    raise ValueError(
        f"PTC_PRECISION must be 'float32' or 'amp_bf16', got {PRECISION!r}"
    )
if SCORE_MODE not in {
    "vit",
    "visual_residual_attn",
    "visual_residual_llm_language",
    "vit_llm_vision",
    "vit_llm_language",
}:
    raise ValueError(
        "PTC_SCORE_MODE must be 'vit', 'visual_residual_attn', "
        "'visual_residual_llm_language', 'vit_llm_vision', or "
        f"'vit_llm_language', got {SCORE_MODE!r}"
    )
if LLM_VISION_WEIGHT < 0:
    raise ValueError("PTC_LLM_VISION_WEIGHT must be non-negative")
if NUM_GROUPS < 1:
    raise ValueError("PTC_NUM_GROUPS must be positive")

import lerobot.policies.pi0_fast.modeling_pi0_fast as MOD

_ORIGINAL_PREDICT_ACTION_CHUNK = MOD.PI0FastPolicy.predict_action_chunk
_MODEL_PRECISION = "float32" if PRECISION == "float32" else "inherit"


def _ensure_precision(policy):
    if getattr(policy, "_appcorr_precision", None) != _MODEL_PRECISION:
        configure_policy_precision(policy, _MODEL_PRECISION)


def _selection_trace_item(orch):
    images = []
    selected = []
    for image, image_mask, components in zip(
        orch.last_input_images,
        orch.last_input_image_masks,
        orch.last_pscore_components["per_image"],
    ):
        if not bool(image_mask.any().item()) or components is None:
            continue
        pixels = image[0].detach().float().cpu()
        if pixels.min().item() < 0:
            pixels = (pixels + 1.0) / 2.0
        pixels = (
            pixels.clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy()
        )
        images.append(pixels)
        selected.append(components["selected"].detach().cpu().tolist())
    return {
        "images": images,
        "selected": selected,
        "score_mode": orch.last_pscore_components["mode"],
    }


def _ptc_predict(self, batch, **kwargs):
    global _SELECTION_ACTION_HORIZON
    if not hasattr(self, "_orch"):
        from appcorr.models.pi0fast.progressive_model import Pi0FastProgressiveModel
        self._orch = Pi0FastProgressiveModel.from_policy(
            self,
            next(self.parameters()).device,
            precision=_MODEL_PRECISION,
        )
        self._orch.capture_selection_frames = (
            MODE == "partial" and SELECTION_VIDEO
        )
        _SELECTION_ACTION_HORIZON = int(self.config.n_action_steps)
        self._ptc_compare_count = 0
    reference = None
    if COMPARE_STOCK:
        reference = _ORIGINAL_PREDICT_ACTION_CHUNK(self, batch, **kwargs)
    if MODE == "block_causal":
        result = self._orch._block_causal_from_batch(
            batch,
            num_groups=NUM_GROUPS,
            base_factor=BASE_FACTOR,
        )
    else:
        result = self._orch._partial_from_batch(
            batch,
            keep=KEEP,
            base_factor=BASE_FACTOR,
            correct_text=CORRECT_TEXT,
            score_mode=SCORE_MODE,
            llm_vision_weight=LLM_VISION_WEIGHT,
        )
        if SELECTION_VIDEO:
            _SELECTION_TRACE.append(_selection_trace_item(self._orch))
    if not hasattr(self, "_ptc_stats_printed"):
        stats = self._orch.last_recompute_stats
        if MODE == "block_causal":
            print(
                "[recompute] "
                f"path=block_causal groups={stats['num_groups']} "
                f"ViT_queries={stats['vit_corrected_query_tokens']} "
                f"ViT_unique={stats['vit_unique_real_tokens']} "
                f"LLM_vision={stats['llm_prefilled_vision_tokens']} "
                f"LLM_language={stats['llm_prefilled_language_tokens']} "
                f"LLM_total={stats['llm_prefilled_tokens']}/"
                f"{stats['llm_valid_prefix_tokens']} "
                f"block_sizes={stats['vision_block_sizes']}",
                flush=True,
            )
        else:
            vit_rate = stats["vit_corrected_tokens"] / stats["vit_total_real_tokens"]
            llm_rate = stats["llm_corrected_query_tokens"] / stats["llm_prefix_tokens"]
            print(
                "[recompute] "
                f"ViT={stats['vit_corrected_tokens']}/"
                f"{stats['vit_total_real_tokens']} ({vit_rate:.1%}) "
                f"LLM_queries={stats['llm_corrected_query_tokens']}/"
                f"{stats['llm_prefix_tokens']} ({llm_rate:.1%}) "
                f"score_QK={stats['llm_attention_score_query_tokens']}x"
                f"{stats['llm_attention_score_key_tokens']} "
                f"query_group={stats['llm_attention_query_group']} "
                f"text_group={stats['text_group_tokens']} "
                f"valid_prefix={stats['llm_valid_prefix_tokens']}",
                flush=True,
            )
        self._ptc_stats_printed = True
    if MODE == "partial" and DUMP_PSCORE and not hasattr(self, "_ptc_pscore_dumped"):
        dump_parent = os.path.dirname(DUMP_PSCORE)
        if dump_parent:
            os.makedirs(dump_parent, exist_ok=True)

        def to_cpu(value):
            if torch.is_tensor(value):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {key: to_cpu(item) for key, item in value.items()}
            if isinstance(value, list):
                return [to_cpu(item) for item in value]
            return value

        torch.save(to_cpu(self._orch.last_pscore_components), DUMP_PSCORE)
        self._ptc_pscore_dumped = True
        print(f"[pscore] saved first-call components to {DUMP_PSCORE}", flush=True)
    if reference is not None:
        diff = (reference.float() - result.float()).abs()
        exact = bool(torch.equal(reference, result))
        print(
            f"[parity] chunk={self._ptc_compare_count} exact={exact} "
            f"max_diff={diff.max().item():.8g} mean_diff={diff.mean().item():.8g}",
            flush=True,
        )
        if not exact and DUMP_MISMATCH and not hasattr(self, "_ptc_mismatch_dumped"):
            dump_parent = os.path.dirname(DUMP_MISMATCH)
            if dump_parent:
                os.makedirs(dump_parent, exist_ok=True)
            cpu_batch = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            torch.save(cpu_batch, DUMP_MISMATCH)
            self._ptc_mismatch_dumped = True
            print(f"[parity] saved mismatching batch to {DUMP_MISMATCH}", flush=True)
        self._ptc_compare_count += 1
        if not exact and STOP_ON_MISMATCH:
            raise RuntimeError("Stopped at first stock/partial parity mismatch")
    return result


def _stock_predict(self, batch, **kwargs):
    _ensure_precision(self)
    return _ORIGINAL_PREDICT_ACTION_CHUNK(self, batch, **kwargs)


if MODE in {"partial", "block_causal"}:
    MOD.PI0FastPolicy.predict_action_chunk = _ptc_predict
    if MODE == "block_causal":
        print(
            "[eval] BLOCK-CAUSAL (progressive ViT, one-pass LLM): "
            f"groups={NUM_GROUPS} base={BASE_FACTOR} precision={PRECISION} "
            "token_pruning=False",
            flush=True,
        )
    else:
        print(
            "[eval] PARTIAL-TOKEN (ViT + LLM): "
            f"keep={KEEP} base={BASE_FACTOR} correct_text={CORRECT_TEXT} "
            f"precision={PRECISION} score_mode={SCORE_MODE} "
            f"llm_vision_weight={LLM_VISION_WEIGHT}",
            flush=True,
        )
else:
    MOD.PI0FastPolicy.predict_action_chunk = _stock_predict
    print(f"[eval] STOCK: precision={PRECISION}", flush=True)

from lerobot.scripts.lerobot_eval import eval_main

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pi0fast_ptc_out"
sys.argv = [
    "lerobot-eval",
    "--policy.path=lerobot/pi0fast-libero",
    "--policy.device=cuda",
    f"--policy.use_amp={'true' if PRECISION == 'amp_bf16' else 'false'}",
    "--env.type=libero",
    "--env.task=libero_spatial",
    f"--env.task_ids=[{os.environ.get('TASK_ID', '0')}]",
    "--eval.batch_size=1",
    f"--eval.n_episodes={os.environ.get('N_EP', '5')}",
    f"--output_dir={OUTPUT_DIR}",
]
eval_main()
if MODE == "partial" and SELECTION_VIDEO:
    write_selection_videos(
        OUTPUT_DIR,
        _SELECTION_TRACE,
        _SELECTION_ACTION_HORIZON,
    )
