import contextlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import torch
import numpy as np
from offload.common import Task

class ModelExecutor(ABC):
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self._dinov3_approx_precision = None
        self._dinov3_correct_precision = None

    def configure_dinov3_approx_precision(self, backbone: torch.nn.Module, config: Any):
        from .dinov3_precision import DINOv3ApproxPrecisionController

        self._dinov3_approx_precision = DINOv3ApproxPrecisionController.from_config(
            backbone.blocks,
            config,
            self.device,
        )

    def begin_dinov3_approx_event(self):
        if self._dinov3_approx_precision is None:
            raise RuntimeError("DINOv3 approximate precision is not configured")
        self._dinov3_approx_precision.begin_event()

    def run_dinov3_approx_block(
        self,
        layer_idx: int,
        x: torch.Tensor,
        rope,
        cache: Dict[str, Any],
        tag: str,
        *,
        source_key: str = "default",
        **kwargs,
    ):
        if self._dinov3_approx_precision is None:
            raise RuntimeError("DINOv3 approximate precision is not configured")
        return self._dinov3_approx_precision.run_block(
            layer_idx,
            x,
            rope,
            cache,
            tag,
            source_key=source_key,
            **kwargs,
        )

    def dinov3_full_inference_precision(self):
        """Context manager: run a stock (FULL_INFERENCE) backbone forward at the configured precision.

        FULL_INFERENCE does not dispatch per block through `run_dinov3_approx_block` -- the m2f and
        detector executors inline their own block loops and the depther calls the whole model in one
        go -- so without this it silently ignores `precision` and runs BF16. That is what made every
        approx-only L0 FP4 config return results bit-identical to its BF16 twin. No-op when the
        precision is bf16/auto, or when no controller is configured.
        """
        if self._dinov3_approx_precision is None:
            return contextlib.nullcontext()
        return self._dinov3_approx_precision.full_inference_blocks()

    def dinov3_approx_event_metadata(self) -> Dict[str, Any]:
        if self._dinov3_approx_precision is None:
            return {}
        return self._dinov3_approx_precision.event_metadata()

    def configure_dinov3_correct_precision(self, backbone: torch.nn.Module, config: Any):
        from .dinov3_precision import DINOv3CorrectPrecisionController
        from offload.common.protocol import normalize_appcorr_kwargs

        self._dinov3_correct_precision = DINOv3CorrectPrecisionController.from_config(
            backbone.blocks,
            config,
            self.device,
        )
        # Captured here rather than passed at every call site: the five DINOv3 executors spread
        # ~10 `run_dinov3_correct_block` calls between them, and a keyword added to some of them is
        # a silent partial fix. Injected once in `run_dinov3_correct_block` below.
        self._persist_correction_residual = bool(
            normalize_appcorr_kwargs(
                getattr(config, "appcorr_kwargs", None),
                getattr(config, "transmission_kwargs", None),
            )["persist_correction_residual"]
        )

    def begin_dinov3_correct_event(self):
        if self._dinov3_correct_precision is None:
            raise RuntimeError("DINOv3 correct precision is not configured")
        self._dinov3_correct_precision.begin_event()

    def run_dinov3_correct_block(
        self,
        layer_idx: int,
        x: torch.Tensor,
        dindice: torch.Tensor,
        rope,
        cache: Dict[str, Any],
        tag: str,
        *,
        source_key: str = "default",
        **kwargs,
    ):
        if self._dinov3_correct_precision is None:
            raise RuntimeError("DINOv3 correct precision is not configured")
        kwargs.setdefault(
            "persist_correction_residual",
            getattr(self, "_persist_correction_residual", False),
        )
        return self._dinov3_correct_precision.run_block(
            layer_idx,
            x,
            dindice,
            rope,
            cache,
            tag,
            source_key=source_key,
            **kwargs,
        )

    def dinov3_correct_event_metadata(self) -> Dict[str, Any]:
        if self._dinov3_correct_precision is None:
            return {}
        return self._dinov3_correct_precision.event_metadata()

    @staticmethod
    def _normalize_patch_score_map(score_map: torch.Tensor | None) -> torch.Tensor | None:
        if score_map is None:
            return None
        if score_map.ndim != 2:
            raise ValueError(f"Expected a 2D score map, got shape {tuple(score_map.shape)}")

        score_sums = score_map.sum(dim=1, keepdim=True, dtype=torch.float32)
        denom = score_sums.clamp_min(torch.finfo(torch.float32).eps).to(dtype=score_map.dtype)
        return score_map / denom

    @abstractmethod
    def load_model(self, model_name: str, config: Any):
        pass

    @abstractmethod
    def preprocess(self, batch_data: Any, task: Task, context: Dict[str, Any], config: Any):
        """Handles OpType.LOAD_INPUT logic for decoded numpy batches or precomputed tensors."""
        pass

    @abstractmethod
    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        """Handles OpType.PREPARE_TOKENS"""
        pass

    @abstractmethod
    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        """Handles OpType.APPROX_FORWARD"""
        pass

    @abstractmethod
    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        """Handles OpType.CORRECT_FORWARD"""
        pass

    @abstractmethod
    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        """Handles OpType.HEAD_INFERENCE"""
        pass

    @abstractmethod
    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        """Handles OpType.FULL_INFERENCE"""
        pass

    @abstractmethod
    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        """Handles OpType.DECIDE_EXIT"""
        pass

    @abstractmethod
    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        """
        Returns the final formatted results for the current batch state.
        Should return a dictionary mapping original request index to the result payload.
        Used by the worker to populate the response.
        """
        pass
