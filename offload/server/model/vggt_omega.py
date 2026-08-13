"""VGGT-Omega executor: multi-view camera pose and depth.

Skeleton stage -- the stock (FULL_INFERENCE) path only. Approx/correct come later, once the shape of
the multi-frame contract has settled against a working baseline.

Two things differ from every existing executor and drive the design:

**The input carries a frame axis.** DINOv3 executors take `[B, H, W, C]`; the aggregator takes
`[B, S, 3, H, W]` and its whole purpose is the attention *between* those S frames. `S` therefore has
to survive the transmission path rather than being folded into the batch, which is why the frame axis
is read off `context['input_frames']` here instead of being reconstructed from `config.batch_size`.

**Inter-frame coupling is nearly total.** 19 of the aggregator's 24 blocks run global attention over
`S x tokens`; only blocks 2, 6, 9, 14, 20 restrict inter-frame exchange to the 17 camera+register
tokens per frame. Degrading one view therefore perturbs every other view's patch tokens, so a future
correction path cannot treat frames as independent. `cached_layer_indices = (4, 11, 17, 23)` is the
useful counterweight: the heads read only those four blocks, so correction only has to agree there.
"""

from typing import Any, Dict

import numpy as np
import torch

from offload.common import Task
from .base import ModelExecutor
from .utils import load_weight_mmap


class VGGTOmegaExecutor(ModelExecutor):
    DEFAULT_WEIGHTS = "~/cjpark/weights/vggt/vggt_omega_1b_512.pt"

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.autocast_dtype = torch.bfloat16

    def load_model(self, model_name: str, config: Any = None):
        from appcorr.models.vggt_omega.models.vggt_omega import VGGTOmega

        profile = config.get_input_profile_config() if config is not None else {}
        weights = profile.get("vggt_weights_path", self.DEFAULT_WEIGHTS)

        model = VGGTOmega()
        state = load_weight_mmap(weights)
        state = state.get("model", state.get("state_dict", state)) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # Loud on purpose: a silently partial load still produces plausible-looking depth and
            # poses, which is exactly the failure mode that is impossible to spot downstream.
            raise RuntimeError(
                f"[VGGT-Omega] state_dict mismatch: {len(missing)} missing, "
                f"{len(unexpected)} unexpected (first missing: {missing[:3]})"
            )
        print(f"[Executor] Loaded VGGT-Omega from {weights} ({len(state)} tensors)")
        self.model = model.eval().to(self.device)

    def _frames_to_tensor(self, frames, config=None) -> torch.Tensor:
        """Native frames -> the patch-aligned, aspect-preserving canvas the model expects.

        Delegates to the forked upstream preprocessing. An anisotropic resize to a square would
        leave depth looking fine and make camera pose unmeasurable, so this is not optional.
        """
        from .vggt_preprocess import preprocess_frames

        profile = config.get_input_profile_config() if config is not None else {}
        if not isinstance(frames, (list, tuple)):
            frames = list(np.asarray(frames))
        return preprocess_frames(
            frames,
            mode=profile.get("vggt_resize_mode", "balanced"),
            image_resolution=int(profile.get("vggt_resolution", 512)),
            patch_size=int(profile.get("vggt_patch_size", 16)),
            device=self.device,
        )

    @torch.inference_mode()
    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        frames = context.get("input_frames")
        if frames is None:
            frames = context.get("input_images_uint8")
        if frames is None:
            raise RuntimeError("Missing context['input_frames'] for VGGT-Omega full_inference().")

        images = self._frames_to_tensor(frames, config)
        with self.dinov3_full_inference_precision(), torch.autocast("cuda", self.autocast_dtype):
            preds = self.model(images)

        context["vggt_preds"] = {k: v for k, v in preds.items()}
        context["vggt_input_hw"] = tuple(images.shape[-2:])

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        preds = context.get("vggt_preds")
        if preds is None:
            return {}
        depth = preds["depth"][0]        # [S, H, W, 1]
        pose = preds["pose_enc"][0]      # [S, 9]
        return {
            0: {
                "depth": depth.float().cpu().numpy(),
                "pose_enc": pose.float().cpu().numpy(),
                "num_frames": int(depth.shape[0]),
            }
        }

    # --- Staged pipeline: not implemented yet -------------------------------------------------
    # FULL_INFERENCE is the only supported op at the skeleton stage. These raise rather than
    # returning empty, so a config that schedules them fails immediately and visibly instead of
    # producing results from a half-run pipeline.

    def preprocess(self, task: Task, context: Dict[str, Any], config: Any):
        """Frames arrive ready to use; nothing to do until a transmission policy reshapes them."""
        return None

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        raise NotImplementedError("VGGT-Omega: PREPARE_TOKENS not implemented (use FULL_INFERENCE)")

    def approx_forward(self, params, context: Dict[str, Any], config: Any):
        raise NotImplementedError("VGGT-Omega: APPROX_FORWARD not implemented (use FULL_INFERENCE)")

    def correct_forward(self, params, context: Dict[str, Any], config: Any):
        raise NotImplementedError("VGGT-Omega: CORRECT_FORWARD not implemented (use FULL_INFERENCE)")

    def head_inference(self, task: Task, context: Dict[str, Any], config: Any):
        raise NotImplementedError("VGGT-Omega: HEAD_INFERENCE not implemented (use FULL_INFERENCE)")

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        """No early exit: every frame is needed for the inter-frame attention the model is built on."""
        return {"num_exits": 0}
