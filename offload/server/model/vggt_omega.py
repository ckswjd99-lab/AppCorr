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

    def _frames_to_tensor(self, frames, config=None, native_shapes=None) -> torch.Tensor:
        """Native frames -> the patch-aligned, aspect-preserving canvas the model expects.

        Delegates to the forked upstream preprocessing. An anisotropic resize to a square would
        leave depth looking fine and make camera pose unmeasurable, so this is not optional.

        `native_shapes` keeps the canvas tied to the *original* frame rather than to whatever shape
        the transmission policy reconstructed; without it, floor and ceiling silently disagree on
        the canvas for 6.3% of Co3D frames. See `preprocess_frames`.
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
            native_shapes=native_shapes,
        )

    @torch.inference_mode()
    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        frames = context.get("input_frames")
        if frames is None:
            frames = context.get("input_images_uint8")
        if frames is None:
            raise RuntimeError("Missing context['input_frames'] for VGGT-Omega full_inference().")

        images = self._frames_to_tensor(frames, config, context.get("input_native_shapes"))
        with self.dinov3_full_inference_precision(), torch.autocast("cuda", self.autocast_dtype):
            preds = self.model(images)

        context["vggt_preds"] = {k: v for k, v in preds.items()}
        context["vggt_input_hw"] = tuple(images.shape[-2:])

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        preds = context.get("vggt_preds")
        if preds is None:
            return {}
        from appcorr.models.vggt_omega.utils.pose_enc import encoding_to_camera

        # [1, S, H, W, 1] -> [S, H, W]. The trailing singleton is the head's channel axis and
        # carrying it further makes boolean-masked depth comparisons broadcast into an [N, N]
        # matrix instead of comparing elementwise.
        depth = preds["depth"][0, ..., 0]
        pose = preds["pose_enc"]         # [1, S, 9]

        # Decode the 9D encoding here rather than client-side: the focal terms are angular (FoV),
        # so recovering intrinsics needs the canvas the model actually ran on, and that shape is
        # per-request and known only here. Handing the raw encoding to the evaluator would make the
        # client re-derive a shape it never sees.
        extrinsics, intrinsics = encoding_to_camera(pose.float().cpu(), context["vggt_input_hw"])
        extrinsics = extrinsics[0].numpy()  # [S, 3, 4], camera-from-world, OpenCV axes

        # Everything below is a single sequence, so it lands at batch index 0 -- the frame axis
        # lives inside each array, not across the returned dict.
        return {
            0: {
                "depth": depth.float().cpu().numpy(),
                "pose_enc": pose[0].float().cpu().numpy(),
                "R": extrinsics[:, :3, :3],
                "T": extrinsics[:, :3, 3],
                "intrinsics": intrinsics[0].numpy() if intrinsics is not None else None,
                "input_hw": tuple(int(v) for v in context["vggt_input_hw"]),
                "num_frames": int(depth.shape[0]),
            }
        }

    def preprocess(self, batch_data: Any, task: Task, context: Dict[str, Any], config: Any):
        """Decoded transmission output -> `context['input_frames']`.

        The batch axis *is* the frame axis: one request is one multi-view sequence, so the S frames
        of that sequence arrive as what every other executor calls a batch. Nothing here reshapes
        them; `_frames_to_tensor` adds the leading singleton batch dim at inference time, because
        that is where the model's canvas is known.

        Frames may be wrapped as `{'image', 'target_shape'}` by the worker when the policy preserves
        native shapes -- which this model's policy always does, since VGGT's canvas is derived
        per-frame from the frame's own aspect ratio.
        """
        items = batch_data if isinstance(batch_data, (list, tuple)) else list(batch_data)
        frames, natives = [], []
        for item in items:
            is_dict = isinstance(item, dict)
            image = np.ascontiguousarray(item["image"] if is_dict else item)
            frames.append(image)
            ts = item.get("target_shape") if is_dict else None
            natives.append(tuple(int(v) for v in ts) if ts is not None else image.shape[:2])
        context["input_frames"] = frames
        # Kept separately from the frames: the canvas must follow the original shape, not the
        # reconstructed one, or the floor and ceiling conditions land on different canvases.
        context["input_native_shapes"] = natives
        return None

    # --- Staged pipeline: not implemented yet -------------------------------------------------
    # FULL_INFERENCE is the only supported op at the skeleton stage. These raise rather than
    # returning empty, so a config that schedules them fails immediately and visibly instead of
    # producing results from a half-run pipeline.

    @staticmethod
    def _approx_kwargs(config: Any) -> Dict[str, Any]:
        """The subset of appcorr options the block's `approx()` actually reads."""
        from offload.common.protocol import normalize_appcorr_kwargs

        opts = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        return {
            "appcorr_method": opts["method"],
            "server_pscore": opts["server_pscore"],
            "attn_col_alive_ratio": opts["attn_col_alive_ratio"],
            "debug": opts["debug"],
        }

    @torch.inference_mode()
    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        """Patch-embed the frames and assemble the aggregator's input tokens.

        This runs the whole 24-block patch-embed ViT, because that stack is per-frame and has no
        cross-frame dependency -- it is the only part of the model that can be done before every
        frame has arrived. Everything after it is gated on the first inter-frame block.
        """
        frames = context.get("input_frames")
        if frames is None:
            raise RuntimeError("Missing context['input_frames'] for VGGT-Omega prepare_tokens().")

        images = self._frames_to_tensor(frames, config, context.get("input_native_shapes"))
        cache: Dict[str, Any] = context.setdefault("cache_feature", {})

        # PREPARE_TOKENS runs once per transmitted group, so the first call sees the approximate
        # image and the second sees the refined one. The patch-embed stack is a correction target
        # like any other: the first pass runs it in approx mode and fills the `pe*` KV cache, the
        # second corrects against that cache rather than recomputing it from scratch. Re-running it
        # in *approx* mode on the second pass would overwrite the cache and leave correction
        # comparing the refined tokens against themselves.
        refining = context.get("vggt_approx_done", False)
        with torch.autocast("cuda", self.autocast_dtype):
            tokens, frame_rope, geom = self.model.aggregator.embed(
                images,
                cache_feature=cache,
                approx_kwargs=(self._correct_kwargs(config) if refining
                               else self._approx_kwargs(config)),
                correct=refining,
            )

        context["vggt_images"] = images          # the dense head re-reads the input images
        context["vggt_tokens"] = tokens
        context["vggt_rope"] = frame_rope
        context["vggt_geom"] = geom
        context["vggt_input_hw"] = tuple(images.shape[-2:])
        if not refining:
            context["vggt_outputs"] = [None] * self.model.aggregator.depth

    @torch.inference_mode()
    def approx_forward(self, params, context: Dict[str, Any], config: Any):
        """Aggregator blocks `[start, end)` on the approximate tokens."""
        if "vggt_tokens" not in context:
            raise RuntimeError("APPROX_FORWARD before PREPARE_TOKENS for VGGT-Omega.")
        agg = self.model.aggregator
        start, end = params.get("layers", (0, agg.depth))

        # The DINOv3 precision controller is built from a single `backbone.blocks` list; VGGT has
        # three stacks, so low-precision approx is not wired here yet. Skip the instrumentation
        # rather than fail, and report nothing rather than report a precision that is not in effect.
        if self._dinov3_approx_precision is not None:
            self.begin_dinov3_approx_event()

        with torch.autocast("cuda", self.autocast_dtype):
            outputs, tokens = agg.run_blocks(
                context["vggt_tokens"],
                context["vggt_rope"],
                context["vggt_geom"],
                block_range=(int(start), int(end)),
                outputs=context["vggt_outputs"],
                cache_feature=context["cache_feature"],
                approx_kwargs=self._approx_kwargs(config),
            )
        context["vggt_tokens"] = tokens
        context["vggt_outputs"] = outputs
        context["vggt_approx_done"] = True
        if self._dinov3_approx_precision is None:
            return {}
        return self.dinov3_approx_event_metadata()

    @staticmethod
    def _correct_kwargs(config: Any) -> Dict[str, Any]:
        from offload.common.protocol import normalize_appcorr_kwargs

        o = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        return {
            "appcorr_method": o["method"],
            "token_keep_ratio": o["token_keep_ratio"],
            "token_keep_thres": o["token_keep_thres"],
            "mobile_pscore": o["mobile_pscore"],
            "mobile_pscore_weight": o["mobile_pscore_weight"],
            "server_pscore": o["server_pscore"],
            "server_pscore_weight": o["server_pscore_weight"],
            "pscore_fusion": o["pscore_fusion"],
            "sdpa_query_bucket_size": o["sdpa_query_bucket_size"],
            "attn_col_alive_ratio": o["attn_col_alive_ratio"],
            "debug": o["debug"],
        }

    @torch.inference_mode()
    def correct_forward(self, params, context: Dict[str, Any], config: Any):
        """Aggregator blocks `[start, end)` in correct mode, on the refined tokens."""
        if not context.get("vggt_approx_done"):
            raise RuntimeError("CORRECT_FORWARD before APPROX_FORWARD for VGGT-Omega.")
        agg = self.model.aggregator
        start, end = params.get("layers", (0, agg.depth))

        with torch.autocast("cuda", self.autocast_dtype):
            outputs, tokens = agg.run_blocks(
                context["vggt_tokens"],
                context["vggt_rope"],
                context["vggt_geom"],
                block_range=(int(start), int(end)),
                outputs=context["vggt_outputs"],
                cache_feature=context["cache_feature"],
                approx_kwargs=self._correct_kwargs(config),
                correct=True,
            )
        context["vggt_tokens"] = tokens
        context["vggt_outputs"] = outputs

    @torch.inference_mode()
    def head_inference(self, task: Task, context: Dict[str, Any], config: Any):
        """Run the camera and depth heads on the cached layer outputs."""
        outputs = context.get("vggt_outputs")
        if outputs is None:
            raise RuntimeError("HEAD_INFERENCE before APPROX_FORWARD for VGGT-Omega.")
        missing = [i for i in sorted(self.model.aggregator.cached_layer_indices) if outputs[i] is None]
        if missing:
            # Loud, because the heads would otherwise read a None and fail far from the cause.
            raise RuntimeError(
                f"VGGT-Omega heads need cached layers {sorted(self.model.aggregator.cached_layer_indices)}; "
                f"blocks {missing} were never run. Did APPROX_FORWARD cover the full depth?"
            )

        model = self.model
        start = model.aggregator.patch_token_start
        preds: Dict[str, Any] = {
            "camera_and_register_tokens": outputs[-1][:, :, :start].contiguous(),
        }
        with torch.autocast("cuda", enabled=False):
            if model.camera_head is not None:
                preds["pose_enc"] = model.camera_head(outputs, patch_token_start=start)
            if model.dense_head is not None:
                depth, depth_conf = model.dense_head(
                    outputs, images=context["vggt_images"], patch_token_start=start
                )
                preds["depth"] = depth
                preds["depth_conf"] = depth_conf
        context["vggt_preds"] = preds

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        """No early exit: every frame is needed for the inter-frame attention the model is built on."""
        return {"num_exits": 0}
