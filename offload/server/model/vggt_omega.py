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
import os
import torch

from offload.common import Task
from .base import ModelExecutor
from .utils import load_weight_mmap


class VGGTOmegaExecutor(ModelExecutor):
    DEFAULT_WEIGHTS = "~/cjpark/weights/vggt/vggt_omega_1b_512.pt"

    def __init__(self, device: torch.device):
        super().__init__(device)
        # APPCORR_VGGT_FP32=1 disables autocast for A/B probes where bf16's shape-dependent
        # reduction order would mask (or mimic) a semantic difference. fp32 autocast is a no-op
        # context, so the flag costs nothing when unset.
        import os as _os
        self.autocast_dtype = (torch.float32 if _os.environ.get("APPCORR_VGGT_FP32")
                               else torch.bfloat16)

    def backbone_modules(self):
        """VGGT's aggregator -- the DINOv3-derived trunk that produces the tokens every head reads.

        The camera, depth and point heads are the header this VFM stops before. `aggregator` is
        checked first because that is what the VGGT builds in this repo call it; the fallbacks cover
        a plain `backbone` naming.
        """
        m = self.model
        for attr in ("aggregator", "backbone", "trunk"):
            sub = getattr(m, attr, None)
            if sub is not None:
                return [sub]
        return [m]

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

    def _build_mobile_hints(self, task, context, config) -> Dict[str, Any] | None:
        """Per-patch residual energy from the transmitted patches -> one hint tensor per stack.

        The score that ranks tokens for correction has two halves: how *attended to* a token is
        (server side) and how *wrong* it is (mobile side, the residual energy the policy stamped on
        each patch). Only the first was in use until this existed, which is half a score.

        The three stacks read different token axes, so the same per-frame hint has to be laid out
        three ways. Camera and register tokens get zero -- they have no image-space residual -- which
        leaves the server score deciding for them rather than inventing a number.
        """
        from offload.common.protocol import normalize_appcorr_kwargs

        opts = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        if opts["mobile_pscore"] == "none" or opts["mobile_pscore_weight"] == 0.0:
            return None
        if task is None or not getattr(task, "payload", None):
            return None

        geom = context.get("vggt_geom")
        if geom is None:
            return None
        S, N = geom["num_frames"], geom["num_tokens"]
        start = self.model.aggregator.patch_token_start
        n_patch = N - start
        gh, gw = geom["patch_grid_size"]
        if gh * gw != n_patch:
            raise RuntimeError(
                f"Patch grid {gh}x{gw} does not match {n_patch} patch tokens; the hint would be "
                "silently misaligned with the tokens it is meant to rank."
            )

        # Hints ride the finest transmitted level -- the residual, which is the error itself.
        finest = min(int(l) for l in config.transmission_kwargs.get("pyramid_levels", [0]))
        hint = np.zeros((S, n_patch), dtype=np.float32)
        seen = 0
        for patch in task.payload:
            if int(getattr(patch, "res_level", finest)) != finest:
                continue
            f = int(getattr(patch, "image_idx", -1))
            i = int(getattr(patch, "spatial_idx", -1))
            if 0 <= f < S and 0 <= i < n_patch:
                hint[f, i] = float(getattr(patch, "pscore_hint", 0.0))
                seen += 1
        if seen == 0:
            return None

        t = torch.from_numpy(hint).to(self.device)                  # [S, n_patch]
        per_frame = t                                                # pe*, frame*: rows are frames
        with_prefix = torch.cat(
            [t.new_zeros((S, start)), t], dim=1                      # [S, N], cam/register at 0
        )
        return {
            "pe": per_frame,
            "frame": per_frame,
            "interg": with_prefix.reshape(1, S * N),                 # global: one row, S*N tokens
            "interr": t.new_zeros((1, S * start)),                   # register-only: no image hint
        }

    def _group_token_mask(self, task, context, config) -> torch.Tensor | None:
        """[S, n_patch] bool: which patch tokens the arriving round actually carries.

        Interleaved correction must be restricted to the tokens whose refined values have arrived.
        Correcting a token the client has not sent yet would 'correct' it toward the approximate
        value it already holds -- a no-op that still costs a recompute and, worse, reports as
        corrected.
        """
        geom = context.get("vggt_geom")
        if geom is None or task is None or not getattr(task, "payload", None):
            return None
        # Derived from the grid, not from geom["num_tokens"]: that key only appears once the
        # frontier crosses into the aggregator and the camera/register prefix is assembled, which
        # happens after this runs.
        S = geom["num_frames"]
        gh, gw = geom["patch_grid_size"]
        n_patch = gh * gw

        # `spatial_idx` indexes the *transmission* grid, which is not the model's token grid for
        # scale-up frames: those are degraded in native coordinates, so a frame transmitted as
        # 14x24 patches is consumed by the model as 24x43 tokens. Reading the index directly then
        # marks 336 of 1032 tokens and silently drops the rest from correction -- 7 of 20 Co3D
        # sequences are affected, and they stay at the floor, which drags the mean from ~3.0 to 5.1.
        # So mark on the transmission grid per frame, then resample to the token grid.
        import torch.nn.functional as F

        from offload.policies.transmission.vggt_laplacian import VGGTLaplacianPolicy

        ph, pw = (int(v) for v in config.patch_size) if not isinstance(config.patch_size, int) \
            else (int(config.patch_size),) * 2
        shapes = context.get("input_native_shapes") or []

        tx_marks: Dict[int, torch.Tensor] = {}
        tx_grids: Dict[int, tuple] = {}
        seen = 0
        for patch in task.payload:
            if int(getattr(patch, "res_level", 0)) != 0:      # residual level only
                continue
            f, i = int(patch.image_idx), int(patch.spatial_idx)
            if not (0 <= f < S):
                continue
            if f not in tx_marks:
                native = shapes[f] if f < len(shapes) else None
                base = VGGTLaplacianPolicy._target_hw_for_level(config, 0, native) if native \
                    else (gh * ph, gw * pw)
                tg = (base[0] // ph, base[1] // pw)
                tx_grids[f] = tg
                tx_marks[f] = torch.zeros(tg, dtype=torch.float32, device=self.device)
            tg = tx_grids[f]
            if 0 <= i < tg[0] * tg[1]:
                tx_marks[f][i // tg[1], i % tg[1]] = 1.0
                seen += 1

        if not seen:
            return None

        mask = torch.zeros((S, n_patch), dtype=torch.bool, device=self.device)
        for f, marks in tx_marks.items():
            if tx_grids[f] == (gh, gw):
                mask[f] = marks.reshape(-1) > 0.5
            else:
                up = F.interpolate(marks[None, None], size=(gh, gw), mode="nearest")
                mask[f] = up.reshape(-1) > 0.5
        return mask

    def _arrived_masks(self, patch_mask: torch.Tensor, context) -> Dict[str, torch.Tensor]:
        """Lay the arrived-token mask out for each stack's own token axis.

        Camera and register tokens are always marked available: they carry no image residual, they
        number 17 per view against ~1032 patches, and leaving them out of every round would mean the
        tokens that actually carry pose never get corrected at all.
        """
        geom = context["vggt_geom"]
        S = geom["num_frames"]
        agg_pre = self.model.aggregator.patch_token_start
        pe_pre = 1 + self.model.aggregator.patch_embed.n_storage_tokens
        N = agg_pre + patch_mask.shape[1]      # same reason: num_tokens is not in geom yet

        def with_prefix(n_pre):
            return torch.cat(
                [patch_mask.new_ones((S, n_pre)), patch_mask], dim=1
            )

        frame_mask = with_prefix(agg_pre)                       # [S, N]
        return {
            "pe": with_prefix(pe_pre),                          # [S, 5 + n_patch]
            "frame": frame_mask,
            "interg": frame_mask.reshape(1, S * N),             # [1, S*N]
            "interr": patch_mask.new_ones((1, S * agg_pre)),    # [1, S*17]
        }

    @torch.inference_mode()
    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        """Project frames to patch tokens. No transformer blocks run here.

        The 24 patch-embed blocks used to run in this call, which made them preprocessing that every
        interleaved round repeated in full -- 28.7% of the forward, paid once per round. They are
        real computation and are now stages 0..23 of a single 48-stage axis that APPROX_FORWARD and
        CORRECT_FORWARD walk; stages 24..47 are the aggregator pairs.
        """
        frames = context.get("input_frames")
        if frames is None:
            raise RuntimeError("Missing context['input_frames'] for VGGT-Omega prepare_tokens().")

        images = self._frames_to_tensor(frames, config, context.get("input_native_shapes"))
        agg = self.model.aggregator
        cache: Dict[str, Any] = context.setdefault("cache_feature", {})
        refining = context.get("vggt_approx_done", False)

        with torch.autocast("cuda", self.autocast_dtype):
            pe_tokens, geom = agg.embed_prologue(images)

        context["vggt_images"] = images
        context["vggt_pe_tokens"] = pe_tokens
        context["vggt_geom"] = geom
        context["vggt_stage"] = 0            # nothing of the depth has been walked yet this round
        context["vggt_tokens"] = None        # assembled when the frontier crosses stage 24
        context["vggt_input_hw"] = tuple(images.shape[-2:])
        if not refining:
            context["vggt_outputs"] = [None] * agg.depth
        else:
            hints = context.get("vggt_mobile_hints")
            if hints is None:
                hints = self._build_mobile_hints(task, context, config)
                context["vggt_mobile_hints"] = hints
            # This round's tokens only -- deliberately not accumulated. State carries between rounds
            # through the KV cache: each correction scatters its recomputed K/V there, so round g
            # already sees every earlier round's corrections when it attends. Re-listing earlier
            # tokens would recompute what is already correct, and on the last round that makes the
            # whole schedule collapse into one-shot correction -- which is exactly what happened:
            # interleaved and one-shot agreed to 16 decimals.
            pm = self._group_token_mask(task, context, config)
            context["vggt_arrived_masks"] = self._arrived_masks(pm, context) if pm is not None else None

    @staticmethod
    def _correct_kwargs(config: Any) -> Dict[str, Any]:
        """Everything `block.correct()` reads. Wider than the approx set: correction also needs the
        selection parameters (which tokens, by what score) that the approximate pass has no use
        for."""
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

    def _walk(self, params, context, config, correct: bool):
        """Advance the 48-stage axis over `[start, end)`, in approx or correct mode.

        Stages 0..23 are patch-embed blocks, 24..47 aggregator pairs; crossing 24 assembles the
        camera/register prefix exactly once. Both modes share this so they cannot disagree about
        where a stage boundary is.
        """
        agg = self.model.aggregator
        pe_n, total = agg.PE_STAGES, agg.PE_STAGES + agg.depth
        lo, hi = params.get("layers", (0, total))
        lo, hi = int(lo), min(int(hi), total)
        if hi <= lo:
            return {}

        kwargs = self._correct_kwargs(config) if correct else self._approx_kwargs(config)
        if correct:
            hints = context.get("vggt_mobile_hints")
            if hints is not None:
                kwargs["mobile_pscore_hints"] = hints
            arrived = context.get("vggt_arrived_masks")
            if arrived is not None:
                kwargs["arrived_masks"] = arrived
        cache = context["cache_feature"]

        with torch.autocast("cuda", self.autocast_dtype):
            if lo < pe_n:
                pe_hi = min(hi, pe_n)
                tokens, _ = agg.run_pe_blocks(
                    context["vggt_pe_tokens"], context["vggt_geom"], (lo, pe_hi),
                    cache_feature=cache, approx_kwargs=kwargs, correct=correct,
                )
                context["vggt_pe_tokens"] = tokens

            if hi > pe_n:
                if context.get("vggt_tokens") is None:
                    tok, rope, geom = agg.assemble_tokens(
                        context["vggt_pe_tokens"], context["vggt_geom"]
                    )
                    context["vggt_tokens"], context["vggt_rope"], context["vggt_geom"] = tok, rope, geom
                outputs, tokens = agg.run_blocks(
                    context["vggt_tokens"], context["vggt_rope"], context["vggt_geom"],
                    block_range=(max(lo, pe_n) - pe_n, hi - pe_n),
                    outputs=context["vggt_outputs"],
                    cache_feature=cache, approx_kwargs=kwargs, correct=correct,
                )
                context["vggt_tokens"], context["vggt_outputs"] = tokens, outputs

        context["vggt_stage"] = hi
        if correct and os.environ.get("APPCORR_VGGT_TRACE"):
            cf = context["cache_feature"]
            kept = cf.get("_token_patch_kept_total", cf.get("_token_pscore_kept_patch_total"))
            full = cf.get("_token_patch_full_total", cf.get("_token_pscore_full_patch_total"))
            keys = [k for k in cf if k.startswith("_token")]
            print(f"[vggt-trace] correct layers=({lo},{hi}) stat_keys={keys} "
                  f"kept={kept} full={full}", flush=True)
        return {}

    @torch.inference_mode()
    def approx_forward(self, params, context: Dict[str, Any], config: Any):
        if "vggt_pe_tokens" not in context:
            raise RuntimeError("APPROX_FORWARD before PREPARE_TOKENS for VGGT-Omega.")
        if self._dinov3_approx_precision is not None:
            self.begin_dinov3_approx_event()
        self._walk(params, context, config, correct=False)
        context["vggt_approx_done"] = True
        if self._dinov3_approx_precision is None:
            return {}
        return self.dinov3_approx_event_metadata()

    @torch.inference_mode()
    def correct_forward(self, params, context: Dict[str, Any], config: Any):
        if not context.get("vggt_approx_done"):
            raise RuntimeError("CORRECT_FORWARD before APPROX_FORWARD for VGGT-Omega.")
        return self._walk(params, context, config, correct=True)

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
