"""
progressive_model.py  (pi0-FAST)

pi0-FAST progressive-prefill model with three supported paths:

* Progressive vision injection: SigLIP runs a low-res base and patch correction, then LeRobot owns
  the full PaliGemma prefix and FAST decode.
* Partial-token ViT+LLM: one SigLIP base pass ranks patches with either the legacy hidden-residual
  pscore, L2-to-L0 visual-residual energy times SigLIP attention, or an optional fusion with Gemma
  vision/language-query attention. Selected vision tokens are corrected in SigLIP, and the same
  selected positions plus the text permanent group are corrected in the bidirectional Gemma prefix
  before an AppCorr KV-cache FAST decode.
* Block-causal: SigLIP is cumulatively corrected in spatial arrival groups. Each newly arrived
  vision group is prefilled through Gemma exactly once, bidirectionally within the group and
  attending only to arrived vision groups. Language is one final bidirectional block over all
  arrived vision and language tokens, followed by the stock-style causal FAST decode. This mode
  intentionally changes PaliGemma's globally bidirectional prefix and is not an exactness tier.

The partial-token path is bit-exact with stock at 100% correction in float32. Exactness depends on
preserving LeRobot/Transformers' embedding scaling operation order, including mathematically
redundant divide/multiply round trips whose float32 rounding can change a later FAST argmax.

Double-scaling note: transformers>=4.57 `GemmaModel.forward` scales inputs_embeds by sqrt(hidden),
but lerobot's `embed_prefix_fast` ALSO pre-scales them -- a double-scaling that collapses the model
(see memory `project_pi0fast_double_scaling_bug`). `install_gemma_scaling_fix()` (applied at import)
neutralizes the extra HF scaling; it is required for any pi0-FAST inference under transformers>=4.57.
"""

from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F


_SCALING_FIX_INSTALLED = False
Pi0FastScoreMode = Literal[
    "vit",
    "visual_residual_attn",
    "visual_residual_llm_language",
    "vit_llm_vision",
    "vit_llm_language",
]


def install_gemma_scaling_fix():
    """Fix the transformers>=4.57 x lerobot-0.4.4 pi0-FAST embedding double-scaling.

    transformers>=4.57 `GemmaModel.forward` scales inputs_embeds by sqrt(hidden). lerobot's
    `embed_prefix_fast` ALSO pre-scales the LANGUAGE/FAST embeds by sqrt(hidden) (via
    `embed_language_tokens(...) * sqrt(dim)`), so those tokens get scaled twice and collapse the
    model. The IMAGE embeds are NOT pre-scaled (they come from `get_image_features`, already
    /sqrt(hidden)), so GemmaModel's single scaling is exactly right for them.

    The correct fix therefore touches ONLY the language/FAST path: patch `embed_language_tokens` to
    pre-divide by sqrt(hidden) so lerobot's subsequent `* sqrt(dim)` cancels, leaving those tokens
    unscaled -> GemmaModel scales them once (correct). Images are untouched (stay correct). This
    fixes BOTH teacher-forcing and closed-loop rollout. (An earlier version divided the *whole*
    inputs_embeds in GemmaModel, which also shrank the already-correct image tokens by sqrt(hidden)
    -> weak vision -> the arm approached objects but grasped imprecisely and never completed a task.)
    Idempotent.
    """
    global _SCALING_FIX_INSTALLED
    if _SCALING_FIX_INSTALLED:
        return
    import math
    import lerobot.policies.pi0_fast.modeling_pi0_fast as MOD
    _orig = MOD.PI0FastPaliGemma.embed_language_tokens

    def _patched(self, tokens):
        emb = _orig(self, tokens)
        return emb / math.sqrt(emb.shape[-1])

    MOD.PI0FastPaliGemma.embed_language_tokens = _patched
    _SCALING_FIX_INSTALLED = True


def configure_policy_precision(
    policy,
    precision: Literal["inherit", "float32", "bfloat16"] = "inherit",
) -> str:
    """Put an already-loaded LeRobot pi0-FAST policy in a known model precision.

    ``lerobot-eval`` constructs the policy before AppCorr can wrap it, so precision must be
    configured on that existing object. ``inherit`` is deliberately a no-op and is the mode used
    for the official AMP/bfloat16 path. The helper is shared by stock and partial launch modes so
    their comparisons cannot accidentally use different weight dtypes.
    """
    if precision not in {"inherit", "float32", "bfloat16"}:
        raise ValueError(
            f"Unsupported pi0-FAST precision {precision!r}; "
            "expected 'inherit', 'float32', or 'bfloat16'."
        )
    if precision != "inherit":
        policy.model.paligemma_with_expert.to_bfloat16_for_selected_params(precision)
    policy._appcorr_precision = precision
    return precision


class Pi0FastProgressiveModel:
    def __init__(self, checkpoint: str, device: torch.device,
                 stats_dataset_repo: str = "HuggingFaceVLA/libero"):
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
        from lerobot.policies.factory import make_pre_post_processors

        install_gemma_scaling_fix()
        pol = PI0FastPolicy.from_pretrained(checkpoint).to(device).eval()
        # Consistent float32 (native mixed precision throws Float-vs-BFloat16 in SigLIP layer_norm).
        pol.model.paligemma_with_expert.to_bfloat16_for_selected_params("float32")
        self.pre, self.post = make_pre_post_processors(
            pol.config, pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}})
        self._setup(pol, device)

    @classmethod
    def from_policy(
        cls,
        pol,
        device: torch.device,
        precision: Literal["inherit", "float32", "bfloat16"] = "inherit",
    ):
        """Build around an already-loaded PI0FastPolicy (e.g. lerobot-eval's) -- no reload, no
        preprocessor (the eval feeds a preprocessed batch to _partial_from_batch).

        ``precision='inherit'`` preserves the policy's dtype (normally used with the official AMP
        context); ``float32`` is the parity/debug path and must be paired with AMP disabled.
        """
        install_gemma_scaling_fix()
        configure_policy_precision(pol, precision)
        self = cls.__new__(cls)
        self.pre = self.post = None
        self._setup(pol, device)
        return self

    def _setup(self, pol, device: torch.device):
        from appcorr.models.pi0fast.siglip_vision import ApproxCorrectSiglipBackbone
        from appcorr.models.pi0fast.gemma_prefill_layer import ApproxCorrectGemmaDecoderLayer

        self.device = device
        self.pol = pol
        self.m = pol.model
        self.pg = self.m.paligemma_with_expert.paligemma
        self.projector = self.pg.model.multi_modal_projector
        self.hidden_size = self.pg.config.text_config.hidden_size
        self.img_scale = self.hidden_size ** 0.5   # get_image_features /sqrt

        self.siglip = ApproxCorrectSiglipBackbone(self.pg.model.vision_tower.vision_model).to(device)
        self.lm = self.pg.model.language_model
        self.llm_layers = [ApproxCorrectGemmaDecoderLayer.from_stock(l).to(device) for l in self.lm.layers]
        self.embed_tokens = self.lm.embed_tokens
        self.rotary_emb = self.lm.rotary_emb
        self.lm_norm = self.lm.norm
        self.lm_head = self.pg.lm_head
        self.tok = pol._paligemma_tokenizer
        self.max_decoding_steps = pol.config.max_decoding_steps
        self.n_action_steps = pol.config.n_action_steps
        self.action_dim = pol.config.output_features["action"].shape[0]

        self.cache_feature: Dict[str, Any] = {}
        self.tokens_per_image = None
        self._injected: Optional[List[torch.Tensor]] = None
        self._orig_embed_image = self.m.paligemma_with_expert.embed_image

    # ------------------------------------------------------------------ vision features
    @staticmethod
    def _base_pixel(px: torch.Tensor, factor: int = 4) -> torch.Tensor:
        """Low-res base: downsample by `factor` then upsample back (the progressive-vision base)."""
        h, w = px.shape[-2:]
        low = F.interpolate(px, size=(h // factor, w // factor), mode="bilinear", align_corners=False)
        return F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)

    @staticmethod
    def _patch_visual_residual_energy(
        exact_pixel: torch.Tensor,
        base_pixel: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        """Per-patch RGB energy of the L2-to-L0 visual residual. Returns [B, token_count]."""
        grid_size = int(round(token_count ** 0.5))
        if grid_size * grid_size != token_count:
            raise ValueError(
                f"Expected a square vision-token grid, got {token_count} tokens"
            )
        height, width = exact_pixel.shape[-2:]
        if base_pixel.shape != exact_pixel.shape:
            raise ValueError(
                "Base and exact pixels must have the same shape, got "
                f"{tuple(base_pixel.shape)} and {tuple(exact_pixel.shape)}"
            )
        if height % grid_size or width % grid_size:
            raise ValueError(
                f"Image shape {(height, width)} is not divisible by grid {grid_size}"
            )
        patch_height = height // grid_size
        patch_width = width // grid_size
        residual = exact_pixel.float() - base_pixel.float()
        return (
            residual.square()
            .unfold(2, patch_height, patch_height)
            .unfold(3, patch_width, patch_width)
            .sum(dim=(1, 4, 5))
            .reshape(exact_pixel.shape[0], token_count)
        )

    def _project(self, feats: torch.Tensor) -> torch.Tensor:
        """last_hidden_state -> multi_modal_projector -> /sqrt(hidden)  (== get_image_features)."""
        return self.projector(feats) / self.img_scale

    def _proj_raw(self, feats: torch.Tensor) -> torch.Tensor:
        """Image embed at Gemma-layer scale, preserving stock round-trip arithmetic.

        Stock PaliGemma computes ``projector / sqrt(hidden)`` in get_image_features and GemmaModel
        multiplies it back by ``sqrt(hidden)``. Returning the raw projector is mathematically equal
        but not bit-exact because it skips that divide/multiply round trip.
        """
        return (self.projector(feats) / self.img_scale) * self.img_scale

    def _language_at_layer_scale(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Match patched LeRobot embed -> LeRobot scale -> GemmaModel scale operation order."""
        embeddings = self.m.paligemma_with_expert.embed_language_tokens(token_ids)
        embeddings = embeddings * self.img_scale
        return embeddings * self.img_scale

    def _rope(self, positions: torch.Tensor):
        dummy = torch.zeros(1, positions.shape[-1], self.hidden_size, device=self.device)
        return self.rotary_emb(dummy, positions.unsqueeze(0) if positions.dim() == 1 else positions)

    @staticmethod
    def _normalize_positive_score(score: torch.Tensor) -> torch.Tensor:
        score = score.float().clamp_min(0)
        return score / score.mean().clamp_min(1e-12)

    @classmethod
    def _fuse_vision_pscore(
        cls,
        vit_pscore: torch.Tensor,
        llm_vision_attention: torch.Tensor,
        llm_vision_weight: float,
    ) -> torch.Tensor:
        """Weighted geometric fusion; weight zero preserves the legacy ranking exactly."""
        if llm_vision_weight < 0:
            raise ValueError("llm_vision_weight must be non-negative")
        if llm_vision_weight == 0:
            return vit_pscore.float()
        vit = cls._normalize_positive_score(vit_pscore).clamp_min(1e-12)
        llm = cls._normalize_positive_score(llm_vision_attention).clamp_min(1e-12)
        return torch.exp(vit.log() + llm_vision_weight * llm.log())

    # ------------------------------------------------------------------ block-causal (progressive ViT, one-pass LLM)
    def _init_block_causal_cache(self, prefix_length: int, dtype: torch.dtype) -> None:
        """Allocate fixed-position Gemma K/V storage for first-time grouped prefills."""
        attn0 = self.llm_layers[0].self_attn
        for layer_index in range(len(self.llm_layers)):
            self.cache_feature[f"llm{layer_index}_kv"] = torch.zeros(
                1,
                attn0.num_key_value_heads,
                prefix_length,
                2,
                attn0.head_dim,
                dtype=dtype,
                device=self.device,
            )

    def _block_causal_prefill(
        self,
        x_sel: torch.Tensor,
        query_indices: torch.Tensor,
        allowed_key_indices: torch.Tensor,
        prefix_length: int,
        pad: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> None:
        """Prefill one block once: bidirectional inside the block, causal only between blocks."""
        query_indices = query_indices.to(device=self.device, dtype=torch.long)
        allowed_key_indices = allowed_key_indices.to(device=self.device, dtype=torch.long)
        if query_indices.numel() == 0:
            return
        allowed = torch.zeros(
            prefix_length,
            dtype=torch.bool,
            device=self.device,
        )
        allowed[allowed_key_indices] = True
        allowed &= pad[0]
        mask_rows = allowed.view(1, 1, prefix_length).expand(
            1,
            query_indices.numel(),
            prefix_length,
        )
        mask4d = self.m._prepare_attention_masks_4d(mask_rows, dtype=x_sel.dtype)
        cos_sel, sin_sel = cos[:, query_indices], sin[:, query_indices]
        for layer_index, layer in enumerate(self.llm_layers):
            x_sel, self.cache_feature = layer.prefill(
                x_sel,
                query_indices,
                self.cache_feature,
                f"llm{layer_index}",
                cos_sel,
                sin_sel,
                key_end=prefix_length,
                causal=False,
                attn_mask=mask4d,
            )
        self.cache_feature["_x"][:, query_indices] = x_sel.to(
            self.cache_feature["_x"].dtype
        )

    @torch.inference_mode()
    def _block_causal_from_batch(
        self,
        b: Dict[str, Any],
        num_groups: int = 4,
        base_factor: int = 4,
    ) -> torch.Tensor:
        """Return normalized actions from progressive vision + one-pass block-causal Gemma."""
        return self._detok(
            self._block_causal_tokens_from_batch(
                b,
                num_groups=num_groups,
                base_factor=base_factor,
            )
        )

    @torch.inference_mode()
    def predict_action_block_causal(
        self,
        obs: Dict[str, Any],
        num_groups: int = 4,
        base_factor: int = 4,
    ) -> np.ndarray:
        """Public observation-level entrypoint for the approximate block-causal path."""
        return self._block_causal_from_batch(
            self.pre(obs),
            num_groups=num_groups,
            base_factor=base_factor,
        )

    @torch.inference_mode()
    def _block_causal_tokens_from_batch(
        self,
        b: Dict[str, Any],
        num_groups: int = 4,
        base_factor: int = 4,
    ) -> torch.Tensor:
        """Progressive SigLIP plus group-level block-causal PaliGemma prefix.

        There is no token pruning and no Gemma approximate/correct pass. For each spatial arrival
        round, SigLIP cumulatively corrects every arrived patch so the final vision state is exact.
        Only the newly arrived positions are then finalized in Gemma. A block sees all prior vision
        blocks and itself; its own queries/keys are fully bidirectional. Valid language tokens plus
        the appended BOS form one final bidirectional block over the complete vision prefix.
        """
        from lerobot.policies.pi0_fast.modeling_pi0_fast import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        if num_groups < 1:
            raise ValueError("num_groups must be positive")
        if base_factor < 1:
            raise ValueError("base_factor must be positive")

        dev = self.device
        images, image_masks = self.pol._preprocess_images(b)
        text_ids = b[OBS_LANGUAGE_TOKENS]
        text_mask = b[OBS_LANGUAGE_ATTENTION_MASK]
        bos = torch.full(
            (1, 1),
            self.tok.bos_token_id,
            device=dev,
            dtype=text_ids.dtype,
        )
        text_ids = torch.cat([text_ids, bos], dim=1)
        text_mask = torch.cat(
            [
                text_mask,
                torch.ones(
                    (1, 1),
                    dtype=text_mask.dtype,
                    device=dev,
                ),
            ],
            dim=1,
        )

        self.cache_feature = {}
        base_features = []
        real_flags = []
        for image_index, image in enumerate(images):
            real = bool(image_masks[image_index].any().item())
            real_flags.append(real)
            features, self.cache_feature = self.siglip.approx_forward(
                self._base_pixel(image, base_factor),
                self.cache_feature,
                f"img{image_index}",
            )
            base_features.append(features)

        tokens_per_image = base_features[0].shape[1]
        num_image_positions = tokens_per_image * len(images)
        text_embeddings = self._language_at_layer_scale(text_ids)
        prefix_length = num_image_positions + text_ids.shape[1]
        pad = torch.cat(
            [
                (
                    torch.ones(
                        1,
                        tokens_per_image,
                        dtype=torch.bool,
                        device=dev,
                    )
                    if real
                    else torch.zeros(
                        1,
                        tokens_per_image,
                        dtype=torch.bool,
                        device=dev,
                    )
                )
                for real in real_flags
            ]
            + [text_mask.bool()],
            dim=1,
        )
        positions = torch.cumsum(pad.long(), dim=1)[0] - 1
        cos, sin = self._rope(positions)

        prefix_dtype = text_embeddings.dtype
        self._init_block_causal_cache(prefix_length, prefix_dtype)
        self.cache_feature["_x"] = torch.zeros(
            1,
            prefix_length,
            self.hidden_size,
            dtype=prefix_dtype,
            device=dev,
        )

        local_groups = [
            group
            for group in torch.tensor_split(
                torch.arange(tokens_per_image, device=dev),
                num_groups,
            )
            if group.numel() > 0
        ]
        arrived_local = torch.zeros(0, dtype=torch.long, device=dev)
        vision_block_sizes = []
        cumulative_vision_queries = 0
        final_features = list(base_features)
        for local_group in local_groups:
            arrived_local = torch.cat([arrived_local, local_group])
            block_indices = []
            block_embeddings = []
            arrived_indices = []
            for image_index, image in enumerate(images):
                if not real_flags[image_index]:
                    continue
                features, self.cache_feature = self.siglip.correct_forward(
                    image,
                    arrived_local,
                    self.cache_feature,
                    f"img{image_index}",
                )
                final_features[image_index] = features
                projected = self._proj_raw(features)
                block_indices.append(
                    local_group + image_index * tokens_per_image
                )
                block_embeddings.append(projected[:, local_group])
                arrived_indices.append(
                    arrived_local + image_index * tokens_per_image
                )
                cumulative_vision_queries += int(arrived_local.numel())
            if not block_indices:
                continue
            query_indices = torch.cat(block_indices)
            allowed_key_indices = torch.cat(arrived_indices)
            self._block_causal_prefill(
                torch.cat(block_embeddings, dim=1),
                query_indices,
                allowed_key_indices,
                prefix_length,
                pad,
                cos,
                sin,
            )
            vision_block_sizes.append(int(query_indices.numel()))

        valid_language_local = text_mask[0].bool().nonzero(as_tuple=False).flatten()
        language_indices = valid_language_local + num_image_positions
        real_vision_ranges = [
            torch.arange(
                image_index * tokens_per_image,
                (image_index + 1) * tokens_per_image,
                device=dev,
            )
            for image_index, real in enumerate(real_flags)
            if real
        ]
        real_vision_indices = (
            torch.cat(real_vision_ranges)
            if real_vision_ranges
            else torch.zeros(0, dtype=torch.long, device=dev)
        )
        self._block_causal_prefill(
            text_embeddings[:, valid_language_local],
            language_indices,
            torch.cat([real_vision_indices, language_indices]),
            prefix_length,
            pad,
            cos,
            sin,
        )

        real_image_count = sum(real_flags)
        self.last_pscore_components = None
        if getattr(self, "capture_debug", False):
            self.debug_block_final_vision_features = [
                features.detach().clone() for features in final_features
            ]
        self.last_recompute_stats = {
            "path": "block_causal",
            "num_groups": len(local_groups),
            "base_factor": base_factor,
            "real_images": real_image_count,
            "tokens_per_image": tokens_per_image,
            "vision_block_sizes": vision_block_sizes,
            "vit_corrected_query_tokens": cumulative_vision_queries,
            "vit_unique_real_tokens": real_image_count * tokens_per_image,
            "llm_prefilled_vision_tokens": sum(vision_block_sizes),
            "llm_prefilled_language_tokens": int(language_indices.numel()),
            "llm_prefilled_tokens": sum(vision_block_sizes)
            + int(language_indices.numel()),
            "llm_prefix_tokens": prefix_length,
            "llm_valid_prefix_tokens": int(pad.sum().item()),
        }
        return self._generate_fast_from_cache(prefix_length, pad)

    # ------------------------------------------------------------------ partial-token (ViT + LLM)
    @torch.inference_mode()
    def predict_action_partial_token(self, obs: Dict[str, Any], keep: float = 0.5,
                                     base_factor: int = 4, correct_text: bool = True,
                                     score_mode: Pi0FastScoreMode = "vit",
                                     llm_vision_weight: float = 1.0) -> np.ndarray:
        """obs -> preprocess -> _partial_from_batch. See _partial_from_batch for the method."""
        return self._partial_from_batch(
            self.pre(obs),
            keep,
            base_factor,
            correct_text,
            score_mode,
            llm_vision_weight,
        )

    @torch.inference_mode()
    def _partial_from_batch(self, b: Dict[str, Any], keep: float = 0.5,
                            base_factor: int = 4, correct_text: bool = True,
                            score_mode: Pi0FastScoreMode = "vit",
                            llm_vision_weight: float = 1.0) -> torch.Tensor:
        """Return the normalized action chunk produced by partial-token generation."""
        gen = self._partial_tokens_from_batch(
            b,
            keep,
            base_factor,
            correct_text,
            score_mode,
            llm_vision_weight,
        )
        return self._detok(gen)

    @torch.inference_mode()
    def _partial_tokens_from_batch(self, b: Dict[str, Any], keep: float = 0.5,
                                   base_factor: int = 4, correct_text: bool = True,
                                   score_mode: Pi0FastScoreMode = "vit",
                                   llm_vision_weight: float = 1.0) -> torch.Tensor:
        """DINOv3-style partial-token progressive prefill on BOTH the SigLIP ViT and the Gemma LLM,
        given an already-preprocessed batch `b` (so it can patch PI0FastPolicy.predict_action_chunk).

        one approx + one pscore-selected correct:
          vision: SigLIP base approx (+legacy hidden or L2-to-L0 visual residual pscore).
          LLM:    bidirectional approx on the base vision features + text, optionally collecting
                  vision-query or valid-language-query -> vision-key received attention for pscore
                  fusion.
          select: rank each image's patches with the requested score mode, then correct the selected
                  patches in SigLIP and the same vision positions (+ the text permanent group) in
                  Gemma. At keep=1.0 (+correct_text) this reproduces the stock prefix -> stock action.
        Returns generated FAST token IDs. `_partial_from_batch` detokenizes these to the normalized
        action chunk expected by PI0FastPolicy.predict_action_chunk.
        """
        from lerobot.policies.pi0_fast.modeling_pi0_fast import (
            OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)
        supported_score_modes = {
            "vit",
            "visual_residual_attn",
            "visual_residual_llm_language",
            "vit_llm_vision",
            "vit_llm_language",
        }
        if score_mode not in supported_score_modes:
            raise ValueError(
                f"Unsupported score_mode {score_mode!r}; expected one of "
                f"{sorted(supported_score_modes)}"
            )
        if llm_vision_weight < 0:
            raise ValueError("llm_vision_weight must be non-negative")
        dev = self.device
        images, img_masks = self.pol._preprocess_images(b)          # 3 imgs, masks [T,T,F]
        if getattr(self, "capture_selection_frames", False):
            self.last_input_images = [
                image.detach().clone() for image in images
            ]
            self.last_input_image_masks = [
                mask.detach().clone() for mask in img_masks
            ]
        text_ids = b[OBS_LANGUAGE_TOKENS]
        text_mask = b[OBS_LANGUAGE_ATTENTION_MASK]
        bos = torch.full((1, 1), self.tok.bos_token_id, device=dev, dtype=text_ids.dtype)
        text_ids = torch.cat([text_ids, bos], dim=1)
        text_mask = torch.cat([text_mask, torch.ones((1, 1), dtype=text_mask.dtype, device=dev)], dim=1)

        # ---- vision base approximation and legacy ViT pscore ----
        self.cache_feature = {}
        base_feats, base_pixels, vit_scores, real_flags = [], [], [], []
        for i, img in enumerate(images):
            real = img_masks[i].any().item()
            real_flags.append(real)
            base_pixel = self._base_pixel(img, base_factor)
            bf, self.cache_feature = self.siglip.approx_forward(
                base_pixel, self.cache_feature, f"img{i}", pscore=real)
            base_feats.append(bf)
            base_pixels.append(base_pixel)
            if real:
                vit_scores.append(self.siglip.get_pscore(self.cache_feature, f"img{i}")[0])
            else:
                vit_scores.append(None)
        tpi = base_feats[0].shape[1]                                  # tokens per image (256)
        n_img = tpi * len(images)

        # ---- build approximate prefix at the Gemma-layer scale ----
        approx_img = torch.cat([self._proj_raw(f) for f in base_feats], dim=1)      # [1, n_img, D]
        text_emb = self._language_at_layer_scale(text_ids)                          # [1, L, D]
        approx_prefix = torch.cat([approx_img, text_emb], dim=1)
        P = approx_prefix.shape[1]; L = text_ids.shape[1]

        # ---- pad mask, 2D bidirectional prefix mask, positions (match lerobot) ----
        pad = torch.cat([
            torch.ones(1, tpi, dtype=torch.bool, device=dev) if img_masks[i].any() else
            torch.zeros(1, tpi, dtype=torch.bool, device=dev) for i in range(len(images))
        ] + [text_mask.bool()], dim=1)                               # [1, P]
        segs = [("image", n_img), ("language", L)]
        mask2d = self.m._create_custom_attention_mask_fast(segs, pad, 1)            # [1, P, P] bool
        mask4d = self.m._prepare_attention_masks_4d(mask2d, dtype=approx_prefix.dtype)  # [1,1,P,P]
        positions = torch.cumsum(pad.long(), dim=1)[0] - 1                          # [P]
        cos, sin = self._rope(positions)

        real_vision_ranges = [
            torch.arange(i * tpi, (i + 1) * tpi, device=dev)
            for i, real in enumerate(real_flags)
            if real
        ]
        real_vision_indices = (
            torch.cat(real_vision_ranges)
            if real_vision_ranges
            else torch.zeros(0, dtype=torch.long, device=dev)
        )
        llm_query_group = None
        score_query_groups = None
        if score_mode == "vit_llm_vision":
            llm_query_group = "vision"
            score_query_groups = {llm_query_group: real_vision_indices}
        elif score_mode in {
            "vit_llm_language",
            "visual_residual_llm_language",
        }:
            llm_query_group = "language"
            language_indices = torch.arange(n_img, P, device=dev)
            valid_language_indices = language_indices[text_mask[0].bool()]
            score_query_groups = {llm_query_group: valid_language_indices}
        score_key_indices = (
            real_vision_indices if score_query_groups is not None else None
        )

        # ---- LLM approx on base vision features; collect attention only for fused scoring ----
        x = approx_prefix
        for i, layer in enumerate(self.llm_layers):
            x, self.cache_feature = layer.approx(
                x,
                self.cache_feature,
                f"llm{i}",
                cos,
                sin,
                causal=False,
                attn_mask=mask4d,
                score_query_groups=score_query_groups,
                score_key_indices=score_key_indices,
            )
        self.cache_feature["_x"] = x

        llm_attention = None
        if llm_query_group is not None:
            llm_attention = torch.stack(
                [
                    self.cache_feature[f"llm{i}_received_attn_{llm_query_group}"]
                    for i in range(len(self.llm_layers))
                ]
            ).mean(dim=0)[0]

        # ---- fuse scores, select per image, and correct SigLIP only after LLM scoring ----
        corr_feats, sel_per_image = [], []
        pscore_components = []
        llm_cursor = 0
        for i, img in enumerate(images):
            if not real_flags[i]:
                corr_feats.append(base_feats[i])
                sel_per_image.append(torch.zeros(0, dtype=torch.long, device=dev))
                pscore_components.append(None)
                continue
            vit_pscore = vit_scores[i]
            visual_residual_energy = None
            siglip_attention = None
            if score_mode == "visual_residual_attn":
                visual_residual_energy = self._patch_visual_residual_energy(
                    img,
                    base_pixels[i],
                    tpi,
                )[0]
                siglip_attention = self.cache_feature[f"img{i}_avg_attn"][0].float()
                llm_pscore = None
                combined_pscore = visual_residual_energy * siglip_attention
            elif score_mode == "visual_residual_llm_language":
                visual_residual_energy = self._patch_visual_residual_energy(
                    img,
                    base_pixels[i],
                    tpi,
                )[0]
                llm_pscore = llm_attention[llm_cursor:llm_cursor + tpi]
                llm_cursor += tpi
                combined_pscore = visual_residual_energy * llm_pscore.float()
            elif llm_attention is None:
                llm_pscore = None
                combined_pscore = vit_pscore.float()
            else:
                llm_pscore = llm_attention[llm_cursor:llm_cursor + tpi]
                llm_cursor += tpi
                combined_pscore = self._fuse_vision_pscore(
                    vit_pscore,
                    llm_pscore,
                    llm_vision_weight,
                )
            k = max(1, min(int(round(keep * tpi)), tpi))
            sel = torch.topk(combined_pscore, k, largest=True).indices.sort().values
            cf, self.cache_feature = self.siglip.correct_forward(
                img,
                sel,
                self.cache_feature,
                f"img{i}",
            )
            corr_feats.append(cf)
            sel_per_image.append(sel)
            pscore_components.append(
                {
                    "vit": vit_pscore.detach().clone(),
                    "visual_residual_energy": (
                        visual_residual_energy.detach().clone()
                        if visual_residual_energy is not None
                        else None
                    ),
                    "siglip_attention": (
                        siglip_attention.detach().clone()
                        if siglip_attention is not None
                        else None
                    ),
                    "llm_vision": (
                        llm_pscore.detach().clone()
                        if llm_query_group == "vision"
                        else None
                    ),
                    "llm_language": (
                        llm_pscore.detach().clone()
                        if llm_query_group == "language"
                        else None
                    ),
                    "llm_attention": (
                        llm_pscore.detach().clone() if llm_pscore is not None else None
                    ),
                    "combined": combined_pscore.detach().clone(),
                    "selected": sel.detach().clone(),
                }
            )
        self.last_pscore_components = {
            "mode": score_mode,
            "llm_query_group": llm_query_group,
            "llm_vision_weight": llm_vision_weight,
            "per_image": pscore_components,
        }

        corr_img = torch.cat([self._proj_raw(f) for f in corr_feats], dim=1)
        corr_prefix = torch.cat([corr_img, text_emb], dim=1)
        if getattr(self, "capture_debug", False):
            self.debug_corrected_prefix = corr_prefix.detach().clone()

        # ---- LLM correct: selected vision tokens (+ text permanent group) ----
        sel_global = torch.cat([sel_per_image[i] + i * tpi for i in range(len(images))]) \
            if any(s.numel() for s in sel_per_image) else torch.zeros(0, dtype=torch.long, device=dev)
        tok_idx = sel_global
        if correct_text:
            tok_idx = torch.cat([sel_global, torch.arange(n_img, P, device=dev)])
        tok_idx = tok_idx.sort().values
        real_image_count = sum(bool(mask.any()) for mask in img_masks)
        self.last_recompute_stats = {
            "score_mode": score_mode,
            "llm_attention_query_group": llm_query_group,
            "llm_vision_weight": llm_vision_weight,
            "llm_attention_score_query_tokens": (
                int(next(iter(score_query_groups.values())).numel())
                if score_query_groups is not None
                else 0
            ),
            "llm_attention_score_key_tokens": (
                int(real_vision_indices.numel())
                if score_query_groups is not None
                else 0
            ),
            "real_images": real_image_count,
            "tokens_per_image": tpi,
            "vit_corrected_tokens": int(sel_global.numel()),
            "vit_total_real_tokens": tpi * real_image_count,
            "llm_corrected_query_tokens": int(tok_idx.numel()),
            "llm_prefix_tokens": P,
            "llm_valid_prefix_tokens": int(pad.sum().item()),
            "text_group_tokens": L if correct_text else 0,
        }
        if getattr(self, "capture_debug", False):
            self.debug_active_indices = tok_idx.detach().clone()
        if tok_idx.numel() > 0:
            cos_s, sin_s = cos[:, tok_idx], sin[:, tok_idx]
            x_sel = corr_prefix[:, tok_idx]
            mask_rows = mask4d[:, :, tok_idx, :]
            for i, layer in enumerate(self.llm_layers):
                x_sel, self.cache_feature = layer.prefill(
                    x_sel, tok_idx, self.cache_feature, f"llm{i}", cos_s, sin_s,
                    key_end=P, causal=False, attn_mask=mask_rows)
                if getattr(self, "capture_debug", False):
                    self.cache_feature[f"llm{i}_corrected_x"] = x_sel.detach().clone()
            xf = self.cache_feature["_x"].clone()
            xf[:, tok_idx] = x_sel.to(xf.dtype)
            self.cache_feature["_x"] = xf

        return self._generate_fast_from_cache(P, pad)

    @torch.inference_mode()
    def _generate_fast_from_cache(self, P: int, pad: torch.Tensor) -> torch.Tensor:
        from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb, repeat_kv
        # RoPE position for FAST tokens continues from the last VALID prefix position (padding does
        # not advance position_ids = cumsum(pad)-1), NOT from the padded length P.
        p_valid = int(pad.sum().item())
        kcache, vcache = [], []
        for i in range(len(self.llm_layers)):
            k, v = self.cache_feature[f"llm{i}_kv"][:, :, :P].unbind(3)
            kcache.append(k.clone()); vcache.append(v.clone())
        first_hidden = self.lm_norm(self.cache_feature["_x"][:, -1:])
        if getattr(self, "capture_debug", False):
            self.debug_fast_hidden = [first_hidden.detach().clone()]
        first = int(self.lm_head(first_hidden)[:, -1].argmax(-1))
        # LeRobot's stock sample_actions_fast_kv_cache always emits max_decoding_steps tokens,
        # including the tail after an EOS token. Match that fixed-length contract exactly: stopping
        # at EOS produces the same decoded action but a different token trace.
        gen = [first]
        cur = first
        for step in range(1, self.max_decoding_steps):
            cos, sin = self._rope(torch.tensor([p_valid - 1 + step], device=self.device))
            x = self._language_at_layer_scale(
                torch.tensor([[cur]], device=self.device)
            )
            n_fast = kcache[0].shape[2] - P + 1                       # keys beyond prefix (incl this one)
            current_pad = torch.cat(
                [
                    pad,
                    torch.ones(1, n_fast, dtype=torch.bool, device=self.device),
                ],
                dim=1,
            )
            # Preserve LeRobot's exact mask construction and tensor shape.
            key_mask = self.m._prepare_attention_masks_4d(
                current_pad.unsqueeze(1),
                dtype=x.dtype,
            )
            for i, layer in enumerate(self.llm_layers):
                attn = layer.self_attn
                h = layer.input_layernorm(x)
                q, k, v = attn._project_heads(h)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                kcache[i] = torch.cat([kcache[i], k], dim=2); vcache[i] = torch.cat([vcache[i], v], dim=2)
                kf = repeat_kv(kcache[i], attn.num_key_value_groups); vf = repeat_kv(vcache[i], attn.num_key_value_groups)
                o = F.scaled_dot_product_attention(
                    q, kf, vf, attn_mask=key_mask, scale=attn.scaling)
                o = o.transpose(1, 2).reshape(1, 1, -1)
                x = x + attn.o_proj(o); x = x + layer.mlp(layer.post_attention_layernorm(x))
            step_hidden = self.lm_norm(x)
            if getattr(self, "capture_debug", False):
                self.debug_fast_hidden.append(step_hidden.detach().clone())
            nxt = int(self.lm_head(step_hidden)[:, -1].argmax(-1))
            gen.append(nxt); cur = nxt
        return torch.tensor([gen], device=self.device, dtype=torch.long)

    def _detok(self, gen_tokens: torch.Tensor) -> np.ndarray:
        pad = torch.zeros(1, self.max_decoding_steps, device=self.device, dtype=torch.long)
        n = min(gen_tokens.shape[1], self.max_decoding_steps)
        pad[:, :n] = gen_tokens[:, :n]
        actions = self.pol.detokenize_actions(pad, action_horizon=self.n_action_steps, action_dim=self.action_dim)
        return actions

    def _features_exact(self, pixel_list: List[torch.Tensor]) -> List[torch.Tensor]:
        outs = []
        for i, px in enumerate(pixel_list):
            feat, self.cache_feature = self.siglip.approx_forward(px, self.cache_feature, f"img{i}")
            outs.append(self._project(feat))
        self.tokens_per_image = outs[0].shape[1]
        return outs

    def _features_progressive(self, pixel_list: List[torch.Tensor], num_groups: int,
                              base_factor: int, patch_keep: Optional[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Per image: approx on a low-res base, then correct arriving patches (all patches -- i.e.
        lossless -- unless a keep-subset is given). At full correction == _features_exact."""
        outs = []
        for i, px in enumerate(pixel_list):
            tag = f"img{i}"
            self.siglip.approx_forward(self._base_pixel(px, base_factor), self.cache_feature, tag)
            npatch = self.cache_feature[f"{tag}_layer0_kv"].shape[2]
            keep = patch_keep[i] if patch_keep is not None else \
                torch.arange(npatch, device=self.device, dtype=torch.long)
            feat, self.cache_feature = self.siglip.correct_forward(px, keep, self.cache_feature, tag)
            outs.append(self._project(feat))
        self.tokens_per_image = outs[0].shape[1]
        return outs

    # ------------------------------------------------------------------ injection into lerobot
    def _pixel_list(self, obs: Dict[str, Any]) -> List[torch.Tensor]:
        b = self.pre(obs)
        images, _ = self.pol._preprocess_images(b)      # config.image_features order
        return [px.to(self.device) for px in images]

    def _run_with_injected(self, obs: Dict[str, Any], features: List[torch.Tensor]) -> np.ndarray:
        """Call lerobot's own predict_action_chunk, but with embed_image overridden to hand back our
        (progressive) precomputed features in call order."""
        queue = list(features)

        def _fake_embed_image(img):
            return queue.pop(0)

        self.m.paligemma_with_expert.embed_image = _fake_embed_image
        try:
            b = self.pre(obs)
            action = self.pol.predict_action_chunk(b)          # [1, n_steps, action_dim]
            action = self.post(action.clone())
        finally:
            self.m.paligemma_with_expert.embed_image = self._orig_embed_image
            self.pol._action_queue.clear()
        return action[0].float().cpu().numpy()

    # ------------------------------------------------------------------ public API
    @torch.inference_mode()
    def predict_action_exact(self, obs: Dict[str, Any]) -> np.ndarray:
        """Progressive-vision approx on the true image (approx == exact at full res) + lerobot's
        stock LLM/FAST decode. Bit-exact with stock lerobot."""
        self.cache_feature = {}
        feats = self._features_exact(self._pixel_list(obs))
        return self._run_with_injected(obs, feats)

    @torch.inference_mode()
    def predict_action_progressive(self, obs: Dict[str, Any], num_groups: int = 4,
                                   base_factor: int = 4,
                                   patch_keep: Optional[List[torch.Tensor]] = None) -> np.ndarray:
        """Low-res base approx + per-group patch correct in the SigLIP tower, then lerobot's stock
        LLM/FAST decode on the corrected features. `patch_keep[i]` (indices) restricts which patches
        of image i are corrected (progressive compression); None => all patches (lossless)."""
        self.cache_feature = {}
        feats = self._features_progressive(self._pixel_list(obs), num_groups, base_factor, patch_keep)
        return self._run_with_injected(obs, feats)
