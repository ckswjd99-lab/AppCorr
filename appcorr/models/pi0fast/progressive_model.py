"""
progressive_model.py  (pi0-FAST)

pi0-FAST progressive-prefill model. The CORE progressive technique for pi0-FAST is
*progressive vision*: the SigLIP tower runs a low-res base approx and then corrects only the
arriving patch groups (ApproxCorrectSiglipBackbone, bit-exact at 100% correct). Everything
downstream of the vision features -- PaliGemma's bidirectional image+language prefix, the FAST
autoregressive action decode, the FAST detokenizer, the padded-sequence mask/position handling --
is left to lerobot's own (correct) `PI0FastPolicy` pipeline, into which we simply *inject* our
progressive vision features by temporarily overriding `embed_image`.

Why not fork the LLM prefill too (as for OpenVLA/OFT)? pi0-FAST's PaliGemma prefix is
BIDIRECTIONAL (image+language attend to each other), so a single-pass causal / block-causal chunked
prefill is NOT lossless (measured: strict-causal +12% CE; the lossless block-causal pattern needs
text<->image bidirectionality, which is not single-pass streamable). The bidirectional prefix is
therefore run in full by lerobot; the progressive saving is entirely in the vision tower. Injecting
features (rather than reimplementing generation) also guarantees bit-exact parity with the stock
model -- at 100% vision correction, `predict_action_progressive == predict_action_exact ==
stock lerobot`.

Double-scaling note: transformers>=4.57 `GemmaModel.forward` scales inputs_embeds by sqrt(hidden),
but lerobot's `embed_prefix_fast` ALSO pre-scales them -- a double-scaling that collapses the model
(see memory `project_pi0fast_double_scaling_bug`). `install_gemma_scaling_fix()` (applied at import)
neutralizes the extra HF scaling; it is required for any pi0-FAST inference under transformers>=4.57.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


_SCALING_FIX_INSTALLED = False


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


class Pi0FastProgressiveModel:
    def __init__(self, checkpoint: str, device: torch.device,
                 stats_dataset_repo: str = "HuggingFaceVLA/libero"):
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from appcorr.models.pi0fast.siglip_vision import ApproxCorrectSiglipBackbone

        install_gemma_scaling_fix()
        self.device = device
        pol = PI0FastPolicy.from_pretrained(checkpoint).to(device).eval()
        # Consistent float32 (native mixed precision throws Float-vs-BFloat16 in SigLIP layer_norm).
        pol.model.paligemma_with_expert.to_bfloat16_for_selected_params("float32")
        self.pol = pol
        self.m = pol.model
        self.pg = self.m.paligemma_with_expert.paligemma
        self.pre, self.post = make_pre_post_processors(
            pol.config, pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}})

        self.projector = self.pg.model.multi_modal_projector
        self.img_scale = self.pg.config.text_config.hidden_size ** 0.5   # get_image_features /sqrt

        # progressive-vision fork (the CORE technique)
        self.siglip = ApproxCorrectSiglipBackbone(self.pg.model.vision_tower.vision_model).to(device)

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

    def _project(self, feats: torch.Tensor) -> torch.Tensor:
        """last_hidden_state -> multi_modal_projector -> /sqrt(hidden)  (== get_image_features)."""
        return self.projector(feats) / self.img_scale

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
