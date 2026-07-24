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
        self.hidden_size = self.pg.config.text_config.hidden_size
        self.img_scale = self.hidden_size ** 0.5   # get_image_features /sqrt

        # progressive-vision fork (the CORE technique)
        self.siglip = ApproxCorrectSiglipBackbone(self.pg.model.vision_tower.vision_model).to(device)

        # LLM (Gemma) fork + handles, for partial-token LLM correct
        from appcorr.models.pi0fast.gemma_prefill_layer import ApproxCorrectGemmaDecoderLayer
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

    def _project(self, feats: torch.Tensor) -> torch.Tensor:
        """last_hidden_state -> multi_modal_projector -> /sqrt(hidden)  (== get_image_features)."""
        return self.projector(feats) / self.img_scale

    def _proj_raw(self, feats: torch.Tensor) -> torch.Tensor:
        """Image embed as the Gemma LAYER sees it (== get_image_features * sqrt(hidden) == raw
        projector output). Our fork layers do no internal sqrt scaling, so we feed this directly;
        text/FAST are fed embed_tokens * sqrt(hidden)."""
        return self.projector(feats)

    def _rope(self, positions: torch.Tensor):
        dummy = torch.zeros(1, positions.shape[-1], self.hidden_size, device=self.device)
        return self.rotary_emb(dummy, positions.unsqueeze(0) if positions.dim() == 1 else positions)

    # ------------------------------------------------------------------ partial-token (ViT + LLM)
    @torch.inference_mode()
    def predict_action_partial_token(self, obs: Dict[str, Any], keep: float = 0.5,
                                     base_factor: int = 4, correct_text: bool = True) -> np.ndarray:
        """DINOv3-style partial-token progressive prefill on BOTH the SigLIP ViT and the Gemma LLM.

        one approx + one pscore-selected correct:
          vision: SigLIP base approx (+pscore = residual*avg_attn) -> correct only the top-`keep`
                  patches per real image.
          LLM:    bidirectional approx on the base vision features + text -> correct ONLY the selected
                  vision tokens' K/V (+ the text 'permanent group' so it re-attends to them). Non-
                  selected vision tokens keep their base LLM K/V. Then FAST-decode from the corrected
                  prefix. At keep=1.0 (+correct_text) this reproduces the stock prefix -> stock action.
        Returns the (still-normalized) action chunk, matching PI0FastPolicy.predict_action_chunk.
        """
        from lerobot.policies.pi0_fast.modeling_pi0_fast import (
            OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)
        dev = self.device
        sqrtH = self.img_scale
        b = self.pre(obs)
        images, img_masks = self.pol._preprocess_images(b)          # 3 imgs, masks [T,T,F]
        text_ids = b[OBS_LANGUAGE_TOKENS]
        text_mask = b[OBS_LANGUAGE_ATTENTION_MASK]
        bos = torch.full((1, 1), self.tok.bos_token_id, device=dev, dtype=text_ids.dtype)
        text_ids = torch.cat([text_ids, bos], dim=1)
        text_mask = torch.cat([text_mask, torch.ones((1, 1), dtype=text_mask.dtype, device=dev)], dim=1)

        # ---- vision: base approx (+pscore) then correct top-keep patches per real image ----
        self.cache_feature = {}
        base_feats, corr_feats, sel_per_image = [], [], []
        for i, img in enumerate(images):
            real = img_masks[i].any().item()
            bf, self.cache_feature = self.siglip.approx_forward(
                self._base_pixel(img, base_factor), self.cache_feature, f"img{i}", pscore=real)
            base_feats.append(bf)
            if real:
                ps = self.siglip.get_pscore(self.cache_feature, f"img{i}")[0]     # [256]
                npn = ps.shape[0]
                k = max(1, min(int(round(keep * npn)), npn))
                sel = torch.topk(ps.float(), k, largest=True).indices.sort().values
                cf, self.cache_feature = self.siglip.correct_forward(img, sel, self.cache_feature, f"img{i}")
                corr_feats.append(cf); sel_per_image.append(sel)
            else:
                corr_feats.append(bf); sel_per_image.append(torch.zeros(0, dtype=torch.long, device=dev))
        tpi = base_feats[0].shape[1]                                  # tokens per image (256)
        n_img = tpi * len(images)

        # ---- build prefix embeds at the Gemma-layer scale ----
        approx_img = torch.cat([self._proj_raw(f) for f in base_feats], dim=1)      # [1, n_img, D]
        corr_img = torch.cat([self._proj_raw(f) for f in corr_feats], dim=1)        # [1, n_img, D]
        text_emb = self.embed_tokens(text_ids) * sqrtH                              # [1, L, D]
        approx_prefix = torch.cat([approx_img, text_emb], dim=1)
        corr_prefix = torch.cat([corr_img, text_emb], dim=1)
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

        # ---- LLM approx (bidirectional) on base vision features ----
        x = approx_prefix
        for i, layer in enumerate(self.llm_layers):
            x, self.cache_feature = layer.approx(x, self.cache_feature, f"llm{i}", cos, sin,
                                                 causal=False, attn_mask=mask4d)
        self.cache_feature["_x"] = x

        # ---- LLM correct: selected vision tokens (+ text permanent group) ----
        sel_global = torch.cat([sel_per_image[i] + i * tpi for i in range(len(images))]) \
            if any(s.numel() for s in sel_per_image) else torch.zeros(0, dtype=torch.long, device=dev)
        tok_idx = sel_global
        if correct_text:
            tok_idx = torch.cat([sel_global, torch.arange(n_img, P, device=dev)])
        tok_idx = tok_idx.sort().values
        if tok_idx.numel() > 0:
            cos_s, sin_s = cos[:, tok_idx], sin[:, tok_idx]
            x_sel = corr_prefix[:, tok_idx]
            mask_rows = mask4d[:, :, tok_idx, :]
            for i, layer in enumerate(self.llm_layers):
                x_sel, self.cache_feature = layer.prefill(
                    x_sel, tok_idx, self.cache_feature, f"llm{i}", cos_s, sin_s,
                    key_end=P, causal=False, attn_mask=mask_rows)
            xf = self.cache_feature["_x"].clone()
            xf[:, tok_idx] = x_sel.to(xf.dtype)
            self.cache_feature["_x"] = xf

        gen = self._generate_fast_from_cache(P, pad)
        return self._detok(gen)

    @torch.inference_mode()
    def _generate_fast_from_cache(self, P: int, pad: torch.Tensor) -> torch.Tensor:
        from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb, repeat_kv
        eos = self.tok.eos_token_id
        # additive key mask over the PREFIX: -inf for padding keys (empty camera + text pad) so the
        # generated FAST tokens never attend to them (matches lerobot's prefix_pad_masks).
        neg = torch.finfo(torch.float32).min
        prefix_key_mask = torch.where(pad.view(1, P), torch.zeros(1, P, device=self.device),
                                      torch.full((1, P), neg, device=self.device))  # [1, P]
        # RoPE position for FAST tokens continues from the last VALID prefix position (padding does
        # not advance position_ids = cumsum(pad)-1), NOT from the padded length P.
        p_valid = int(pad.sum().item())
        kcache, vcache = [], []
        for i in range(len(self.llm_layers)):
            k, v = self.cache_feature[f"llm{i}_kv"][:, :, :P].unbind(3)
            kcache.append(k.clone()); vcache.append(v.clone())
        first = int(self.lm_head(self.lm_norm(self.cache_feature["_x"][:, -1:]))[:, -1].argmax(-1))
        gen = [] if first == eos else [first]
        cur = first
        for step in range(1, self.max_decoding_steps):
            if not gen:
                break
            cos, sin = self._rope(torch.tensor([p_valid - 1 + step], device=self.device))
            x = self.embed_tokens(torch.tensor([[cur]], device=self.device)) * self.img_scale
            n_fast = kcache[0].shape[2] - P + 1                       # keys beyond prefix (incl this one)
            key_mask = torch.cat([prefix_key_mask, torch.zeros(1, n_fast, device=self.device)], dim=1)
            key_mask = key_mask.view(1, 1, 1, P + n_fast).to(x.dtype)
            for i, layer in enumerate(self.llm_layers):
                attn = layer.self_attn
                h = layer.input_layernorm(x)
                q, k, v = attn._project_heads(h)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                kcache[i] = torch.cat([kcache[i], k], dim=2); vcache[i] = torch.cat([vcache[i], v], dim=2)
                kf = repeat_kv(kcache[i], attn.num_key_value_groups); vf = repeat_kv(vcache[i], attn.num_key_value_groups)
                o = F.scaled_dot_product_attention(q, kf, vf, attn_mask=key_mask, scale=attn.scaling)
                o = o.transpose(1, 2).reshape(1, 1, -1)
                x = x + attn.o_proj(o); x = x + layer.mlp(layer.post_attention_layernorm(x))
            nxt = int(self.lm_head(self.lm_norm(x))[:, -1].argmax(-1))
            if nxt == eos:
                break
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
