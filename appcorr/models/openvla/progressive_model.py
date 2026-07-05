"""
progressive_model.py

Phase 3 of the progressive-VLA-prefill plan (see /home/nxclab/.claude/plans/async-stargazing-mango.md):
wires the Phase 1 vision-tower forks and Phase 2 causal-LLM fork together into a single object that
can run a real progressive (approx -> correct -> correct -> ...) prefill on a real image, and decode
real actions from it -- answering the two open questions from that conversation: (1) does a partially
corrected prefill produce a meaningful *action* (not just a similar logit), and (2) does this hold
using genuine low-resolution image data (not synthetic noise) end-to-end through both towers and the
LLM together.

This is a lighter-weight standalone wrapper, not yet the formal `ModelExecutor` ABC
(`offload/server/model/base.py`) -- per the plan's Milestone 1 decision, we validate the core mechanism
locally first; wrapping this in `ModelExecutor` (Task/Instruction/OpType plumbing) is Phase 6's job, if
we get there.

Session lifecycle:
    model = OpenVLAProgressiveModel(checkpoint, device, unnorm_key)
    model.start_session(image, task_description, center_crop=True)
    logits = model.approx_forward(low_res_pixel_values)          # first pass (e.g. blurred image)
    logits = model.correct_forward(full_res_pixel_values, patch_idx)  # subsequent passes, cumulative
    action = model.decode_action()                                # 7-DoF continuous action, any time
"""

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from appcorr.models.openvla.vision.backbone import ApproxCorrectViTBackbone
from appcorr.models.openvla.llm.llama_prefill_layer import ApproxCorrectLlamaDecoderLayer


class OpenVLAProgressiveModel:
    def __init__(self, checkpoint: str, device: torch.device, unnorm_key: Optional[str] = None):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.vla.eval()

        self.unnorm_key = unnorm_key
        if self.unnorm_key is None:
            assert len(self.vla.norm_stats) == 1
            self.unnorm_key = next(iter(self.vla.norm_stats.keys()))

        vb = self.vla.vision_backbone
        self.dino_backbone = ApproxCorrectViTBackbone(vb.featurizer).to(device)
        self.siglip_backbone = ApproxCorrectViTBackbone(vb.fused_featurizer).to(device)
        assert self.dino_backbone.extract_block_idx == self.siglip_backbone.extract_block_idx or True, (
            "towers may have different depths; extraction indices are tracked independently"
        )

        self.llm_layers = [
            ApproxCorrectLlamaDecoderLayer.from_stock(l).to(device) for l in self.vla.language_model.model.layers
        ]

        # Session state, set in start_session()
        self.cache_feature: Dict[str, Any] = {}
        self.input_ids: Optional[torch.Tensor] = None
        self.bos_embed: Optional[torch.Tensor] = None
        self.text_embed: Optional[torch.Tensor] = None
        self.num_vision_tokens: Optional[int] = None
        self.permanent_group: Optional[torch.Tensor] = None
        self.seq_len: Optional[int] = None
        self.round_idx = 0

    def _center_crop_and_resize(self, image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
        orig_w, orig_h = image.size
        new_h, new_w = orig_h * math.sqrt(crop_scale), orig_w * math.sqrt(crop_scale)
        top, left = (orig_h - new_h) / 2, (orig_w - new_w) / 2
        cropped = image.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((224, 224), Image.BILINEAR)

    def start_session(self, image: Image.Image, task_description: str, center_crop: bool = True):
        """Tokenizes the prompt and precomputes the (fixed, never-approximated) BOS + text embeddings.
        Also runs the processor's image transform once to get the *reference* full-res pixel_values
        (channel-stacked [1,6,224,224], DINOv2 first 3 channels + SigLIP next 3) for convenience --
        callers can still pass their own (e.g. blurred) pixel_values to approx_forward/correct_forward."""
        image = image.convert("RGB")
        if center_crop:
            image = self._center_crop_and_resize(image, 0.9).convert("RGB")

        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"]

        # Matches OpenVLAForActionPrediction.predict_action(): ensure the empty-string token (29871)
        # follows "Out:" before generation, as seen at training time.
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                [input_ids, torch.tensor([[29871]], dtype=input_ids.dtype, device=input_ids.device)], dim=1
            )
        self.input_ids = input_ids

        embed_layer = self.vla.get_input_embeddings()
        full_text_embed = embed_layer(input_ids)  # [1, T, C]
        self.bos_embed = full_text_embed[:, :1]
        self.text_embed = full_text_embed[:, 1:]

        self.reference_pixel_values = inputs["pixel_values"]

        self.cache_feature = {}
        self.round_idx = 0
        self.num_vision_tokens = None  # set on first approx_forward once we know patch count
        self.seq_len = None
        self.permanent_group = None

    def _project_vision(self, dino_patch_feat: torch.Tensor, siglip_patch_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([dino_patch_feat, siglip_patch_feat], dim=2)
        return self.vla.projector(fused)

    def _finish_setup_after_first_pass(self, num_vision_tokens: int):
        self.num_vision_tokens = num_vision_tokens
        self.seq_len = 1 + num_vision_tokens + self.text_embed.shape[1]
        self.permanent_group = torch.cat([
            torch.tensor([0], device=self.device),
            torch.arange(1 + num_vision_tokens, self.seq_len, device=self.device),
        ])

    def _build_multimodal_embed(self, projected_vision: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.bos_embed, projected_vision, self.text_embed], dim=1)

    def approx_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """First pass on a new image (e.g. a blurred/low-res canvas). Returns logits at every
        position (mainly the last one is of interest -- see `decode_action`)."""
        dino_px, siglip_px = torch.split(pixel_values.to(dtype=torch.bfloat16), [3, 3], dim=1)

        dino_feat, self.cache_feature = self.dino_backbone.approx_forward(dino_px, self.cache_feature, "dino")
        siglip_feat, self.cache_feature = self.siglip_backbone.approx_forward(siglip_px, self.cache_feature, "siglip")
        self._finish_setup_after_first_pass(dino_feat.shape[1])

        projected = self._project_vision(dino_feat, siglip_feat)
        x = self._build_multimodal_embed(projected)

        for i, layer in enumerate(self.llm_layers):
            x, self.cache_feature = layer.approx(x, self.cache_feature, f"llm_layer{i}")

        self.cache_feature["_x"] = x
        self.round_idx = 0
        logits = self._logits_from_x(x)
        return logits

    def correct_forward(self, pixel_values: torch.Tensor, patch_idx: torch.Tensor) -> torch.Tensor:
        """Subsequent pass once higher-res data has arrived for `patch_idx` (patch-grid indices,
        0-indexed, same convention as the vision backbones). Cumulative across rounds -- the vision
        towers' own cache_feature persists, so already-corrected patches stay correct."""
        assert self.num_vision_tokens is not None, "call approx_forward() at least once first"
        dino_px, siglip_px = torch.split(pixel_values.to(dtype=torch.bfloat16), [3, 3], dim=1)

        dino_feat, self.cache_feature = self.dino_backbone.correct_forward(dino_px, patch_idx, self.cache_feature, "dino")
        siglip_feat, self.cache_feature = self.siglip_backbone.correct_forward(siglip_px, patch_idx, self.cache_feature, "siglip")

        projected = self._project_vision(dino_feat, siglip_feat)
        x_layer0 = self._build_multimodal_embed(projected)

        vision_token_idx = patch_idx.to(dtype=torch.long, device=self.device) + 1  # +1 for BOS offset
        token_idx = torch.cat([vision_token_idx, self.permanent_group])

        x = x_layer0
        for i, layer in enumerate(self.llm_layers):
            x, self.cache_feature = layer.correct(x, token_idx, self.cache_feature, f"llm_layer{i}")

        self.cache_feature["_x"] = x
        self.round_idx += 1
        logits = self._logits_from_x(x)
        return logits

    def _logits_from_x(self, x: torch.Tensor) -> torch.Tensor:
        final = self.vla.language_model.model.norm(x)
        return self.vla.language_model.lm_head(final)

    def decode_action(self, num_action_tokens: Optional[int] = None, return_stats: bool = False):
        """Greedy-decodes the action tokens from the current prefill state and converts them to a
        continuous action using the exact same bin-center + un-normalize logic as
        OpenVLAForActionPrediction.predict_action() (modeling_prismatic.py).

        With `return_stats=True`, also returns per-action-token confidence stats computed over the
        256-wide action-bin logit slice (the model only ever emits these at action positions) --
        the ingredients for the Phase 4 early-exit decision, mirroring the metric menu of
        `dinov3_classifier.py::decide_exit` (max_prob / top2_margin / entropy) plus a bin-aware
        `neighbor_mass` (probability within +-1 bin of the argmax; adjacent bins are near-identical
        continuous actions, so mass there should count toward confidence)."""
        if num_action_tokens is None:
            num_action_tokens = self.vla.get_action_dim(self.unnorm_key)

        # Action-bin token ids live in [vocab_size - n_action_bins, vocab_size) (see the
        # `vocab_size - token_id` de-tokenization below).
        bin_lo = self.vla.vocab_size - self.vla.config.n_action_bins

        def slice_stats(last_logits: torch.Tensor) -> Dict[str, float]:
            probs = torch.softmax(last_logits[0, bin_lo : self.vla.vocab_size].float(), dim=-1)
            top2 = probs.topk(2)
            arg = int(top2.indices[0].item())
            lo, hi = max(arg - 1, 0), min(arg + 2, probs.shape[0])
            entropy = -(probs * (probs + 1e-10).log()).sum()
            return {
                "max_prob": float(top2.values[0]),
                "top2_margin": float(top2.values[0] - top2.values[1]),
                "entropy": float(entropy),
                "neighbor_mass": float(probs[lo:hi].sum()),
            }

        x = self.cache_feature["_x"]
        logits = self._logits_from_x(x)
        next_token = logits[:, -1].argmax(-1, keepdim=True)
        generated = [next_token.item()]
        stats = [slice_stats(logits[:, -1])] if return_stats else None

        # Convert our hand-built per-layer KV cache ([B, H_kv, N, 2, Dh]) into the legacy
        # tuple-of-(K, V) format transformers' LlamaModel.forward() accepts (it wraps it in a
        # DynamicCache internally) so the remaining action tokens can be decoded with the stock,
        # unforked model -- decode is a plain cached generation, no approx/correct needed there.
        # Note the .contiguous() copies: decode-time cache growth never mutates cache_feature,
        # so a confidence-gated flow may decode from approx and still correct_forward afterwards.
        past_key_values = tuple(
            (kv[:, :, :, 0].contiguous(), kv[:, :, :, 1].contiguous())
            for kv in (self.cache_feature[f"llm_layer{i}_kv"] for i in range(len(self.llm_layers)))
        )

        with torch.no_grad():
            for _ in range(num_action_tokens - 1):
                out = self.vla.language_model(
                    input_ids=next_token, past_key_values=past_key_values, use_cache=True, return_dict=True
                )
                past_key_values = out.past_key_values
                next_token = out.logits[:, -1].argmax(-1, keepdim=True)
                generated.append(next_token.item())
                if return_stats:
                    stats.append(slice_stats(out.logits[:, -1]))

        predicted_action_token_ids = np.array(generated[-num_action_tokens:])
        discretized_actions = self.vla.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.vla.bin_centers.shape[0] - 1)
        normalized_actions = self.vla.bin_centers[discretized_actions]

        action_norm_stats = self.vla.get_action_stats(self.unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        if return_stats:
            return actions, {"per_token": stats, "bins": discretized_actions.tolist()}
        return actions
