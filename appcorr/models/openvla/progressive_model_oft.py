"""
progressive_model_oft.py

OpenVLA-OFT progressive-prefill model, built by DIRECT weight loading (no openvla-oft package, no
transformers fork, no tensorflow -- all of which are a version-conflict rabbit hole on this box).

Reuses the vanilla OpenVLA HF class purely as a weight container: the OFT base weights
(vision_backbone.featurizer/fused_featurizer, projector, language_model) map onto it with 0
missing/0 unexpected. The OFT-specific continuous-action stack (L1RegressionActionHead,
ProprioProjector) is re-defined in oft_heads.py and loaded from the released .pt files.

OFT-LIBERO specifics reproduced here:
  - 2 input images (agentview + wrist), each -> DINOv2+SigLIP -> concat feat-dim -> concat images
    -> projector -> 512 vision tokens.
  - 1 proprio token (8-dim eef_pos(3)+axisangle(3)+gripper(2), bounds_q99-normalized -> projector).
  - Sequence: [BOS, vision(512), proprio(1), prompt, 29871, action x 56].  The 56 action tokens
    (NUM_ACTIONS_CHUNK=8 x ACTION_DIM=7) attend BIDIRECTIONALLY over the whole sequence (parallel
    decoding); their hidden states -> L1 head -> (8,7) normalized actions -> bounds_q99-unnormalize.

Two forward modes share the same code path (approx/correct forks):
  - exact / full:  vision approx on the TRUE image (approx == exact at 100%), full-depth LLM prefill.
  - progressive:   vision approx on a low-res base + per-group correct + LLM chunked prefill.

The action-token forward is a single non-causal ("full") attention pass over the whole sequence,
implemented via the LLM attention's `.correct()` with an all-allowed mask.
"""

import glob
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from appcorr.models.openvla.vision.backbone import ApproxCorrectViTBackbone
from appcorr.models.openvla.llm.llama_prefill_layer import ApproxCorrectLlamaDecoderLayer
from appcorr.models.openvla.oft_heads import (
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    load_action_head,
    load_proprio_projector,
)

EMPTY_TOKEN = 29871           # the '' token appended after "Out:"
ACTION_PLACEHOLDER_ID = 1     # OFT appends `ones(...)` as action placeholders (embed is overwritten? no -- used as-is)


def quat2axisangle(quat):
    """xyzw quaternion -> axis-angle (3,). Matches robosuite/OFT's transform_utils.quat2axisangle."""
    quat = np.asarray(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat = quat / np.linalg.norm(quat)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if den < 1e-8:
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def _normalize_bounds_q99(x, q01, q99):
    x = np.asarray(x, dtype=np.float64)
    q01 = np.asarray(q01, dtype=np.float64)
    q99 = np.asarray(q99, dtype=np.float64)
    return np.clip(2.0 * (x - q01) / (q99 - q01 + 1e-8) - 1.0, -1.0, 1.0)


def _unnormalize_bounds_q99(x, q01, q99, mask=None):
    x = np.asarray(x, dtype=np.float64)
    q01 = np.asarray(q01, dtype=np.float64)
    q99 = np.asarray(q99, dtype=np.float64)
    out = 0.5 * (x + 1.0) * (q99 - q01) + q01
    if mask is not None:
        out = np.where(mask, out, x)  # unnormalized only where mask (e.g. skip gripper dim)
    return out


class OFTProgressiveModel:
    def __init__(self, checkpoint: str, device: torch.device, oft_snapshot_dir: str,
                 unnorm_key: Optional[str] = None):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = device
        # Vanilla OpenVLA class as weight container; then overwrite with OFT weights.
        base_ckpt = "openvla/openvla-7b-finetuned-libero-spatial"
        self.processor = AutoProcessor.from_pretrained(base_ckpt, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            base_ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        from safetensors.torch import load_file
        sd = {}
        for f in sorted(glob.glob(os.path.join(oft_snapshot_dir, "model-*.safetensors"))):
            sd.update(load_file(f))
        res = self.vla.load_state_dict(sd, strict=False)
        assert not res.missing_keys and not res.unexpected_keys, \
            f"OFT weight load mismatch: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
        self.vla = self.vla.to(device).eval()

        # OFT heads
        ah = glob.glob(os.path.join(oft_snapshot_dir, "action_head--*.pt"))[0]
        pp = glob.glob(os.path.join(oft_snapshot_dir, "proprio_projector--*.pt"))[0]
        self.action_head = load_action_head(ah, device)
        self.proprio_projector = load_proprio_projector(pp, device)

        # OFT dataset statistics for (un)normalization
        import json
        stats = json.load(open(os.path.join(oft_snapshot_dir, "dataset_statistics.json")))
        self.unnorm_key = unnorm_key or next(iter(stats.keys()))
        self.stats = stats[self.unnorm_key]

        # Progressive forks (shared across the 2 images / all layers)
        vb = self.vla.vision_backbone
        self.dino_backbone = ApproxCorrectViTBackbone(vb.featurizer).to(device)
        self.siglip_backbone = ApproxCorrectViTBackbone(vb.fused_featurizer).to(device)
        self.llm_layers = [ApproxCorrectLlamaDecoderLayer.from_stock(l).to(device)
                           for l in self.vla.language_model.model.layers]

        self.cache_feature: Dict[str, Any] = {}
        self.num_patches_per_image = None
        self.last_progressive_trace: List[Dict[str, Any]] = []

    # ---- preprocessing ----
    def _pixel_values_for_image(self, img_np: np.ndarray) -> torch.Tensor:
        """PIL resize(224)+center-crop(0.9) then the vanilla processor's DINO+SigLIP transform.
        Returns [1, 6, H, W] (dino 3ch + siglip 3ch)."""
        image = Image.fromarray(img_np).convert("RGB")
        w, h = image.size
        import math
        nh, nw = h * math.sqrt(0.9), w * math.sqrt(0.9)
        image = image.crop(((w - nw) / 2, (h - nh) / 2, (w + nw) / 2, (h + nh) / 2)).resize((224, 224), Image.BILINEAR)
        px = self.processor.image_processor(image, return_tensors="pt")["pixel_values"]
        return px.to(self.device, dtype=torch.bfloat16)

    def _dino_siglip_split(self, pixel_values_6ch: torch.Tensor):
        # OpenVLA stacks [dino(3), siglip(3)] on the channel dim.
        return pixel_values_6ch[:, :3], pixel_values_6ch[:, 3:]

    def start_session(self, task_label: str):
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        input_ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        if input_ids[0, -1].item() != EMPTY_TOKEN:
            input_ids = torch.cat(
                [input_ids, torch.tensor([[EMPTY_TOKEN]], device=self.device, dtype=input_ids.dtype)], dim=1
            )
        self.input_ids = input_ids  # [1, T_text]  (includes BOS at pos 0)
        embed = self.vla.get_input_embeddings()
        full = embed(input_ids)
        self.bos_embed = full[:, :1]            # [1,1,D]
        self.text_embed = full[:, 1:]           # [1, T_text-1, D]
        self.cache_feature = {}
        self.last_progressive_trace = []

    # ---- vision ----
    def _vision_tokens(self, images_np: List[np.ndarray], correct: bool = False,
                       patch_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run both towers on each of the 2 images (approx or correct), concat -> projector -> [1,512,D]."""
        per_image = []
        for i, img in enumerate(images_np):
            px = self._pixel_values_for_image(img)
            dino_px, siglip_px = self._dino_siglip_split(px)
            if not correct:
                dfeat, self.cache_feature = self.dino_backbone.approx_forward(dino_px, self.cache_feature, f"dino{i}")
                sfeat, self.cache_feature = self.siglip_backbone.approx_forward(siglip_px, self.cache_feature, f"siglip{i}")
            else:
                dfeat, self.cache_feature = self.dino_backbone.correct_forward(dino_px, patch_idx, self.cache_feature, f"dino{i}")
                sfeat, self.cache_feature = self.siglip_backbone.correct_forward(siglip_px, patch_idx, self.cache_feature, f"siglip{i}")
            per_image.append(torch.cat([dfeat, sfeat], dim=2))  # [1, 256, D_dino+D_siglip]
        patches = torch.cat(per_image, dim=1)  # [1, 512, D_dino+D_siglip]
        self.num_patches_per_image = per_image[0].shape[1]
        return self.vla.projector(patches)     # [1, 512, llm_dim]

    def _proprio_token(self, proprio_raw: np.ndarray) -> torch.Tensor:
        pst = self.stats["proprio"]
        pnorm = _normalize_bounds_q99(proprio_raw, pst["q01"], pst["q99"])
        pt = torch.tensor(pnorm, device=self.device, dtype=torch.bfloat16).reshape(1, -1)
        return self.proprio_projector(pt).unsqueeze(1)  # [1,1,D]

    # ---- exact (full) action prediction ----
    @torch.inference_mode()
    def predict_action_exact(self, images_np: List[np.ndarray], proprio_raw: np.ndarray) -> np.ndarray:
        vision = self._vision_tokens(images_np, correct=False)          # [1,512,D]
        proprio = self._proprio_token(proprio_raw)                      # [1,1,D]
        prefix = torch.cat([self.bos_embed, vision, proprio, self.text_embed], dim=1)  # [1, P, D]
        P = prefix.shape[1]
        # OFT zeros out the action-token embeddings (`input_embeddings * ~all_actions_mask`); the
        # action hidden states are produced purely from attention over vision/proprio/text.
        n_act = NUM_ACTIONS_CHUNK * ACTION_DIM
        act_embed = torch.zeros((1, n_act, prefix.shape[2]), device=self.device, dtype=prefix.dtype)
        x = torch.cat([prefix, act_embed], dim=1)   # [1, P+56, D]
        N = x.shape[1]
        # Full forward through all layers with OFT attention pattern:
        #   prefix positions: causal;  action positions: attend to everything (bidirectional).
        attn_mask = self._oft_attention_mask(P, N, x.dtype)
        pos_ids = torch.arange(N, device=self.device).unsqueeze(0)
        cos, sin = self.llm_layers[0].self_attn.rotary_emb(x, pos_ids)
        for layer in self.llm_layers:
            x = self._exact_layer_forward(layer, x, attn_mask, cos, sin)
        x = self.vla.language_model.model.norm(x)
        action_hidden = x[:, P:P + n_act]                # [1,56,D]
        norm_actions = self.action_head.predict_action(action_hidden)  # [1,8,7]
        return self._unnorm_actions(norm_actions[0].float().cpu().numpy())

    def _oft_attention_mask(self, P: int, N: int, dtype) -> torch.Tensor:
        """[1,1,N,N] additive mask: causal for the first P (prefix) query rows, all-visible for the
        action query rows (they see the whole sequence, bidirectionally)."""
        allowed = torch.zeros((N, N), dtype=torch.bool, device=self.device)
        # prefix rows: causal
        idx = torch.arange(N, device=self.device)
        allowed[:P] = idx.unsqueeze(0) <= idx[:P].unsqueeze(1)
        # action rows: everything
        allowed[P:] = True
        mask = torch.zeros((N, N), dtype=dtype, device=self.device)
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        return mask.view(1, 1, N, N)

    def _exact_layer_forward(self, layer, x, attn_mask, cos, sin):
        """One decoder layer, full-sequence, with an explicit additive attention mask (no KV cache)."""
        attn = layer.self_attn
        h = layer.input_layernorm(x)
        B, N, _ = h.shape
        q = attn.q_proj(h).view(B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(B, N, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(B, N, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, attn.num_key_value_groups)
        v = repeat_kv(v, attn.num_key_value_groups)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(B, N, -1)
        x = x + attn.o_proj(o)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return x

    # ---- progressive (chunked-prefill) action prediction ----
    def _prefill_block(self, x0: torch.Tensor, token_idx: torch.Tensor, causal: bool, key_end: int):
        """Prefill `token_idx` through all LLM layers (O(Q) path), scattering top-layer states into
        cache['_x']. causal=True for the prefix, causal=False for the bidirectional action block."""
        pos = token_idx.unsqueeze(0)
        cos, sin = self.llm_layers[0].self_attn.rotary_emb(x0, pos)
        x_sel = x0[:, token_idx]
        for i, layer in enumerate(self.llm_layers):
            x_sel, self.cache_feature = layer.prefill(
                x_sel, token_idx, self.cache_feature, f"llm_layer{i}", cos=cos, sin=sin,
                key_end=key_end, causal=causal,
            )
        xf = self.cache_feature["_x"].clone()
        xf[:, token_idx] = x_sel.to(xf.dtype)
        self.cache_feature["_x"] = xf

    def _vision_base(
        self,
        img_np: np.ndarray,
        tag_i: int,
        base_img_np: Optional[np.ndarray],
    ):
        """Initialize one camera's stateful ViTs and return its 256 base tokens.

        The exact image is also prepared here for later group corrections.  If
        ``base_img_np`` is absent, the base itself is exact and no correction is
        necessary; this is the grouped-full parity path.
        """
        full_px = self._pixel_values_for_image(img_np)
        dino_px, siglip_px = self._dino_siglip_split(full_px)
        dtag, stag = f"dino{tag_i}", f"siglip{tag_i}"
        if base_img_np is not None:
            bpx = self._pixel_values_for_image(base_img_np)
            base_dino_px, base_siglip_px = self._dino_siglip_split(bpx)
        else:
            base_dino_px, base_siglip_px = dino_px, siglip_px
        dfeat, self.cache_feature = self.dino_backbone.approx_forward(
            base_dino_px,
            self.cache_feature,
            dtag,
        )
        sfeat, self.cache_feature = self.siglip_backbone.approx_forward(
            base_siglip_px,
            self.cache_feature,
            stag,
        )
        projected = self.vla.projector(torch.cat([dfeat, sfeat], dim=2))
        return projected, dino_px, siglip_px

    def _vision_correct_group(
        self,
        dino_px: torch.Tensor,
        siglip_px: torch.Tensor,
        tag_i: int,
        patch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Correct one camera's selected patches and project the fresh rows."""
        dfeat, self.cache_feature = self.dino_backbone.correct_forward(
            dino_px,
            patch_idx,
            self.cache_feature,
            f"dino{tag_i}",
        )
        sfeat, self.cache_feature = self.siglip_backbone.correct_forward(
            siglip_px,
            patch_idx,
            self.cache_feature,
            f"siglip{tag_i}",
        )
        return self.vla.projector(torch.cat([dfeat, sfeat], dim=2))

    def _init_chunked_prefill(self, prefix_embed: torch.Tensor):
        """Allocate the fixed OFT KV/cache tensors and append no tokens yet."""
        batch, prefix_len, dim = prefix_embed.shape
        num_action_tokens = NUM_ACTIONS_CHUNK * ACTION_DIM
        total_len = prefix_len + num_action_tokens
        x0 = torch.cat(
            [
                prefix_embed,
                torch.zeros(
                    batch,
                    num_action_tokens,
                    dim,
                    device=self.device,
                    dtype=prefix_embed.dtype,
                ),
            ],
            dim=1,
        )
        attn0 = self.llm_layers[0].self_attn
        for layer_idx in range(len(self.llm_layers)):
            self.cache_feature[f"llm_layer{layer_idx}_kv"] = torch.zeros(
                batch,
                attn0.num_key_value_heads,
                total_len,
                2,
                attn0.head_dim,
                device=self.device,
                dtype=x0.dtype,
            )
        self.cache_feature["_x"] = x0.clone()
        return x0, prefix_len, total_len

    def _prefill_suffix_and_actions(
        self,
        x0: torch.Tensor,
        prefix_len: int,
        total_len: int,
        vision_end: int,
    ) -> torch.Tensor:
        """Finish causal proprio/text, then run the stock bidirectional action block."""
        rest = torch.arange(
            vision_end,
            prefix_len,
            device=self.device,
        )
        if rest.numel() > 0:
            self._prefill_block(
                x0,
                rest,
                causal=True,
                key_end=prefix_len,
            )
        self._prefill_block(
            x0,
            torch.arange(prefix_len, total_len, device=self.device),
            causal=False,
            key_end=total_len,
        )
        x = self.vla.language_model.model.norm(self.cache_feature["_x"])
        return x[:, prefix_len:total_len]

    @torch.inference_mode()
    def predict_action_progressive(self, images_np: List[np.ndarray], proprio_raw: np.ndarray,
                                   base_images_np: Optional[List[np.ndarray]] = None,
                                   num_groups: int = 4) -> np.ndarray:
        """Interleave per-camera vision correction with causal LLM cache appends.

        Sequence order is fixed by OFT:
        ``BOS -> agentview groups 0..3 -> wrist groups 0..3 -> proprio/text``.
        A group's ViT rows are corrected and projected immediately before those
        same 64 LLM positions are appended.  Earlier causal positions are never
        revisited.  The final 56-token bidirectional action block remains stock.

        ``base_images_np=None`` is the grouped-full parity path. The progressive
        path always corrects all 256 patches as four complete groups; there is
        no secondary token pruning or partial-correction policy.
        """
        self.cache_feature = {}
        self.last_progressive_trace = []
        npatch = 256
        if len(images_np) != 2:
            raise ValueError(f"OFT LIBERO expects two images, got {len(images_np)}")
        if base_images_np is not None and len(base_images_np) != len(images_np):
            raise ValueError("base_images_np must match images_np")
        if npatch % num_groups:
            raise ValueError(f"{npatch} patches are not divisible by {num_groups}")

        # Future camera positions are allocated but remain unread (key_end
        # excludes them) until that camera's base arrives. This gives the actual
        # transmission order:
        #   agent base -> agent groups -> wrist base -> wrist groups.
        vision = torch.zeros(
            1,
            len(images_np) * npatch,
            self.bos_embed.shape[-1],
            device=self.device,
            dtype=self.bos_embed.dtype,
        )
        proprio = self._proprio_token(proprio_raw)
        prefix = torch.cat([self.bos_embed, vision, proprio, self.text_embed], dim=1)
        x0, prefix_len, total_len = self._init_chunked_prefill(prefix)
        self._prefill_block(
            x0,
            torch.arange(1, device=self.device),
            causal=True,
            key_end=1,
        )

        group_size = npatch // num_groups
        for image_idx, image in enumerate(images_np):
            base_img = (
                None
                if base_images_np is None
                else base_images_np[image_idx]
            )
            base_tokens, dino_px, siglip_px = self._vision_base(
                image,
                image_idx,
                base_img,
            )
            image_token_start = 1 + image_idx * npatch
            x0[
                :,
                image_token_start:image_token_start + npatch,
            ] = base_tokens.to(x0.dtype)
            self.last_progressive_trace.append(
                {
                    "op": "vision_base",
                    "image": image_idx,
                    "tokens": npatch,
                    "is_low_res": base_img is not None,
                }
            )
            for group_idx in range(num_groups):
                patch_start = group_idx * group_size
                group_patches = torch.arange(
                    patch_start,
                    patch_start + group_size,
                    device=self.device,
                )
                if base_images_np is not None:
                    corrected = self._vision_correct_group(
                        dino_px,
                        siglip_px,
                        image_idx,
                        group_patches,
                    )
                    absolute = image_token_start + group_patches
                    x0[:, absolute] = corrected[:, group_patches].to(x0.dtype)
                    self.last_progressive_trace.append(
                        {
                            "op": "vision_correct",
                            "image": image_idx,
                            "group": group_idx,
                            "patches": int(group_patches.numel()),
                        }
                    )

                group_token_idx = image_token_start + group_patches
                self._prefill_block(
                    x0,
                    group_token_idx,
                    causal=True,
                    key_end=int(group_token_idx[-1]) + 1,
                )
                self.last_progressive_trace.append(
                    {
                        "op": "llm_prefill",
                        "image": image_idx,
                        "group": group_idx,
                        "tokens": int(group_token_idx.numel()),
                        "key_end": int(group_token_idx[-1]) + 1,
                    }
                )

        action_hidden = self._prefill_suffix_and_actions(
            x0,
            prefix_len,
            total_len,
            1 + len(images_np) * npatch,
        )
        norm_actions = self.action_head.predict_action(action_hidden)
        return self._unnorm_actions(norm_actions[0].float().cpu().numpy())

    def _unnorm_actions(self, norm_actions: np.ndarray) -> np.ndarray:
        ast = self.stats["action"]
        q01, q99 = np.asarray(ast["q01"]), np.asarray(ast["q99"])
        mask = np.asarray(ast.get("mask", [True] * ACTION_DIM))  # gripper dim often not normalized
        return np.stack([_unnormalize_bounds_q99(a, q01, q99, mask) for a in norm_actions], axis=0)  # [8,7]
