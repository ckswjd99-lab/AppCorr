"""
openclip_executor.py

ModelExecutor for CLIP-ViT-bigG/14 (transformers.CLIPModel), driving the forked vision tower
(appcorr/models/openclip/vision/) through the existing GroupTriggerPolicy scheduling contract
(same layers=(start_l,end_l)/group_id params used by DINOv3ClassifierExecutor -- see
offload/policies/scheduling/group_trigger.py). Text tower always runs a plain one-shot full
forward (never progressively streamed) since only images are transmitted patch-by-patch.

Two task modes, selected by `config.dataset_kwargs.get('clip_task', 'zeroshot')`:
    - "zeroshot": ImageNet-1k zero-shot classification. `load_model` precomputes the 1000-class
      text embedding matrix once (80-template ensemble, via open_clip's IMAGENET_CLASSNAMES/
      OPENAI_IMAGENET_TEMPLATES). `head_inference` computes cosine-sim logits (scaled by
      logit_scale) against all 1000 classes, returns top5.
    - "retrieval": MS-COCO image-text retrieval. `head_inference` just returns the normalized
      image embedding itself (no classification head) -- the eval driver accumulates these and
      computes recall@k against separately-precomputed caption embeddings.

batch_size is forced to 1 by the eval drivers (same convention as
`analysis/experiments/dinov3_classifier_offload_eval.py`), so there is no per-batch-item variable
masking to handle -- `group_map`/`patch_idx` are single-image tensors.
"""

from typing import Any, Dict

import numpy as np
import torch

from offload.common import Task
from offload.common.protocol import normalize_appcorr_kwargs
from .base import ModelExecutor

MODEL_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

# Standard OpenAI CLIP normalization (confirmed via CLIPProcessor.image_processor for this checkpoint).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class OpenCLIPExecutor(ModelExecutor):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.tower = None
        self.clip_model = None
        self.processor = None
        self.zeroshot_weights = None  # [1000, proj_dim], only built for clip_task == "zeroshot"
        self.logit_scale = 100.0
        self.clip_task = "zeroshot"
        self.norm_mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1).to(self.device).float()
        self.norm_std = torch.tensor(CLIP_STD).view(1, 3, 1, 1).to(self.device).float()

    def backbone_modules(self):
        """Vision tower plus the projection into the shared embedding space.

        Zero-shot classification's "head" is a precomputed text-prototype matrix, and retrieval has
        no head at all, so the image embedding IS this VFM's feature and the projection belongs
        inside the backbone rather than after it.
        """
        return [getattr(self.clip_model, "vision_model", None),
                getattr(self.clip_model, "visual_projection", None)]

    def load_model(self, model_name: str, config: Any):
        from transformers import CLIPModel, CLIPProcessor
        from appcorr.models.openclip.vision.backbone import ApproxCorrectCLIPVisionTower

        self.clip_task = config.dataset_kwargs.get("clip_task", "zeroshot")
        print(f"[Executor] Loading Model: {MODEL_ID} (task={self.clip_task})...")
        self.clip_model = CLIPModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        self.tower = ApproxCorrectCLIPVisionTower(
            self.clip_model.vision_model, self.clip_model.visual_projection
        ).to(self.device).eval()
        self.logit_scale = self.clip_model.logit_scale.exp().item()

        if self.clip_task == "zeroshot":
            self.zeroshot_weights = self._build_zeroshot_classifier()
        self.model = self.clip_model  # satisfies ModelExecutor's self.model bookkeeping convention

    def _build_zeroshot_classifier(self) -> torch.Tensor:
        from open_clip import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

        print("[Executor] Building 1000-class zero-shot classifier (80-template ensemble)...")
        all_embeds = []
        with torch.no_grad():
            for classname in IMAGENET_CLASSNAMES:
                texts = [tmpl(classname) for tmpl in OPENAI_IMAGENET_TEMPLATES]
                inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
                text_feats = self.clip_model.get_text_features(**inputs).pooler_output
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                class_embed = text_feats.mean(dim=0)
                class_embed = class_embed / class_embed.norm()
                all_embeds.append(class_embed)
        return torch.stack(all_embeds, dim=0).float()  # [1000, proj_dim]

    def _num_patches(self, config: Any) -> int:
        H, W = config.image_shape[:2]
        ph, pw = (config.patch_size, config.patch_size) if isinstance(config.patch_size, int) else config.patch_size
        return (H // ph) * (W // pw)

    def preprocess(self, batch_data: Any, task: Task, context: Dict[str, Any], config: Any):
        if isinstance(batch_data, torch.Tensor):
            tensor = batch_data.to(device=self.device, non_blocking=True)
            if tensor.shape[1] != 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(0, 3, 1, 2)
            tensor = tensor.float()
            if batch_data.dtype == torch.uint8:
                tensor = tensor / 255.0
        else:
            tensor = torch.from_numpy(batch_data).to(device=self.device, non_blocking=True)
            tensor = tensor.permute(0, 3, 1, 2).float() / 255.0

        tensor = (tensor - self.norm_mean) / self.norm_std
        context["input_tensor"] = tensor.to(dtype=torch.bfloat16)

        num_patches = self._num_patches(config)
        if "group_map" not in context:
            context["group_map"] = torch.full((1, num_patches), -1, device=self.device, dtype=torch.long)
        if "mobile_pscore_hint_map" not in context:
            context["mobile_pscore_hint_map"] = torch.zeros((1, num_patches), device=self.device, dtype=torch.float32)
        group_map = context["group_map"]
        hint_map = context["mobile_pscore_hint_map"]
        for p in task.payload:
            if 0 <= p.spatial_idx < num_patches:
                group_map[0, p.spatial_idx] = p.group_id
                hint_map[0, p.spatial_idx] = float(getattr(p, "pscore_hint", 0.0))

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        if "input_tensor" not in context:
            return
        input_tokens = self.tower.prepare_full_tokens(context["input_tensor"])
        context["input_tokens"] = input_tokens
        if "current_feature" not in context:
            context["current_feature"] = input_tokens

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        start_l, end_l = params.get("layers", (0, len(self.tower.blocks)))
        x_feature = context["input_tokens"] if start_l == 0 else context.get("current_feature", context["input_tokens"])
        cache = context.get("cache_feature", {})
        want_mean = self._pscore_kind(config) == "patch_attn_prob_layermean"
        x_feature, cache = self.tower.approx_forward(x_feature, start_l, end_l, cache,
                                                     tag_prefix="vision",
                                                     collect_cls_attn=not want_mean,
                                                     collect_attn_mean=want_mean)
        # Refresh the importance signal after every approx chunk (not just the final one) -- a
        # partial-depth average is a usable, if less refined, proxy, and later groups' pruning
        # decisions benefit from using whatever depth has been seen so far rather than waiting.
        cache = (self.tower.finalize_attn_layermean(cache, tag_prefix="vision")
                 if self._pscore_kind(config) == "patch_attn_prob_layermean"
                 else self.tower.finalize_cls_attn_layermean(cache, tag_prefix="vision"))
        context["current_feature"] = x_feature
        context["cache_feature"] = cache

    @staticmethod
    def _pscore_kind(config: Any) -> str:
        """Which attention signal feeds the server side of the importance score.

        `cls_attn_prob_layermean` (default, and what every OpenCLIP result before 2026-08-26 used):
            the CLS token's attention row. Defensible for CLIP specifically -- the image embedding
            IS the CLS output -- but that argument holds at the final layer and weakens once the
            rows are averaged across layers, and it discards all patch-to-patch interaction.
        `patch_attn_prob_layermean`: the column mean, i.e. attention RECEIVED per token. The signal
            DINOv3 and Gemma 3 use, and the one that needs no CLS token.

        Read from the RAW config, not the normalised one: `normalize_appcorr_kwargs` supplies its own
        default for `server_pscore`, so reading the normalised value would silently switch existing
        configs onto whatever that default happens to be.
        """
        raw = getattr(config, "appcorr_kwargs", None) or {}
        kind = str(raw.get("server_pscore", "cls_attn_prob_layermean"))
        if kind not in ("cls_attn_prob_layermean", "patch_attn_prob_layermean"):
            raise ValueError(
                f"openclip: unknown server_pscore {kind!r}; expected 'cls_attn_prob_layermean' or "
                "'patch_attn_prob_layermean'. Research code -- an unrecognised selection signal is a "
                "fault, not something to fall back from."
            )
        return kind

    def _prune_patch_idx(self, patch_idx: torch.Tensor, context: Dict[str, Any], config: Any) -> torch.Tensor:
        """Applies the validated `residual_energy x avg_cls_attn` thresholded importance score
        (see analysis/experiments/ENERGY_GROUPING_LOG.md's classifier finding, ported here for
        CLIP) to sub-select which of a group's arrived patches actually get corrected this round.
        Selection is top-k when the config sets `token_keep_ratio` -- an exact keep rate, the same
        knob the DINOv3 family exposes -- and thresholded on the same score otherwise. No-op (keeps
        all of patch_idx) when neither is configured, or when the signals are not ready yet (e.g.
        mobile_pscore_hint_map carries no real residual-energy hints for this group, or no approx
        chunk has run to seed cls_attn_layermean)."""
        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        token_keep_thres = appcorr_options.get("token_keep_thres")
        # Top-k selection, matching the DINOv3 family's `token_keep_ratio`, so a keep RATE can be
        # asked for directly instead of being reverse-engineered from a threshold sweep.
        #
        # Only honoured when the config states it. `normalize_appcorr_kwargs` defaults the ratio to
        # 0.2, so reading the normalized value would silently turn every existing OpenCLIP run from
        # "keep everything" into "keep 20%" -- a behaviour change disguised as a new feature.
        raw_appcorr = getattr(config, "appcorr_kwargs", None) or {}
        token_keep_ratio = (float(raw_appcorr["token_keep_ratio"])
                            if "token_keep_ratio" in raw_appcorr else None)
        use_topk = token_keep_ratio is not None and token_keep_ratio < 1.0
        if token_keep_thres is None and not use_topk:
            return patch_idx

        cache = context.get("cache_feature", {})
        key = ("vision_cls_attn_layermean" if self._pscore_kind(config) == "cls_attn_prob_layermean"
               else "vision_attn_layermean")
        layermean = cache.get(key)
        hint_map = context.get("mobile_pscore_hint_map")
        if layermean is None or hint_map is None:
            return patch_idx

        server_score = layermean[0, patch_idx]  # [Q]
        mobile_score = hint_map[0, patch_idx]  # [Q], raw residual energy
        if bool((mobile_score == 0).all()):
            # No real residual-energy hint for this group yet (e.g. approx-only/no mobile hint
            # configured) -- pruning would be meaningless (everything scores 0), so skip it.
            return patch_idx

        combined = server_score.float() * mobile_score.float()

        import os
        if os.environ.get("CALIBRATE_PSCORE"):
            qs = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=combined.device)
            pct = torch.quantile(combined, qs).tolist()
            print(
                f"[calibrate][openclip] combined_patch_scores percentiles "
                f"[0,10,25,50,75,90,100]% = {[f'{v:.6g}' for v in pct]} mean={combined.mean().item():.6g} "
                f"n={combined.numel()}",
                flush=True,
            )

        if use_topk:
            n = int(combined.numel())
            k = max(1, min(int(round(n * token_keep_ratio)), n))
            keep_mask = torch.zeros(n, dtype=torch.bool, device=combined.device)
            keep_mask.scatter_(0, combined.topk(k).indices, True)
        else:
            keep_mask = combined >= token_keep_thres
        full_count = float(patch_idx.numel())
        kept_count = float(int(keep_mask.sum().item())) if bool(keep_mask.any()) else full_count
        cache["_token_prune_kept_patch_total"] = cache.get("_token_prune_kept_patch_total", 0.0) + kept_count
        cache["_token_prune_full_patch_total"] = cache.get("_token_prune_full_patch_total", 0.0) + full_count
        context["cache_feature"] = cache
        if not bool(keep_mask.any()):
            return patch_idx  # never prune a group down to nothing
        return patch_idx[keep_mask]

    def _bucket_pad_patch_idx(self, patch_idx: torch.Tensor, config: Any) -> torch.Tensor:
        """Pads the query set to a fixed bucket-size multiple by duplicating existing indices, so
        variable-length pruned query sets don't each trigger a novel cuBLAS/SDPA kernel-shape
        dispatch (the same mitigation validated for the DINOv3 classifier). Safe: a duplicated
        index reads/writes the exact same underlying value every time (x_active for that row is
        identical since it's read from the same source position), so padding never corrupts
        anything, just spends compute on redundant rows in exchange for a stable kernel shape."""
        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        bucket = appcorr_options.get("sdpa_query_bucket_size", 0) or 0
        if bucket <= 0 or patch_idx.numel() == 0:
            return patch_idx
        total_len = patch_idx.numel() + self.tower.num_prefix_tokens
        target = ((total_len + bucket - 1) // bucket) * bucket
        pad_n = target - total_len
        if pad_n <= 0:
            return patch_idx
        pad = patch_idx[-1:].expand(pad_n)
        return torch.cat([patch_idx, pad])

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        start_l, end_l = params.get("layers", (0, len(self.tower.blocks)))
        group_id = params.get("group_id", 1)
        group_map = context["group_map"]
        patch_idx = torch.where(group_map[0] == group_id)[0]
        if patch_idx.numel() == 0:
            return
        patch_idx = self._prune_patch_idx(patch_idx, context, config)
        patch_idx = self._bucket_pad_patch_idx(patch_idx, config)

        x_feature = context["input_tokens"]
        cache = context.get("cache_feature", {})
        x_feature, cache = self.tower.correct_forward(x_feature, patch_idx, start_l, end_l, cache, tag_prefix="vision")
        cache = (self.tower.finalize_attn_layermean(cache, tag_prefix="vision")
                 if self._pscore_kind(config) == "patch_attn_prob_layermean"
                 else self.tower.finalize_cls_attn_layermean(cache, tag_prefix="vision"))
        context["current_feature"] = x_feature
        context["cache_feature"] = cache

    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        x_full = context.get("current_feature")
        image_embeds = self.tower.get_image_embeds(x_full).float()

        if self.clip_task == "zeroshot":
            logits = self.logit_scale * image_embeds @ self.zeroshot_weights.T
            top5_probs, top5_indices = torch.topk(torch.softmax(logits, dim=-1), k=5, dim=1)
            context["output"] = top5_indices
            return {
                "top5_probs": top5_probs.cpu().numpy().tolist(),
                "top5_indices": top5_indices.cpu().numpy().tolist(),
            }
        else:  # retrieval
            context["output"] = image_embeds
            return {"image_embeds": image_embeds.cpu().numpy().tolist()}

    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        inp = context.get("input_tensor")
        if inp is None:
            return
        with torch.no_grad():
            image_embeds = self.clip_model.get_image_features(pixel_values=inp).pooler_output.float()
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        if self.clip_task == "zeroshot":
            logits = self.logit_scale * image_embeds @ self.zeroshot_weights.T
            _, top5_indices = torch.topk(torch.softmax(logits, dim=-1), k=5, dim=1)
            context["output"] = top5_indices
        else:
            context["output"] = image_embeds

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        """One entry per sample in the batch, keyed by its index within the batch.

        This used to return `{0: output[0]}` -- the first sample only -- while the configs run at
        `batch_size: 32`. `worker.py` fills the rest with `final_map.get(i, [])`, and the ImageNet
        evaluator skips an empty prediction for the correct-count but still counts it in the
        denominator, so the reported accuracy was capped at exactly 1/32 = 3.125%. Measured: top5
        3.125% (one per batch, 20 batches), top1 2.97% (19 of those 20 correct) -- i.e. the model was
        fine at ~95% on what it actually scored, and the loss was entirely in this collection step.
        It hit ceiling, floor and every corrected arm equally, so nothing about the accuracy ordering
        between them was visible either.
        """
        if "output" not in context:
            return {}
        output = context["output"]
        rows = output.cpu().numpy().tolist()
        return {i: row for i, row in enumerate(rows)}

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return {}
