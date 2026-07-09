"""
qwen25vl_executor.py

ModelExecutor for Qwen2.5-VL (32B/72B), driving the forked vision tower
(appcorr/models/qwen25vl/vision/) and forked causal LLM decoder
(appcorr/models/qwen25vl/llm/decoder_layer.py) through the existing GroupTriggerPolicy scheduling
contract. `total_layers` for GroupTrigger's chunking refers to the LLM's layer count (64) -- the
vision tower is corrected to FULL depth (all 32 layers) once per group arrival (not chunked against
the LLM's schedule), matching OpenVLA's own design (DINOv2/SigLIP correction also always runs each
tower to completion per round; only the LLM's causal correction is what GroupTriggerPolicy chunks).

Each image's patches are transmitted at MERGE-GROUP granularity (`transmission_kwargs.patch_size`
should be set to `(28,28)`, matching `spatial_merge_size(2) * patch_size(14)` -- see Phase 1's
fork), so `Patch.spatial_idx` directly indexes the same `num_merge_groups = seq_len // 4` space the
vision tower fork's `correct_forward(group_idx=...)` expects, with no extra mapping needed.

Per-request question text travels via `Patch.text_payload` (set by the driver on the base-layer
patches), read once in `preprocess()` and cached in `context`.

batch_size is forced to 1 (same convention as every prior driver this session).
"""

from typing import Any, Dict

import torch

from offload.common import Task
from .base import ModelExecutor

MODEL_ID_32B = "Qwen/Qwen2.5-VL-32B-Instruct"


class Qwen25VLExecutor(ModelExecutor):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.processor = None
        self.vision_tower = None
        self.llm_layers = None
        self.num_llm_layers = 0
        self.image_token_id = None

    def load_model(self, model_name: str, config: Any):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from appcorr.models.qwen25vl.vision.backbone import ApproxCorrectQwen25VLVisionTower
        from appcorr.models.qwen25vl.llm.decoder_layer import ApproxCorrectQwen25VLDecoderLayer

        model_path = config.dataset_kwargs.get("model_path", MODEL_ID_32B)
        print(f"[Executor] Loading Model: {model_path} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.image_token_id = self.model.config.image_token_id

        self.vision_tower = ApproxCorrectQwen25VLVisionTower(self.model.model.visual).to(self.device).eval()
        text_model = self.model.model.language_model
        self.llm_layers = [
            ApproxCorrectQwen25VLDecoderLayer.from_stock(layer, text_model.rotary_emb) for layer in text_model.layers
        ]
        self.num_llm_layers = len(self.llm_layers)
        print(f"[Executor] Loaded. vision_layers={len(self.vision_tower.blocks)} llm_layers={self.num_llm_layers}")

    def preprocess(self, batch_data: Any, task: Task, context: Dict[str, Any], config: Any):
        if isinstance(batch_data, torch.Tensor):
            canvas = batch_data[0].cpu().numpy() if batch_data.ndim == 4 else batch_data.cpu().numpy()
        else:
            canvas = batch_data[0] if batch_data.ndim == 4 else batch_data
        canvas = canvas.astype("uint8")

        proc_out = self.processor.image_processor(images=[canvas], return_tensors="pt")
        context["pixel_values"] = proc_out["pixel_values"].to(device=self.device, dtype=torch.bfloat16)
        context["image_grid_thw"] = proc_out["image_grid_thw"].to(self.device)

        num_groups = context["pixel_values"].shape[0] // self.vision_tower.spatial_merge_unit
        if "group_map" not in context:
            context["group_map"] = torch.full((1, num_groups), -1, device=self.device, dtype=torch.long)
        if "question" not in context:
            for p in task.payload:
                if getattr(p, "text_payload", ""):
                    context["question"] = p.text_payload
                    break
        group_map = context["group_map"]
        for p in task.payload:
            if 0 <= p.spatial_idx < num_groups:
                group_map[0, p.spatial_idx] = p.group_id

    def _build_prompt(self, context: Dict[str, Any]):
        """Builds the text prompt + multimodal input scaffolding ONCE per request. Always exact
        (tokenization/text embedding has no notion of corrected-vs-stale), matching every prior
        fork's philosophy for cheap non-block prep steps -- but unlike DINOv3/CLIP, this also needs
        the image_grid_thw (to know how many image-token placeholders to insert), which is why it
        runs inside prepare_tokens (after preprocess has set pixel_values/image_grid_thw) rather
        than being independent of the image."""
        question = context["question"]
        grid_thw = context["image_grid_thw"]
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        merge_unit = self.vision_tower.spatial_merge_unit
        num_image_tokens = int((grid_thw.prod(dim=-1) // merge_unit).sum().item())
        image_pad = "<|image_pad|>" * num_image_tokens
        text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", f"<|vision_start|>{image_pad}<|vision_end|>")

        tok_out = self.processor.tokenizer(text, return_tensors="pt")
        input_ids = tok_out["input_ids"].to(self.device)
        attention_mask = tok_out["attention_mask"].to(self.device)
        image_mask_1d = (input_ids[0] == self.image_token_id)
        mm_token_type_ids = image_mask_1d.long().unsqueeze(0)

        position_ids, _ = self.model.model.get_rope_index(
            input_ids, mm_token_type_ids, image_grid_thw=grid_thw, attention_mask=attention_mask
        )
        context["input_ids"] = input_ids
        context["attention_mask"] = attention_mask
        context["image_mask_1d"] = image_mask_1d
        context["position_ids"] = position_ids
        context["permanent_group_idx"] = (~image_mask_1d).nonzero(as_tuple=True)[0]
        context["image_token_positions"] = image_mask_1d.nonzero(as_tuple=True)[0]  # merge-group g -> this[g]

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        if "pixel_values" not in context:
            return
        vision_ctx = self.vision_tower.prepare_full_tokens(context["pixel_values"], context["image_grid_thw"])
        context["vision_ctx"] = vision_ctx
        context["vision_current_feature"] = vision_ctx["hidden_states"]

        if "input_ids" not in context:
            self._build_prompt(context)
            input_embeds = self.model.model.language_model.embed_tokens(context["input_ids"])
            context["llm_input_embeds_template"] = input_embeds  # text rows exact; image rows placeholder

    def _splice_image_embeds(self, context: Dict[str, Any], image_embeds_merged: torch.Tensor) -> torch.Tensor:
        """Scatters the current (possibly partially-corrected) merged image embeddings into the
        cached text-embedding template at the real image-token positions, matching stock's own
        `masked_scatter` -- always exact given whatever `image_embeds_merged` currently is."""
        template = context["llm_input_embeds_template"].clone()
        image_mask = context["image_mask_1d"].unsqueeze(0).unsqueeze(-1).expand_as(template)
        return template.masked_scatter(image_mask, image_embeds_merged.to(template.dtype))

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        start_l, end_l = params.get("layers", (0, self.num_llm_layers))

        if start_l == 0:
            # First call of the request: approx the vision tower to FULL depth once (matching
            # OpenVLA's vision towers, always run to completion per round, not chunked), then
            # build the initial multimodal embedding sequence from the (approx-only) image.
            vctx = context["vision_ctx"]
            x_v, cache_v = self.vision_tower.approx_forward(
                context["vision_current_feature"], 0, len(self.vision_tower.blocks), vctx, {}, tag_prefix="vision"
            )
            context["vision_current_feature"] = x_v
            context["vision_cache"] = cache_v
            merged = self.vision_tower.get_merged_output(x_v, vctx)
            context["llm_input_embeds"] = self._splice_image_embeds(context, merged)
            x_feature = context["llm_input_embeds"]
        else:
            x_feature = context["llm_current_feature"]

        cache = context.get("llm_cache", {})
        for i in range(start_l, end_l):
            layer = self.llm_layers[i]
            x_feature, cache = layer.approx(x_feature, context["position_ids"], cache, tag=f"llm_layer{i}")
        context["llm_current_feature"] = x_feature
        context["llm_cache"] = cache

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        start_l, end_l = params.get("layers", (0, self.num_llm_layers))
        group_id = params.get("group_id", 1)
        group_map = context["group_map"]
        group_idx = torch.where(group_map[0] == group_id)[0]
        if group_idx.numel() == 0:
            return

        # Vision tower: correct the newly-arrived merge-groups to FULL depth (all 32 layers), from
        # the CURRENT canvas (already reconstructed by the transmission layer's decode, matching
        # every prior fork's "restart x from prepare_full_tokens's output each round" invariant).
        vctx = self.vision_tower.prepare_full_tokens(context["pixel_values"], context["image_grid_thw"])
        x_v, vcache = self.vision_tower.correct_forward(
            vctx["hidden_states"], group_idx, 0, len(self.vision_tower.blocks), vctx, context["vision_cache"], tag_prefix="vision"
        )
        context["vision_cache"] = vcache
        merged = self.vision_tower.get_merged_output(x_v, vctx)
        context["llm_input_embeds"] = self._splice_image_embeds(context, merged)

        # LLM: correct the permanent group (all non-image positions) UNION the image-token
        # positions corresponding to the merge-groups that just got corrected, for this round's
        # layer chunk.
        image_token_positions = context["image_token_positions"][group_idx.to(context["image_token_positions"].device)]
        token_idx = torch.cat([context["permanent_group_idx"], image_token_positions]).unique()

        x_feature = context["llm_input_embeds"]
        cache = context.get("llm_cache", {})
        for i in range(start_l, end_l):
            layer = self.llm_layers[i]
            x_feature, cache = layer.correct(
                x_feature, token_idx, cache, tag=f"llm_layer{i}", position_ids=context["position_ids"]
            )
        context["llm_current_feature"] = x_feature
        context["llm_cache"] = cache

    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        """The FIRST generated token is decoded from the actual corrected/approx prefill state
        (`context["llm_current_feature"]`'s final position) -- this is the token whose correctness
        this whole mechanism is meant to validate (RealWorldQA answers are frequently a single
        letter/word, so this token alone often IS the whole answer). If more tokens are needed,
        remaining generation falls back to a stock `model.generate()` call on
        `[input_ids, first_token]` -- this recomputes the full sequence via stock forward for the
        (typically short) continuation, a deliberate, documented simplification: building a real
        incremental KV-cache-append decode loop on top of the hand-rolled correction cache was out
        of scope for validating the core approx/correct mechanism, and RealWorldQA answers are
        short enough (1 letter to a few words) that this fallback's cost is bounded and does not
        change what the FIRST token (already fixed before the fallback runs) was."""
        x_full = context.get("llm_current_feature")
        hidden = self.model.model.language_model.norm(x_full)
        logits_last = self.model.lm_head(hidden[:, -1, :].to(self.model.lm_head.weight.dtype))
        first_token = logits_last.argmax(dim=-1)

        if first_token.item() == self.processor.tokenizer.eos_token_id:
            answer_text = ""
        else:
            with torch.no_grad():
                extended_ids = torch.cat([context["input_ids"], first_token.unsqueeze(0)], dim=1)
                extended_mask = torch.cat(
                    [context["attention_mask"], torch.ones_like(first_token.unsqueeze(0))], dim=1
                )
                gen_ids = self.model.generate(
                    input_ids=extended_ids,
                    attention_mask=extended_mask,
                    pixel_values=context["pixel_values"],
                    image_grid_thw=context["image_grid_thw"],
                    max_new_tokens=63,
                    do_sample=False,
                )
            gen_trimmed = gen_ids[:, context["input_ids"].shape[1]:]
            answer_text = self.processor.tokenizer.decode(gen_trimmed[0], skip_special_tokens=True)

        context["output"] = answer_text
        return {"answer_text": answer_text}

    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        if "pixel_values" not in context:
            return
        if "input_ids" not in context:
            self._build_prompt(context)
        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids=context["input_ids"],
                attention_mask=context["attention_mask"],
                pixel_values=context["pixel_values"],
                image_grid_thw=context["image_grid_thw"],
                max_new_tokens=64,
                do_sample=False,
            )
        gen_trimmed = gen_ids[:, context["input_ids"].shape[1]:]
        answer_text = self.processor.tokenizer.decode(gen_trimmed[0], skip_special_tokens=True)
        context["output"] = answer_text

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        if "output" not in context:
            return {}
        return {0: context["output"]}

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return {}
