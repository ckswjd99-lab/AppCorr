"""NORA progressive vision and grouped Qwen2.5-VL causal prefill.

The action decoder is deliberately left unchanged.  This wrapper only changes
how the multimodal prompt is encoded:

* ``stock`` uses the released NORA forward/generate path.
* ``grouped_full`` uses exact vision features and appends contiguous causal
  prompt groups to a Qwen DynamicCache.
* ``pipelined`` computes a low-resolution vision base, progressively corrects
  raster-ordered merged vision cells, and appends each corrected group once.
* ``approx`` runs the released path on the low-resolution base image.

NORA's 224x224 image is 16x16 raw Qwen vision patches.  The 2x2 merger produces
an 8x8 raster of 64 LLM vision tokens.  Group boundaries are contiguous in that
raster order, which is required for append-only causal prefill.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from .vision import ApproxCorrectQwenVision


ACTION_TOKEN_MIN = 151665
ACTION_TOKEN_MAX = 153712
LIBERO_SPATIAL_LOW = np.asarray(
    [
        -0.7454732114076613,
        -0.6616071462631226,
        -0.9375,
        -0.1071428582072258,
        -0.20678570866584778,
        -0.1842857152223587,
        0.0,
    ],
    dtype=np.float32,
)
LIBERO_SPATIAL_HIGH = np.asarray(
    [
        0.9375,
        0.8758928775787354,
        0.9321428537368774,
        0.1039285734295845,
        0.17678570747375488,
        0.14571428298950195,
        1.0,
    ],
    dtype=np.float32,
)


@dataclass
class NoraPrediction:
    """One decoded NORA action chunk plus parity/debug metadata."""

    actions: np.ndarray
    normalized_actions: np.ndarray
    action_token_ids: torch.Tensor
    generated_ids: torch.Tensor
    timings: Dict[str, float]


class NoraProgressiveModel:
    """Released NORA-long checkpoint with progressive prompt-prefill modes."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        action_horizon: int = 5,
    ) -> None:
        from transformers import (
            AutoProcessor,
            GenerationConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        self.device = torch.device(device)
        self.dtype = dtype
        self.action_horizon = action_horizon
        self.processor = AutoProcessor.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        )
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast",
            trust_remote_code=True,
        )
        self.fast_tokenizer.action_dim = 7
        self.fast_tokenizer.time_horizon = action_horizon
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.generation_config = GenerationConfig.from_pretrained(
            checkpoint
        )
        self.model.generation_config.do_sample = False
        self.model.eval()
        self.vision = ApproxCorrectQwenVision(self.model.visual)

    @staticmethod
    def low_res_base(image: np.ndarray, factor: int = 4) -> np.ndarray:
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
        width, height = pil.size
        base = pil.resize(
            (max(width // factor, 1), max(height // factor, 1)),
            Image.Resampling.BILINEAR,
        ).resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(base, dtype=np.uint8)

    def prepare_inputs(
        self,
        image: np.ndarray | Image.Image,
        instruction: str,
    ) -> Dict[str, torch.Tensor]:
        from qwen_vl_utils import process_vision_info

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "resized_height": 224,
                        "resized_width": 224,
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in inputs.items()}

    def image_span(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[int, int]:
        positions = torch.where(
            input_ids[0] == self.model.config.image_token_id
        )[0]
        if positions.numel() == 0:
            raise ValueError("Prepared NORA prompt contains no image tokens")
        start = int(positions[0])
        end = int(positions[-1]) + 1
        expected = torch.arange(start, end, device=positions.device)
        if not torch.equal(positions, expected):
            raise ValueError("NORA image tokens are not one contiguous causal span")
        return start, end

    def _embed_prompt(
        self,
        inputs: Dict[str, torch.Tensor],
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        embeds = self.model.model.embed_tokens(inputs["input_ids"])
        image_mask = inputs["input_ids"] == self.model.config.image_token_id
        if int(image_mask.sum()) != image_embeds.shape[0]:
            raise ValueError(
                f"image token/features mismatch: {int(image_mask.sum())} vs "
                f"{image_embeds.shape[0]}"
            )
        return embeds.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(embeds),
            image_embeds.to(device=embeds.device, dtype=embeds.dtype),
        )

    def _position_ids(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        position_ids, rope_deltas = self.model.get_rope_index(
            inputs["input_ids"],
            inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"),
            inputs.get("second_per_grid_ts"),
            inputs.get("attention_mask"),
        )
        return position_ids, rope_deltas

    def _prefill_segments(
        self,
        inputs: Dict[str, torch.Tensor],
        prompt_embeds: torch.Tensor,
        boundaries: List[Tuple[int, int]],
    ):
        from transformers.cache_utils import DynamicCache

        position_ids, rope_deltas = self._position_ids(inputs)
        cache = DynamicCache()
        prompt_len = prompt_embeds.shape[1]
        if not boundaries or boundaries[0][0] != 0:
            raise ValueError("Grouped prefill must begin at prompt position zero")
        if boundaries[-1][1] != prompt_len - 1:
            raise ValueError("Grouped prefill must leave exactly the last token")

        for start, end in boundaries:
            self._append_segment(
                cache,
                prompt_embeds,
                position_ids,
                start,
                end,
            )
        if cache.get_seq_length() != prompt_len - 1:
            raise RuntimeError(
                f"prefill cache length {cache.get_seq_length()} != {prompt_len - 1}"
            )
        self.model.rope_deltas = rope_deltas
        return cache

    def _append_segment(
        self,
        cache,
        prompt_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
        end: int,
    ) -> None:
        if end <= start:
            return
        if cache.get_seq_length() != start:
            raise ValueError(
                f"non-contiguous Qwen cache append: cache={cache.get_seq_length()}, "
                f"segment=[{start}, {end})"
            )
        attention_mask = torch.ones(
            (1, end),
            dtype=torch.long,
            device=self.device,
        )
        self.model.model(
            inputs_embeds=prompt_embeds[:, start:end],
            position_ids=position_ids[:, :, start:end],
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(
                start,
                end,
                dtype=torch.long,
                device=self.device,
            ),
            return_dict=True,
        )

    def _generate_from_cache(
        self,
        inputs: Dict[str, torch.Tensor],
        cache,
    ) -> torch.Tensor:
        # GenerationMixin recognizes that the cache already contains prompt[:-1]
        # and slices the full input_ids to the one remaining prompt token.  Keeping
        # full input_ids also preserves repetition-penalty history exactly.
        prompt_len = inputs["input_ids"].shape[1]
        attention_mask = torch.ones(
            (1, prompt_len),
            dtype=torch.long,
            device=self.device,
        )
        return self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )

    def _decode(
        self,
        generated_ids: torch.Tensor,
        timings: Dict[str, float],
    ) -> NoraPrediction:
        action_mask = (
            (generated_ids[0] >= ACTION_TOKEN_MIN)
            & (generated_ids[0] <= ACTION_TOKEN_MAX)
        )
        action_ids = generated_ids[0, action_mask]
        if action_ids.numel() == 0:
            raise RuntimeError("NORA generated no FAST action tokens")
        normalized = np.asarray(
            self.fast_tokenizer.decode(
                [action_ids.detach().cpu() - ACTION_TOKEN_MIN]
            ),
            dtype=np.float32,
        )
        actions = (
            0.5
            * (normalized + 1.0)
            * (LIBERO_SPATIAL_HIGH - LIBERO_SPATIAL_LOW)
            + LIBERO_SPATIAL_LOW
        )
        return NoraPrediction(
            actions=np.asarray(actions[0]),
            normalized_actions=np.asarray(normalized[0]),
            action_token_ids=action_ids.detach().cpu(),
            generated_ids=generated_ids.detach().cpu(),
            timings=timings,
        )

    @torch.inference_mode()
    def predict_stock(
        self,
        image: np.ndarray | Image.Image,
        instruction: str,
    ) -> NoraPrediction:
        inputs = self.prepare_inputs(image, instruction)
        started = time.perf_counter()
        generated = self.model.generate(**inputs)
        elapsed = time.perf_counter() - started
        return self._decode(generated, {"total": elapsed})

    @torch.inference_mode()
    def predict_approx(
        self,
        image: np.ndarray | Image.Image,
        instruction: str,
        *,
        base_factor: int = 4,
    ) -> NoraPrediction:
        image_np = np.asarray(image, dtype=np.uint8)
        return self.predict_stock(
            self.low_res_base(image_np, base_factor),
            instruction,
        )

    def _group_boundaries(
        self,
        prompt_len: int,
        image_start: int,
        image_end: int,
        num_groups: int,
    ) -> List[Tuple[int, int]]:
        num_image = image_end - image_start
        if num_image % num_groups:
            raise ValueError(
                f"{num_image} image tokens are not divisible by {num_groups} groups"
            )
        group_size = num_image // num_groups
        boundaries = [(0, image_start)]
        boundaries.extend(
            [
                (
                    image_start + group_idx * group_size,
                    image_start + (group_idx + 1) * group_size,
                )
                for group_idx in range(num_groups)
            ]
        )
        boundaries.append((image_end, prompt_len - 1))
        return [(start, end) for start, end in boundaries if end > start]

    @torch.inference_mode()
    def predict_grouped_full(
        self,
        image: np.ndarray | Image.Image,
        instruction: str,
        *,
        num_groups: int = 4,
    ) -> NoraPrediction:
        inputs = self.prepare_inputs(image, instruction)
        started = time.perf_counter()
        image_embeds = self.model.visual(
            inputs["pixel_values"].to(self.model.visual.dtype),
            grid_thw=inputs["image_grid_thw"],
        )
        vision_done = time.perf_counter()
        prompt = self._embed_prompt(inputs, image_embeds)
        image_start, image_end = self.image_span(inputs["input_ids"])
        boundaries = self._group_boundaries(
            prompt.shape[1],
            image_start,
            image_end,
            num_groups,
        )
        cache = self._prefill_segments(inputs, prompt, boundaries)
        prefill_done = time.perf_counter()
        generated = self._generate_from_cache(inputs, cache)
        generated_done = time.perf_counter()
        return self._decode(
            generated,
            {
                "interleaved_vision_prefill": vision_done - started,
                "text_suffix_prefill": prefill_done - vision_done,
                "generation": generated_done - prefill_done,
                "total": generated_done - started,
            },
        )

    @torch.inference_mode()
    def predict_pipelined(
        self,
        image: np.ndarray | Image.Image,
        instruction: str,
        *,
        num_groups: int = 4,
        base_factor: int = 4,
    ) -> NoraPrediction:
        image_np = np.asarray(image, dtype=np.uint8)
        full_inputs = self.prepare_inputs(image_np, instruction)
        base_inputs = self.prepare_inputs(
            self.low_res_base(image_np, base_factor),
            instruction,
        )
        if not torch.equal(
            full_inputs["input_ids"],
            base_inputs["input_ids"],
        ):
            raise ValueError("Full/base NORA prompts have different token layouts")
        if not torch.equal(
            full_inputs["image_grid_thw"],
            base_inputs["image_grid_thw"],
        ):
            raise ValueError("Full/base NORA images have different token grids")

        started = time.perf_counter()
        base_features, vision_cache = self.vision.approx(
            base_inputs["pixel_values"].to(self.model.visual.dtype),
            base_inputs["image_grid_thw"],
        )
        prompt = self._embed_prompt(full_inputs, base_features)
        image_start, image_end = self.image_span(full_inputs["input_ids"])
        num_cells = image_end - image_start
        if num_cells % num_groups:
            raise ValueError(
                f"{num_cells} vision tokens are not divisible by {num_groups}"
            )

        from transformers.cache_utils import DynamicCache

        position_ids, rope_deltas = self._position_ids(full_inputs)
        cache = DynamicCache()
        self._append_segment(
            cache,
            prompt,
            position_ids,
            0,
            image_start,
        )

        # Interleave vision correction with causal cache appends. Earlier LLM
        # groups are never revisited.
        cell_group_size = num_cells // num_groups
        for group_idx in range(num_groups):
            cell_start = group_idx * cell_group_size
            cells = torch.arange(
                cell_start,
                cell_start + cell_group_size,
                device=self.device,
            )
            corrected, vision_cache = self.vision.correct(
                full_inputs["pixel_values"].to(self.model.visual.dtype),
                full_inputs["image_grid_thw"],
                cells,
                vision_cache,
            )
            prompt[
                :,
                image_start + cell_start : image_start + cell_start + cell_group_size,
            ] = corrected[cells].to(prompt.dtype)
            token_start = image_start + cell_start
            self._append_segment(
                cache,
                prompt,
                position_ids,
                token_start,
                token_start + cell_group_size,
            )
        vision_done = time.perf_counter()
        self._append_segment(
            cache,
            prompt,
            position_ids,
            image_end,
            prompt.shape[1] - 1,
        )
        if cache.get_seq_length() != prompt.shape[1] - 1:
            raise RuntimeError("pipelined Qwen prefill did not reach the prompt frontier")
        self.model.rope_deltas = rope_deltas
        prefill_done = time.perf_counter()
        generated = self._generate_from_cache(full_inputs, cache)
        generated_done = time.perf_counter()
        return self._decode(
            generated,
            {
                "vision": vision_done - started,
                "prefill": prefill_done - vision_done,
                "generation": generated_done - prefill_done,
                "total": generated_done - started,
            },
        )
