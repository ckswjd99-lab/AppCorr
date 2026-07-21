"""
introspect.py -- model loading + multimodal input plumbing for the stock HuggingFace
Qwen2.5-VL model (transformers 5.13.0's `Qwen2_5_VLForConditionalGeneration`).

This module is ONLY about getting the exact tensors the stock model uses -- it contains no
correction math and no benchmark logic (kept separate per the project spec). Every function here
mirrors what the stock `Qwen2_5_VLModel.forward` does internally, but exposed as discrete steps so
we can (a) extract the visual embeddings that get spliced into the LLM, and (b) drive the LLM
prefill ourselves (monolithic or chunked) with identical inputs.

Verified against the installed modeling file
(transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py):
  - `model` (Qwen2_5_VLForConditionalGeneration) .model = Qwen2_5_VLModel, .lm_head = Linear(H, vocab).
  - `model.model` .visual (vision transformer, INCLUDES the 2x2 patch merger), .language_model
    (Qwen2_5_VLTextModel, causal, M-RoPE), .get_rope_index(...), .get_image_features(...).
  - visual embeddings spliced into the LLM = `get_image_features(pixel_values, image_grid_thw)
    .pooler_output` (a tuple, one [n_merged_i, H] tensor per image), i.e. AFTER the merger, in the
    LLM's consumed order (grid raster order, already un-permuted from windowed-attention order).
  - image tokens are the `<|image_pad|>` placeholders (id = config.image_token_id); the stock model
    replaces their embeddings via masked_scatter.
  - position_ids for M-RoPE are `[3, B, T]` (temporal/height/width axes) from get_rope_index, which
    needs input_ids + mm_token_type_ids (1 at image positions) + image_grid_thw.
"""

from dataclasses import dataclass
from typing import Optional

import torch


def load_model(model_id: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load the stock Qwen2.5-VL model + processor. model_id is configurable (3B for dev, 7B/32B later)."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


@dataclass
class PreparedInputs:
    """Everything the stock model derives from (image, prompt), for a single-image single-prompt request."""
    input_ids: torch.Tensor          # [1, T]
    attention_mask: torch.Tensor     # [1, T]
    pixel_values: torch.Tensor       # [num_patches, patch_dim]
    image_grid_thw: torch.Tensor     # [num_images, 3]
    image_mask: torch.Tensor         # [1, T] bool, True at <|image_pad|> positions
    image_token_id: int
    seq_len: int
    n_visual_tokens: int             # number of merged image tokens (= image_mask.sum())


def prepare_inputs(model, processor, image, prompt: str, device: str = "cuda") -> PreparedInputs:
    """Build the exact chat-template multimodal inputs the stock model consumes (image BEFORE text,
    the standard Qwen2.5-VL ordering)."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    proc = processor(text=[text], images=[image], return_tensors="pt")
    proc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in proc.items()}

    image_token_id = model.config.image_token_id
    input_ids = proc["input_ids"]
    image_mask = input_ids == image_token_id
    return PreparedInputs(
        input_ids=input_ids,
        attention_mask=proc["attention_mask"],
        pixel_values=proc["pixel_values"],
        image_grid_thw=proc["image_grid_thw"],
        image_mask=image_mask,
        image_token_id=image_token_id,
        seq_len=int(input_ids.shape[1]),
        n_visual_tokens=int(image_mask.sum().item()),
    )


@torch.inference_mode()
def extract_visual_embeds(model, prepared: PreparedInputs) -> torch.Tensor:
    """Run the vision encoder (+ merger) and return the [n_visual_tokens, H] embeddings EXACTLY as
    the stock model splices them into the LLM sequence. This is `get_image_features(...).pooler_output`
    concatenated over images (one image here)."""
    out = model.model.get_image_features(prepared.pixel_values, prepared.image_grid_thw)
    embeds = out.pooler_output  # tuple of [n_merged_i, H], one per image
    return torch.cat(list(embeds), dim=0).to(model.dtype)  # [n_visual_tokens, H]


@torch.inference_mode()
def build_inputs_embeds(model, prepared: PreparedInputs, visual_embeds: torch.Tensor) -> torch.Tensor:
    """Embed the text tokens and scatter the visual embeddings into the <|image_pad|> positions,
    reproducing the stock model's masked_scatter. Returns [1, T, H]."""
    inputs_embeds = model.model.language_model.embed_tokens(prepared.input_ids)  # [1, T, H]
    mask = prepared.image_mask.unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(mask, visual_embeds.to(inputs_embeds.dtype))
    return inputs_embeds


@torch.inference_mode()
def compute_position_ids(model, prepared: PreparedInputs) -> torch.Tensor:
    """M-RoPE position_ids [3, 1, T] for the full sequence, computed ONCE (get_rope_index needs the
    whole sequence layout). Chunked prefill slices this per chunk -- it must NOT be recomputed per
    chunk, or the visual/text axes would be wrong."""
    mm_token_type_ids = prepared.image_mask.long()  # 1 at image positions, 0 elsewhere
    position_ids, _ = model.model.get_rope_index(
        prepared.input_ids, mm_token_type_ids,
        image_grid_thw=prepared.image_grid_thw, attention_mask=prepared.attention_mask,
    )
    return position_ids  # [3, 1, T]


@dataclass
class TokenLayout:
    """Absolute-position spans of the three contiguous regions in the Qwen prompt:
    pre-image text (chat prefix + <|vision_start|>), the image tokens, and post-image text
    (<|vision_end|> + question + assistant prefix). Used to build streaming-order chunk boundaries."""
    pre_start: int
    img_start: int
    img_end: int
    post_end: int


def token_layout(prepared: PreparedInputs) -> TokenLayout:
    idx = prepared.image_mask[0].nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        raise ValueError("no image tokens found in the prompt")
    img_start = int(idx.min().item())
    img_end = int(idx.max().item()) + 1
    # sanity: image tokens must be one contiguous run (true for single-image Qwen prompts)
    assert img_end - img_start == prepared.n_visual_tokens, (
        f"image tokens not contiguous: span {img_end-img_start} vs count {prepared.n_visual_tokens}"
    )
    return TokenLayout(pre_start=0, img_start=img_start, img_end=img_end, post_end=prepared.seq_len)


def streaming_chunk_boundaries(layout: TokenLayout, num_groups: int) -> list:
    """Chunk boundaries (list of (start, end, label)) in streaming order: the pre-image text as one
    chunk, then the image tokens split into `num_groups` contiguous visual groups, then the
    post-image text (query) as one chunk. This is both the equivalence-test split and the structure
    the oracle/correction pipeline streams in."""
    chunks = []
    if layout.img_start > layout.pre_start:
        chunks.append((layout.pre_start, layout.img_start, "pre_text"))
    n_img = layout.img_end - layout.img_start
    edges = [layout.img_start + (n_img * g) // num_groups for g in range(num_groups + 1)]
    for g in range(num_groups):
        a, b = edges[g], edges[g + 1]
        if b > a:
            chunks.append((a, b, f"visual_group_{g}"))
    if layout.post_end > layout.img_end:
        chunks.append((layout.img_end, layout.post_end, "post_text"))
    return chunks
