"""CPU gates for the GLM-5.3-Flash PROMPT half (`appcorr/models/glm53/axis.py`,
`appcorr/vllm_stream/client.py::Glm53Composer`, the driver's family plumbing).

Processor and tokenizer only -- no weights, no GPU. It needs `transformers >= 5.16`
(`models/glm5_next` + the `Glm5NextProcessor` that `AutoProcessor` resolves); this box's
`appcorr` env is 5.13.0, so run it with the served env's interpreter:

    OMP_NUM_THREADS=8 PYTHONPATH=$PWD HF_HUB_OFFLINE=1 HF_HOME=/NHNHOME/huggingface \\
    /NHNHOME/share/cjpark/backup/env/appcorr-vllm-main/bin/python3.11 -m pytest \\
        tests/test_glm53_prompt.py -q -s

What is gated:

  1. the template's shape -- `[gMASK]<sop>`, the `<|begin_of_image|>` / `<|end_of_image|>`
     sentinels flanking an EXPANDED `<|image|>` run, and the generation prompt ending `<think>`;
  2. `Glm53Axis.build_inputs` appending `</think>` (and only that), with every per-token field
     extended in step, and `_image_token_run` accepting the result;
  3. `Glm53Composer.prompt_text` making the SAME edit one level up (on the text), so the vLLM
     arm and the HF arm run the same prompt;
  4. `_positions` returning `(None, 0)` and `uses_mrope` being False -- the thing that decides
     whether a streaming chunk carries a position tensor at all;
  5. `low_res_inputs` actually shrinking the grid through `max_image_tokens` (the `size` dict is
     a placeholder on this processor);
  6. `clean_text("glm53", ...)` stripping the box sentinels, and `composer_for` RAISING on an
     unregistered model id instead of returning the Qwen composer.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_ID = "zai-org/GLM-5.3-Flash"
IMAGE, START, END = 154854, 154830, 154831
THINK_OPEN, THINK_CLOSE = 154841, 154842


def _have():
    try:
        import transformers.models.glm5_next  # noqa: F401
        from appcorr.models.glm53.vision.backbone import resolve_snapshot
        return os.path.exists(os.path.join(resolve_snapshot(MODEL_ID), "chat_template.jinja"))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _have(), reason="needs transformers >= 5.16 (models/glm5_next) and the local snapshot")


@pytest.fixture(scope="module")
def proc():
    from transformers import AutoProcessor
    from appcorr.models.glm53.vision.backbone import resolve_snapshot
    return AutoProcessor.from_pretrained(resolve_snapshot(MODEL_ID))


@pytest.fixture(scope="module")
def axis(proc):
    """`Glm53Axis` with the prompt half only -- the real class and the real MRO (so `super()`
    inside `build_inputs` reaches `Glm46VAxis` -> `QwenVLStreamingAxis`), constructed without a
    45-layer model behind it."""
    from appcorr.models.glm53.axis import Glm53Axis

    class _PromptAxis(Glm53Axis):
        def __init__(self, processor):
            nn.Module.__init__(self)
            self.processor = processor
            self.cfg = SimpleNamespace(image_token_id=IMAGE, image_start_token_id=START,
                                       image_end_token_id=END,
                                       vision_config=SimpleNamespace(spatial_merge_size=2))
            self.image_token_id = IMAGE
            self.positions_mode = "fast"

    return _PromptAxis(proc)


def _image(seed=1, h=120, w=160):
    from PIL import Image
    return Image.fromarray((np.random.RandomState(seed).rand(h, w, 3) * 255).astype("uint8"))


# --- 1 / 2: the prompt ------------------------------------------------------------------------ #

def test_template_shape_and_sentinels(proc):
    txt = proc.tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}],
        tokenize=False, add_generation_prompt=True)
    print("\n[1] prompt text:", repr(txt))
    assert txt.startswith("[gMASK]<sop>")
    assert "<|begin_of_image|><|image|><|end_of_image|>" in txt
    assert txt.endswith("<|assistant|><think>")
    # the two switches GLM-4.6V had and this template does NOT
    assert "/nothink" not in txt and "enable_thinking" not in txt


def test_build_inputs_appends_think_close(axis):
    inputs = axis.build_inputs(_image(), "What is this?")
    ids = inputs["input_ids"][0]
    print(f"\n[2] seq={ids.shape[0]} grid={inputs['image_grid_thw'].tolist()} "
          f"tail={ids[-4:].tolist()}")
    assert int(ids[-1]) == THINK_CLOSE and int(ids[-2]) == THINK_OPEN
    for key in ("attention_mask", "mm_token_type_ids"):
        assert inputs[key].shape[1] == ids.shape[0], key
    assert int(inputs["mm_token_type_ids"][0, -1]) == 0

    lo, n = axis._image_token_run(inputs["input_ids"])
    print(f"    image run [{lo}, {lo + n}) of {ids.shape[0]}; "
          f"ids[lo-1]={int(ids[lo - 1])} ids[lo+n]={int(ids[lo + n])}")
    assert int(ids[lo - 1]) == START and int(ids[lo + n]) == END
    t, h, w = (int(v) for v in inputs["image_grid_thw"][0])
    assert n == (h // 2) * (w // 2) * t, (n, t, h, w)   # the processor EXPANDS the placeholder
    assert int((ids == IMAGE).sum()) == n

    think_on = axis.build_inputs(_image(), "What is this?", think=True)
    assert int(think_on["input_ids"][0, -1]) == THINK_OPEN
    assert think_on["input_ids"].shape[1] == ids.shape[0] - 1


def test_composer_prompt_text_matches_the_axis_edit(proc):
    """`Glm53Composer.prompt_text` edits the TEXT, `Glm53Axis.build_inputs` edits the IDS. They
    must produce the same token sequence, or the served arm and the HF arm run different prompts
    and every A/B between them is confounded. Asserted on the tokenizer, without an engine."""
    from appcorr.vllm_stream.client import Glm53Composer
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}]
    base = proc.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    class _PromptComposer(Glm53Composer):
        """Real class, real MRO (`prompt_text` calls `super().prompt_text`), no engine."""
        def __init__(self, hf):
            self.hf = hf
            self.image_pad_id = IMAGE

    comp_text = _PromptComposer(proc).prompt_text("What is this?")
    print(f"\n[3] composer tail: {comp_text[-24:]!r}")
    assert comp_text == base + "</think>"
    ids_text = proc.tokenizer(comp_text, add_special_tokens=False)["input_ids"]
    ids_base = proc.tokenizer(base, add_special_tokens=False)["input_ids"]
    assert ids_text == ids_base + [THINK_CLOSE], (ids_text[-3:], ids_base[-3:])


# --- 4: positions ------------------------------------------------------------------------------ #

def test_no_mrope(axis):
    from appcorr.models.qwen_vl_axis import QwenVLStreamingAxis
    assert QwenVLStreamingAxis.uses_mrope is True      # the family default is unchanged
    assert axis.uses_mrope is False
    pos, delta = axis._positions({}, image_run=(0, 1), grid_thw=(1, 2, 2))
    print(f"\n[4] _positions -> {pos!r}, delta={delta}")
    assert pos is None and delta == 0
    with pytest.raises(NotImplementedError):
        axis._positions_reference({})
    axis.positions_mode = "check"
    try:
        with pytest.raises(ValueError):
            axis._positions({})
    finally:
        axis.positions_mode = "fast"


# --- 5: the low-res knob ------------------------------------------------------------------------ #

def test_low_res_inputs_shrinks_the_grid(axis):
    full = axis.build_inputs(_image(seed=2, h=480, w=640), "What is this?")
    small = axis.low_res_inputs(_image(seed=2, h=480, w=640), "What is this?", max_image_tokens=64)
    gf = [int(v) for v in full["image_grid_thw"][0]]
    gs = [int(v) for v in small["image_grid_thw"][0]]
    nf = int((full["input_ids"][0] == IMAGE).sum())
    ns = int((small["input_ids"][0] == IMAGE).sum())
    print(f"\n[5] grid {gf} ({nf} tokens) -> max_image_tokens=64 -> {gs} ({ns} tokens)")
    assert ns < nf and ns <= 64, (nf, ns)
    assert gs[1] * gs[2] < gf[1] * gf[2]


# --- 6: the driver / gate plumbing --------------------------------------------------------------- #

def test_clean_text_strips_the_box_sentinels():
    from analysis.experiments.qwen_vllm_accuracy import FAMILIES, FAMILY_MAX_PX, clean_text
    assert "glm53" in FAMILIES
    assert FAMILY_MAX_PX["glm53"] == 8000 * (14 * 2) ** 2 == 6_272_000
    assert clean_text("glm53", "<|begin_of_box|>C<|end_of_box|>") == "C"
    assert clean_text("glm53", "plain answer") == "plain answer"


def test_composer_for_raises_on_an_unknown_model():
    from analysis.experiments.vllm_stream_gate import composer_for
    assert composer_for("zai-org/GLM-5.3-Flash").__name__ == "Glm53Composer"
    assert composer_for("zai-org/GLM-4.6V-FP8").__name__ == "Glm46VComposer"
    assert composer_for("Qwen/Qwen3.5-122B-A10B-FP8").__name__ == "Qwen25VLComposer"
    with pytest.raises(ValueError, match="no prompt composer registered"):
        composer_for("some-org/Brand-New-VL-9B")
