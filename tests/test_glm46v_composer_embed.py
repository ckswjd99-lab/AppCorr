"""CPU test for the prompt composer's two per-family parameters (`appcorr/vllm_stream/`).

No vLLM, no weights, no GPU: both things under test are lookups, and both of them are why the
Qwen composer would have failed on GLM-4.6V at the first request.

  1. The embedding table. vLLM's Qwen2-VL-family classes put `embed_input_ids` on the top-level
     `*ForConditionalGeneration`; `Glm4vForConditionalGeneration` does not (it implements
     `embed_multimodal` and leaves the table on `model.language_model`, `glm4_1v.py:2196` /
     `glm4_moe.py:582`), so the hard-coded `model.embed_input_ids` was an AttributeError waiting
     for the first GLM request. `embed_lookup.resolve_embed_fn` walks both shapes; the fake
     module trees below are those two shapes plus the failure case.
  2. The placeholder token id, `Qwen25VLComposer.IMAGE_TOKEN` (`<|image_pad|>`) vs
     `Glm46VComposer.IMAGE_TOKEN` (`<|image|>`), and the chat-template kwargs
     (`enable_thinking=False` for GLM). Those two class attributes cannot be exercised without
     importing `client.py`, which imports vLLM, so what is checked here is that the GLM values
     are the ones the real tokenizer and template want -- against the local snapshot, skipped
     when it is absent.
"""
import types

import pytest

MODEL_ID = "zai-org/GLM-4.6V-FP8"


def _fake(**tree):
    """A module tree of plain namespaces; a leaf given as a string becomes a callable that
    returns it, so the test can tell WHICH table was reached, not just that one was."""
    ns = types.SimpleNamespace()
    for k, v in tree.items():
        setattr(ns, k, (lambda tag: (lambda ids: tag))(v) if isinstance(v, str) else v)
    return ns


def test_resolve_embed_fn_qwen_shape():
    from appcorr.vllm_stream.embed_lookup import embed_path, resolve_embed_fn
    m = _fake(embed_input_ids="top",
              language_model=_fake(embed_input_ids="lm"))     # both present: top wins
    assert resolve_embed_fn(m)(None) == "top"
    assert embed_path(m) == "model.embed_input_ids"


def test_resolve_embed_fn_glm_shape():
    """`Glm4vForConditionalGeneration`: no `embed_input_ids` at the top, one on language_model."""
    from appcorr.vllm_stream.embed_lookup import embed_path, resolve_embed_fn
    m = _fake(embed_multimodal=lambda **kw: None,
              language_model=_fake(embed_input_ids="lm"))
    assert resolve_embed_fn(m)(None) == "lm"
    assert embed_path(m) == "model.language_model.embed_input_ids"


def test_resolve_embed_fn_nested_shape():
    from appcorr.vllm_stream.embed_lookup import resolve_embed_fn
    m = _fake(language_model=_fake(model=_fake(embed_input_ids="lm.model")))
    assert resolve_embed_fn(m)(None) == "lm.model"


def test_resolve_embed_fn_raises_with_the_paths_it_tried():
    from appcorr.vllm_stream.embed_lookup import resolve_embed_fn
    m = _fake(get_input_embeddings="wrong-one-on-purpose")
    with pytest.raises(AttributeError, match="model.language_model.embed_input_ids"):
        resolve_embed_fn(m)
    # and it must NOT silently accept `get_input_embeddings`, which on some classes splices
    # multimodal features in -- exactly what the embeds path exists to bypass
    with pytest.raises(AttributeError):
        resolve_embed_fn(_fake(get_input_embeddings="wrong-one-on-purpose"))


def _snapshot_available() -> bool:
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(MODEL_ID, "config.json")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _snapshot_available(), reason=f"{MODEL_ID} not in the local HF cache")
def test_glm_composer_constants_match_the_checkpoint():
    """The two values `Glm46VComposer` overrides, checked against the real tokenizer and
    template rather than against the port memo (which said `/nothink` only)."""
    from transformers import AutoConfig, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    assert type(proc).__name__ == "Glm46VProcessor"            # NOT Glm4vProcessor
    assert proc.tokenizer.convert_tokens_to_ids("<|image|>") == int(cfg.image_token_id) == 151363
    assert proc.tokenizer.convert_tokens_to_ids("<|image_pad|>") in (None,
                                                                    proc.tokenizer.unk_token_id)
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q"}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert text.startswith("[gMASK]<sop>")
    assert "<|begin_of_image|><|image|><|end_of_image|>" in text
    assert "/nothink" in text and "<think></think>" in text
