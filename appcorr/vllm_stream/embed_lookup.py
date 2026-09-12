"""Where a vLLM model class keeps its input-embedding table.

One function, its own module, and no vLLM import: `client.py` cannot be imported without vLLM
(it pulls in `request.py` -> `vllm.v1.request`), and this rule is worth testing on a CPU with a
fake module tree rather than only on a live 106B engine.

The rule itself. vLLM's Qwen2-VL-family classes expose `embed_input_ids(ids)` on the TOP-LEVEL
`*ForConditionalGeneration` object, so `Qwen25VLComposer.embed` called `model.embed_input_ids`
directly. GLM-4.6V does not: `Glm4vForConditionalGeneration` (which
`Glm4vMoeForConditionalGeneration` subclasses, `vllm/model_executor/models/glm4_1v.py:1735,
2356`) implements `embed_multimodal(**kwargs)` and leaves the token embedding on the language
model (`glm4_moe.py:582`). Calling `model.embed_input_ids` on it is an `AttributeError` at the
first request, so the lookup walks:

    model.embed_input_ids  ->  model.language_model.embed_input_ids
                           ->  model.model.embed_input_ids
                           ->  model.language_model.model.embed_input_ids

and raises with the tried paths if none exists -- never falling back to something that would
silently return a different tensor (e.g. `get_input_embeddings`, which on some classes also
splices multimodal features in and would defeat the whole point of pushing our own image rows).
"""
from __future__ import annotations

from typing import Any, Callable, List, Tuple

# Attribute chains tried, in order. First hit wins.
EMBED_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("embed_input_ids",),                   # Qwen2.5-VL / Qwen3.5
    ("language_model", "embed_input_ids"),  # GLM-4.6V / GLM-4.5V (Glm4vForConditionalGeneration)
    ("model", "embed_input_ids"),
    ("language_model", "model", "embed_input_ids"),
)


def _walk(obj: Any, path: Tuple[str, ...]):
    for name in path:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj if callable(obj) else None


def resolve_embed_fn(model: Any) -> Callable:
    """The model's token-embedding callable: `fn(input_ids) -> [T, D]`.

    Raises `AttributeError` naming every path tried, rather than guessing.
    """
    for path in EMBED_PATHS:
        fn = _walk(model, path)
        if fn is not None:
            return fn
    tried: List[str] = [".".join(("model",) + p) for p in EMBED_PATHS]
    raise AttributeError(
        f"{type(model).__name__} exposes no input-embedding callable; tried {tried}. "
        "Add its path to appcorr/vllm_stream/embed_lookup.EMBED_PATHS -- do NOT substitute "
        "`get_input_embeddings`, which on some classes splices multimodal features in.")


def embed_path(model: Any) -> str:
    """The path `resolve_embed_fn` would take, for logging / gate output."""
    for path in EMBED_PATHS:
        if _walk(model, path) is not None:
            return ".".join(("model",) + path)
    return "<none>"
