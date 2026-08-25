"""Model-agnostic capture of backbone FLOPs: module hooks for GEMMs, a wrapper for attention.

Deriving each model's FLOPs from its config was the obvious approach and it is the wrong one. It
means re-reading seven forwards, and every one of them has something that would be got wrong from
the outside: SAM 3 runs global attention on 4 of 32 layers and windowed on the rest, the DINOv3
detector issues ten separate backbone calls per image, Qwen2.5-VL is windowed with full attention at
[7,15,23,31], OV2's token count is a function of the image. A formula written against a config
drifts from the code silently.

Hooks read the real shapes instead, so the count follows whatever the model actually did:

  * `nn.Linear` and `nn.Conv*` forward hooks -- every projection, MLP and patch embedding, at the
    shapes that were really used. This is also why the fork modules in this repo need no special
    handling: they reuse the stock model's `Linear`s rather than constructing their own, so a hook
    installed on the stock module catches the fork's calls too.
  * `torch.nn.functional.scaled_dot_product_attention` -- wrapped for the duration, because the
    QK^T and AV products are not `Linear`s and are the quadratic term that dominates at the token
    counts here. Every model in this repo reaches attention through this function, including the
    stock HF paths, whose `sdpa_attention_forward` calls it.

**A missed path is the failure mode that matters**, because it produces a confidently low number
rather than an error. `attention_capture_ratio` exists for that: it compares captured attention
FLOPs against an independent estimate and lets a gate assert the two agree. Any model that reaches
attention through flash-attn varlen or a fused kernel will show up there as a shortfall, and can
then declare its attention explicitly with `record_attention`.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterable, List, Optional

import torch
from torch import nn

from .counter import FlopCounter

_ATTN_ORIG = None
_PATCH_DEPTH = 0


def _linear_flops(module: nn.Linear, out: torch.Tensor) -> int:
    # 2 * M * N * K, with M*N read off the real output. Bias adds are not counted, per convention.
    return 2 * out.numel() * module.in_features


def _conv_flops(module: nn.modules.conv._ConvNd, out: torch.Tensor) -> int:
    k = 1
    for d in module.kernel_size:
        k *= d
    return 2 * out.numel() * (module.in_channels // module.groups) * k


def install(counter: FlopCounter, roots: Iterable[nn.Module]) -> List[torch.utils.hooks.RemovableHandle]:
    """Hook every Linear/Conv under `roots`.

    `roots` is the backbone, and passing it is how the head gets excluded -- by not being in the
    subtree, rather than by being named somewhere. For a VLM pass the vision tower and the language
    model; for a VFM pass the feature trunk and stop before the task head.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []
    seen = set()
    roots = list(roots)
    if not roots or any(r is None for r in roots):
        # A None root used to be skipped silently. That turns "backbone_modules() could not find the
        # trunk" into "count nothing from it" -- and because `patch_attention` is global, attention
        # keeps being counted, so the arm reports a smaller but entirely plausible number instead of
        # zero. `dinov3_detector` returned [None] for months and COCO measured 1874 GF/image against
        # a closed-form ~54,900. Research code: crash instead (CLAUDE.md).
        raise ValueError(
            f"flops.install: backbone root is None (roots={roots!r}). The executor's "
            "backbone_modules() did not resolve the trunk; fix the accessor rather than letting "
            "the count silently omit it."
        )
    for root in roots:
        for mod in root.modules():
            if id(mod) in seen:
                continue
            seen.add(id(mod))
            if isinstance(mod, nn.Linear):
                handles.append(mod.register_forward_hook(
                    lambda m, i, o, c=counter: c.record(linear=_linear_flops(m, o))))
            elif isinstance(mod, nn.modules.conv._ConvNd):
                handles.append(mod.register_forward_hook(
                    lambda m, i, o, c=counter: c.record(conv=_conv_flops(m, o))))
    return handles


def remove(handles: Iterable[torch.utils.hooks.RemovableHandle]) -> None:
    for h in handles:
        h.remove()


def record_attention(counter: FlopCounter, q: torch.Tensor, k: torch.Tensor) -> None:
    """QK^T plus AV, from the query and key shapes. [..., H, S, D].

    Head count comes from the QUERY. Under GQA the keys carry fewer heads and are expanded before
    the product, so counting the key's heads would under-report by the group factor -- 4x on
    Qwen3's 32/8, 5x on Qwen2.5-VL-32B's 40/8.
    """
    if q.dim() < 3 or k.dim() < 3:
        return
    d = q.shape[-1]
    sq, sk = q.shape[-2], k.shape[-2]
    lead = 1
    for s in q.shape[:-2]:
        lead *= s
    counter.record(attention=2 * 2 * lead * sq * sk * d)


@contextmanager
def patch_attention(counter: FlopCounter):
    """Wrap `scaled_dot_product_attention` for the duration.

    Re-entrant: nested scopes share one patch and only the outermost restores, so an executor that
    opens a scope inside a driver that already opened one does not double-count or leave the
    function permanently wrapped.
    """
    global _ATTN_ORIG, _PATCH_DEPTH
    if _PATCH_DEPTH == 0:
        _ATTN_ORIG = torch.nn.functional.scaled_dot_product_attention

        def wrapped(query, key, value, *args, **kwargs):
            record_attention(counter, query, key)
            return _ATTN_ORIG(query, key, value, *args, **kwargs)

        torch.nn.functional.scaled_dot_product_attention = wrapped
    _PATCH_DEPTH += 1
    try:
        yield
    finally:
        _PATCH_DEPTH -= 1
        if _PATCH_DEPTH == 0:
            torch.nn.functional.scaled_dot_product_attention = _ATTN_ORIG
            _ATTN_ORIG = None


def attention_capture_ratio(counter: FlopCounter, *, layers: int, heads: int, head_dim: int,
                            seq_len: int, windows: Optional[Iterable[int]] = None) -> float:
    """Captured attention FLOPs over an independent estimate of what they should be.

    The point is to catch a SILENT miss. A fused or varlen attention kernel that never reaches
    `scaled_dot_product_attention` costs the counter the entire quadratic term while every other
    number still looks sane, and the result is a critical-FLOPs figure that is confidently too low.

    `windows` gives the per-layer key length when attention is not global on every layer (SAM 3:
    576 on 28 layers and 5184 on 4; Qwen2.5-VL: a 112-pixel window except at [7,15,23,31]).
    Omitting it assumes `seq_len` everywhere.
    """
    got = sum(b.attention for r in counter.requests for b in r.buckets.values())
    if windows is None:
        expect = layers * (2 * 2 * heads * seq_len * seq_len * head_dim)
    else:
        expect = sum(2 * 2 * heads * seq_len * int(w) * head_dim for w in windows)
    return (got / expect) if expect else math.inf
