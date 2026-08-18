"""Quantize the correction *delta* instead of the corrected activation, at the Linear layers only.

The claim being tested: AppCorr's approximate pass does the work on the large values, so correction
is left holding a smaller operand, and a smaller operand carries less absolute quantization error.
Measured per-layer on COCO 1024px, quantizing `d` instead of `a+d` injects 2.41x less error into the
output on average, and 5-7x less in the deepest blocks. This module exists to carry that from a
per-layer L2 number through to real mIoU, which is the only number worth quoting.

**Why this is not the exact (a, d) decomposition.** A full decomposition rewrites every non-linearity
as `g(a+d) - g(a)` and has to contend with RoPE mutating the KV cache in place and with
`blocks_out_sum` fusing the attention and MLP increments irreversibly. None of that is needed here.
The proposition is only about what gets fed to a Linear, so only the Linear is split:

    today   Linear(quant(a + d))
    delta   Linear(a) + Linear(quant(d))          with d = (a+d) - a

Everything else -- attention core, LayerNorms, SwiGLU, the residual bookkeeping -- keeps running on
full values exactly as it does now. Since `W a + W d = W (a+d)` in exact arithmetic, turning the
quantization off makes the delta path an *identity* on the current one, which is the validation gate:
`mode="exact"` must reproduce the BF16 correction bit for bit. If it does not, the cache is
mis-keyed and no number from `quant_delta` means anything.

**Why a shared registry.** The approximate and correction paths run on different module objects --
the correction blocks are `copy.deepcopy` of the originals -- so a Linear cannot cache its own input
for its counterpart to read. `ApproxInputCapture` records into a registry keyed by (block index,
site) and `DeltaSplitLinear` reads it back.

**Fake quantization, deliberately.** Both arms quantize-dequantize and multiply in BF16, so the two
differ only in *what* is quantized, never in which kernel ran. The shipped FP4 correction number
(61.4828 mIoU) is a cross-check on the `quant_full` arm, not a term in the comparison.

Cost: the base term is recomputed here rather than read back from the approximate pass, which costs
an extra BF16 GEMM per Linear. A deployment would cache `W a` instead and pay nothing. This is an
accuracy harness; it is slower than the path it measures and must not be timed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .fp4_granularity_linear import fake_quantize

MODES = ("exact", "quant_full", "quant_delta")
SPARSITIES = ("off", "2:4", "unstructured50")


def sparsify(x: torch.Tensor, kind: str) -> torch.Tensor:
    """Zero part of `x` by magnitude. Applied to the delta only -- see the module docstring.

    "2:4" keeps the larger 2 of every contiguous 4 along the contraction dim. That is the one sparse
    layout Blackwell tensor cores accelerate (2x), so it is the only variant with a path to real
    speed; "unstructured50" keeps the top half of each row instead, which is strictly freer and
    therefore bounds what the 2:4 structure costs.
    """
    if kind == "off":
        return x
    if kind == "2:4":
        v = x.reshape(-1, 4)
        idx = v.abs().topk(2, dim=-1).indices
        mask = torch.zeros_like(v, dtype=torch.bool).scatter_(-1, idx, True)
        return (v * mask).reshape(x.shape)
    if kind == "unstructured50":
        flat = x.reshape(-1, x.shape[-1])
        k = max(1, flat.shape[-1] // 2)
        thresh = flat.abs().topk(k, dim=-1).values[:, -1:]
        return (flat * (flat.abs() >= thresh)).reshape(x.shape)
    raise ValueError(f"sparsity must be one of {SPARSITIES}, got {kind!r}")

# (block_index, site) -> the approximate pass's inputs to that Linear, one [B_i, N, C] per approx
# call. `crop_cover` runs the approximate pass once per crop, each with batch 1, while the correction
# runs batched across the crops -- so a single overwritten tensor is short exactly the batch rows the
# correction asks for. They are concatenated in call order, which is the order the correction's batch
# index is built in.
_A_CACHE: dict[tuple[int, str], tuple[int, list[torch.Tensor]]] = {}
# Bumped once per approximate forward. There is no single "request start" hook to clear on:
# `begin_dinov3_correct_event` fires on every correction (so clearing there wipes what the
# approximate pass just wrote -- the bug this replaced), and `begin_dinov3_approx_event` fires once
# per layer range, so clearing everything there wipes earlier blocks the later rounds still need.
# Instead each key remembers the generation it was filled in: a block re-approximated under a newer
# generation starts a fresh list, and a block filled in an earlier round of the *same* request keeps
# its entries. Every block is re-approximated once per request, so this self-clears.
_GEN = [0]
# Rows of `_A_CACHE` the current correction is operating on, published by the correction path.
_SELECTION: dict[str, torch.Tensor | None] = {"batch_idx": None, "token_idx": None}


def reset_cache() -> None:
    """Drop everything (controller setup / teardown)."""
    _A_CACHE.clear()
    _GEN[0] = 0
    _SELECTION["batch_idx"] = _SELECTION["token_idx"] = None


def begin_approx_generation() -> None:
    """Called at the start of each approximate forward; see `_GEN`."""
    _GEN[0] += 1


def set_selection(batch_idx: torch.Tensor | None, token_idx: torch.Tensor | None) -> None:
    """Publish which (batch, token) rows the correction is about to recompute."""
    _SELECTION["batch_idx"] = batch_idx
    _SELECTION["token_idx"] = token_idx


def cache_bytes() -> int:
    return sum(t.numel() * t.element_size() for _, v in _A_CACHE.values() for t in v)


class ApproxInputCapture(nn.Module):
    """Approximate-path Linear that records its input for the correction path to difference against.

    Holds a reference, not a copy: the approximate pass has already finished with the tensor by the
    time correction runs, and cloning 40 blocks' worth would double a 5 GB cache for nothing.
    """

    def __init__(self, inner: nn.Linear, key: tuple[int, str]) -> None:
        super().__init__()
        self.inner = inner
        self.key = key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.detach()
        v = v if v.dim() == 3 else v.unsqueeze(0)
        gen, parts = _A_CACHE.get(self.key, (None, None))
        if gen != _GEN[0]:
            gen, parts = _GEN[0], []
            _A_CACHE[self.key] = (gen, parts)
        parts.append(v)
        return self.inner(x)

    # Callers reach through these blocks for shape metadata -- the correct-bucket warmup reads
    # `attn.qkv.in_features` off the first block, for one. A wrapper that hides them turns into an
    # AttributeError several call frames away from anything mentioning this file.
    @property
    def in_features(self) -> int:
        return self.inner.in_features

    @property
    def out_features(self) -> int:
        return self.inner.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.inner.weight

    @property
    def bias(self):
        return self.inner.bias


class DeltaSplitLinear(nn.Module):
    """Correction-path Linear, quantizing either the whole activation or only the delta."""

    def __init__(
        self, linear: nn.Linear, key: tuple[int, str], mode: str, fmt: str = "fp4",
        sparsity: str = "off",
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if sparsity not in SPARSITIES:
            raise ValueError(f"sparsity must be one of {SPARSITIES}, got {sparsity!r}")
        self.key = key
        self.mode = mode
        self.fmt = fmt
        self.sparsity = sparsity
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w = linear.weight.detach()
        self.register_buffer("weight", w, persistent=False)
        # The weight is quantized once, and identically in both arms, so the comparison isolates the
        # activation. The base term deliberately uses the BF16 weight: in a deployment that term is
        # not recomputed at all, it is the approximate pass's own BF16 result.
        self.register_buffer(
            "weight_q",
            w if fmt == "none" else fake_quantize(w, fmt, "block16"),
            persistent=False,
        )
        self.register_buffer(
            "bias", None if linear.bias is None else linear.bias.detach().clone(), persistent=False
        )

    def _base_rows(self, x: torch.Tensor) -> torch.Tensor | None:
        """The approximate pass's input for the rows `x` holds, or None if unavailable."""
        _gen, parts = _A_CACHE.get(self.key, (None, None))
        bi, ti = _SELECTION["batch_idx"], _SELECTION["token_idx"]

        def _bail(reason: str):
            # Name the branch. "fell back" alone sent me guessing between four different causes.
            if not DeltaSplitLinear._shape_reported:
                DeltaSplitLinear._shape_reported = True
                shapes = [tuple(t.shape) for t in (parts or [])]
                print(f"[delta-split] BAIL {reason} at {self.key}: parts={shapes}, "
                      f"sel={'none' if bi is None else (int(bi.max()), int(ti.max()))}, "
                      f"x={tuple(x.shape)}", flush=True)
            return None

        if not parts:
            return _bail("no-cache")
        if bi is None or ti is None:
            return _bail("no-selection")
        # Token counts vary per image but are equal across one image's crops; an unequal set means
        # the crops were not what this cache is assumed to hold, so refuse rather than pad silently.
        if len({t.shape[1] for t in parts}) != 1:
            return _bail("ragged-tokens")
        a = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        # Bounds check before the gather. Indexing out of range on CUDA is a device-side assert
        # that poisons the context and surfaces as a cuBLAS failure several ops later, so the shape
        # mismatch has to be caught here, on the host, where it can still be named.
        # `.max()` syncs; this is an accuracy harness and the sync is not on any measured path.
        if bi.numel() and (int(bi.max()) >= a.shape[0] or int(ti.max()) >= a.shape[1]):
            return _bail(f"out-of-range cache={tuple(a.shape)}")
        rows = a[bi, ti]
        if rows.shape[-1] != x.shape[-1]:
            return _bail("width")
        n = x.shape[0] if x.dim() == 2 else x.shape[-2]
        if rows.shape[0] == n:
            return rows.to(x.dtype).reshape(x.shape)
        if rows.shape[0] < n:
            # The attention path pads its query rows up to `sdpa_query_bucket_size`. Pad the base
            # with zeros to match: a padded row then carries base 0 and delta x, is quantized on its
            # own per-row block scales so it cannot disturb a real row, and is discarded downstream.
            pad = torch.zeros(
                (n - rows.shape[0], rows.shape[1]), device=rows.device, dtype=rows.dtype
            )
            rows = torch.cat([rows, pad], dim=0)
            return rows.to(x.dtype).reshape(x.shape)
        return rows[:n].to(x.dtype).reshape(x.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "quant_full":
            # Sparsity is never applied here: `a+d` is the answer itself, and zeroing part of it is
            # not a cheaper correction but a hole. The asymmetry is the point of the experiment.
            return F.linear(self._q(x), self.weight_q, self.bias)

        a = self._base_rows(x)
        if a is None:
            # No cached base for this call -- fall back to the whole-activation form rather than
            # silently skipping the correction. Counted so a run cannot pass on fallbacks alone.
            DeltaSplitLinear.fallbacks += 1
            if DeltaSplitLinear.fallbacks == 1:
                # An arm that fell back everywhere still passes the identity gate while having
                # split nothing, so say it the moment it happens rather than at teardown.
                print(f"[delta-split] WARNING: no cached base for {self.key}; fell back to the "
                      "whole-activation path. Results are not a delta measurement.", flush=True)
            if self.mode == "exact":
                return F.linear(x, self.weight, self.bias)
            return F.linear(self._q(x), self.weight_q, self.bias)

        if not DeltaSplitLinear._traced:
            DeltaSplitLinear._traced = True
            print(f"[delta-split] delta path live: key={self.key}, rows={x.shape[0]}, "
                  f"cache={cache_bytes() / 2**20:.0f} MB", flush=True)
        d = sparsify(x - a, self.sparsity)
        base = F.linear(a, self.weight, self.bias)
        if self.mode == "exact":
            return base + F.linear(d, self.weight, None)
        return base + F.linear(self._q(d), self.weight_q, None)

    def _q(self, t: torch.Tensor) -> torch.Tensor:
        return t if self.fmt == "none" else fake_quantize(t, self.fmt, "block16")

    fallbacks = 0
    _traced = False
    _shape_reported = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"key={self.key}, mode={self.mode}"
        )


_SITES = ("attn.qkv", "attn.proj", "mlp.w1", "mlp.w2", "mlp.w3")


def install(approx_blocks: nn.ModuleList, correct_blocks: nn.ModuleList, mode: str,
            fmt: str = "fp4", sparsity: str = "off") -> tuple[int, int]:
    """Capture on the approximate blocks, split on the correction blocks. Returns the two counts.

    `mlp.w1` and `mlp.w2` are fed the same tensor, so they share a cache entry rather than storing it
    twice -- 25 MB per block at ADE20K's 3136 tokens.
    """
    captured = split = 0
    for bi, blk in enumerate(approx_blocks):
        for site in _SITES:
            parent_path, _, attr = site.rpartition(".")
            parent = blk.get_submodule(parent_path)
            child = getattr(parent, attr)
            if isinstance(child, ApproxInputCapture):
                continue
            key = (bi, "mlp.w12" if site in ("mlp.w1", "mlp.w2") else site)
            setattr(parent, attr, ApproxInputCapture(child, key))
            captured += 1
    for bi, blk in enumerate(correct_blocks):
        for site in _SITES:
            parent_path, _, attr = site.rpartition(".")
            parent = blk.get_submodule(parent_path)
            child = getattr(parent, attr)
            if not isinstance(child, nn.Linear):
                raise RuntimeError(f"block {bi} {site} is {type(child).__name__}, expected nn.Linear")
            key = (bi, "mlp.w12" if site in ("mlp.w1", "mlp.w2") else site)
            setattr(parent, attr, DeltaSplitLinear(child, key, mode, fmt, sparsity))
            split += 1
    return captured, split
