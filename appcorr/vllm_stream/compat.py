"""Environment work-arounds for vllm 0.11.2 on this box (not part of the streaming design).

No-ops on 0.28.0: the ViT attention moved to `MMEncoderAttention`, whose backend choice takes
the head size into account."""
from __future__ import annotations


def fix_qwen2_5_vit_upstream_fa() -> None:
    """vllm 0.11.2 `Qwen2_5_VisionAttention.__init__` asks `maybe_get_vit_flash_attn_backend` for
    the ViT backend but drops the `use_upstream_fa=True` it returns (keeps its own False), so
    the tower always runs vLLM's bundled FA2 -- which on this build rejects Qwen2.5-VL's head dim
    80 ("headdim not being a multiple of 32"). `mm_encoder_attn_backend=TORCH_SDPA` does not help:
    the CUDA branch converts any non-FA choice to FLASH_ATTN when upstream flash_attn is importable.
    Upstream flash_attn (2.8.3 here, what HF uses) handles head dim 80, so make the tower use it."""
    from . import vllm_version
    if vllm_version() != "0.11.2":
        return
    from vllm.model_executor.models import qwen2_5_vl as m
    cls = m.Qwen2_5_VisionAttention
    if getattr(cls, "_appcorr_upstream_fa", False):
        return
    orig = cls.__init__

    def __init__(self, *a, **k):
        orig(self, *a, **k)
        from vllm.platforms import current_platform
        if self.is_flash_attn_backend and current_platform.is_cuda():
            self.use_upstream_fa = True

    cls.__init__ = __init__
    cls._appcorr_upstream_fa = True
