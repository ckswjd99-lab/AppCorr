from __future__ import annotations

import copy
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch import nn


_FP8_LINEAR_SUFFIXES = ("attn.qkv", "attn.proj", "mlp.w1", "mlp.w2", "mlp.w3")
_COMPILED_CACHE_TAG = "__fp8_layer__"


def _linear_rows(x: torch.Tensor) -> int:
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError(f"Expected a non-empty Linear input, got shape {tuple(x.shape)}")
    return x.numel() // x.shape[-1]


def _find_loaded_libcuda() -> Path | None:
    maps_path = Path("/proc/self/maps")
    if maps_path.exists():
        try:
            for line in maps_path.read_text(encoding="utf-8").splitlines():
                candidate = line.rsplit(" ", 1)[-1]
                if candidate.endswith("/libcuda.so.1"):
                    path = Path(candidate)
                    if path.exists():
                        return path
        except OSError:
            pass
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            check=False,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if "libcuda.so.1" not in line or "=>" not in line:
                continue
            path = Path(line.rsplit("=>", 1)[-1].strip())
            if path.exists():
                return path
    except OSError:
        pass
    return None


def _configure_compile_environment() -> None:
    """Supply writable/executable compiler paths without overriding user choices."""
    existing_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    default_inductor_cache = (
        existing_cache is None
        or (
            Path(existing_cache).parent == Path(tempfile.gettempdir())
            and Path(existing_cache).name.startswith("torchinductor_")
        )
    )
    cache_root = (
        Path.home() / ".cache" / "appcorr" / "torchinductor"
        if default_inductor_cache
        else Path(existing_cache)
    ).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root)

    triton_cache = cache_root / "triton"
    triton_cache.mkdir(parents=True, exist_ok=True)
    existing_triton_cache = os.environ.get("TRITON_CACHE_DIR")
    if existing_triton_cache is None or default_inductor_cache:
        os.environ["TRITON_CACHE_DIR"] = str(triton_cache)

    compile_tmp = cache_root / "tmp"
    compile_tmp.mkdir(parents=True, exist_ok=True)
    if "TMPDIR" not in os.environ:
        os.environ["TMPDIR"] = str(compile_tmp)
        # tempfile may have cached /tmp before the precision controller starts.
        tempfile.tempdir = str(compile_tmp)

    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME:
            cuda_include = str(Path(CUDA_HOME) / "include")
            cpath = os.environ.get("CPATH")
            if cpath:
                if cuda_include not in cpath.split(os.pathsep):
                    os.environ["CPATH"] = f"{cuda_include}{os.pathsep}{cpath}"
            else:
                os.environ["CPATH"] = cuda_include
    except ImportError:
        pass

    if "TRITON_PTXAS_PATH" not in os.environ:
        ptxas = shutil.which("ptxas")
        if ptxas:
            os.environ["TRITON_PTXAS_PATH"] = ptxas

    if "TRITON_LIBCUDA_PATH" not in os.environ:
        libcuda = _find_loaded_libcuda()
        if libcuda is not None:
            lib_dir = cache_root / "lib"
            lib_dir.mkdir(parents=True, exist_ok=True)
            link = lib_dir / "libcuda.so"
            if not link.exists():
                link.symlink_to(libcuda)
            os.environ["TRITON_LIBCUDA_PATH"] = str(lib_dir)
            library_path = os.environ.get("LIBRARY_PATH")
            if library_path:
                os.environ["LIBRARY_PATH"] = f"{lib_dir}{os.pathsep}{library_path}"
            else:
                os.environ["LIBRARY_PATH"] = str(lib_dir)


class DINOv3ApproxPrecisionController:
    """Select BF16 or compiled FP8 blocks for DINOv3 approximate execution."""

    def __init__(
        self,
        blocks: nn.ModuleList,
        *,
        precision: str,
        auto_min_rows: int,
        device: torch.device,
    ) -> None:
        self.blocks = blocks
        self.precision = precision
        self.auto_min_rows = auto_min_rows
        self.device = device
        self.fp8_blocks: nn.ModuleList | None = None
        self._compiled_fp8_approx: list[Any] = []
        self._fp8_unavailable_reason: str | None = None
        self._event_routes: Dict[str, tuple[str, int]] = {}

        # DINOv3's BF16 correction path also uses local Triton kernels. Keep
        # their compiler discovery symmetric with the FP8 path so the default
        # precision does not depend on machine-specific shell setup.
        if self.device.type == "cuda" and torch.cuda.is_available():
            _configure_compile_environment()

        if precision in {"fp8", "auto"}:
            self._initialize_fp8()

    @classmethod
    def from_config(
        cls,
        blocks: nn.ModuleList,
        config: Any,
        device: torch.device,
    ) -> "DINOv3ApproxPrecisionController":
        return cls(
            blocks,
            precision=config.precision,
            auto_min_rows=config.fp8_auto_min_rows,
            device=device,
        )

    @property
    def fp8_available(self) -> bool:
        return self.fp8_blocks is not None

    @property
    def fp8_unavailable_reason(self) -> str | None:
        return self._fp8_unavailable_reason

    def _handle_fp8_unavailable(self, reason: str) -> None:
        self._fp8_unavailable_reason = reason
        if self.precision == "fp8":
            raise RuntimeError(f"precision='fp8' is unavailable: {reason}")
        warnings.warn(
            f"precision='auto' is falling back to BF16: {reason}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _initialize_fp8(self) -> None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            self._handle_fp8_unavailable("FP8 requires a CUDA device")
            return

        capability = torch.cuda.get_device_capability(self.device)
        if capability < (8, 9):
            self._handle_fp8_unavailable(
                f"compute capability {capability[0]}.{capability[1]} is below 8.9"
            )
            return

        _configure_compile_environment()
        try:
            from torchao.quantization import (
                Float8DynamicActivationFloat8WeightConfig,
                quantize_,
            )
            from torchao.quantization.granularity import PerTensor
        except ImportError as exc:
            self._handle_fp8_unavailable(f"TorchAO 0.15+ is required ({exc})")
            return

        fp8_config = Float8DynamicActivationFloat8WeightConfig(
            granularity=PerTensor(),
            set_inductor_config=True,
        )

        fp8_blocks = nn.ModuleList()
        quantized_count = 0
        try:
            for block in self.blocks:
                fp8_block = (
                    copy.deepcopy(block)
                    .to(dtype=torch.bfloat16)
                    .eval()
                    .requires_grad_(False)
                )
                eligible = [
                    (name, module)
                    for name, module in fp8_block.named_modules()
                    if isinstance(module, nn.Linear)
                    and name.endswith(_FP8_LINEAR_SUFFIXES)
                ]
                if len(eligible) != len(_FP8_LINEAR_SUFFIXES):
                    raise RuntimeError(
                        "Expected exactly five FP8 Linear layers per DINOv3 block, "
                        f"found {[name for name, _ in eligible]}"
                    )
                incompatible = [
                    (name, tuple(module.weight.shape))
                    for name, module in eligible
                    if module.in_features % 16 != 0 or module.out_features % 16 != 0
                ]
                if incompatible:
                    raise RuntimeError(
                        f"FP8 _scaled_mm requires dimensions divisible by 16: {incompatible}"
                    )

                eligible_names = {name for name, _ in eligible}
                quantize_(
                    fp8_block,
                    fp8_config,
                    filter_fn=lambda module, fqn, names=eligible_names: (
                        isinstance(module, nn.Linear) and fqn in names
                    ),
                )
                converted = [
                    name
                    for name, module in fp8_block.named_modules()
                    if name in eligible_names
                    and type(module.weight).__name__ == "Float8Tensor"
                ]
                if len(converted) != len(eligible_names):
                    raise RuntimeError(
                        "TorchAO skipped one or more requested FP8 weights: "
                        f"converted={converted}, requested={sorted(eligible_names)}"
                    )
                quantized_count += len(converted)
                fp8_blocks.append(fp8_block)
        except Exception as exc:
            self._handle_fp8_unavailable(str(exc))
            return

        expected_count = len(self.blocks) * len(_FP8_LINEAR_SUFFIXES)
        if quantized_count != expected_count:
            self._handle_fp8_unavailable(
                f"quantized {quantized_count} Linear layers, expected {expected_count}"
            )
            return

        self.fp8_blocks = fp8_blocks
        self._compiled_fp8_approx = [
            torch.compile(block.approx, fullgraph=False, dynamic=False)
            for block in fp8_blocks
        ]
        print(
            "[FP8] Prepared "
            f"{quantized_count} approximate Linear weights across {len(fp8_blocks)} blocks."
        )

    def begin_event(self) -> None:
        self._event_routes = {}

    def _effective_precision(self, rows: int) -> str:
        if self.precision == "bf16" or not self.fp8_available:
            return "bf16"
        if self.precision == "fp8":
            return "fp8"
        return "fp8" if rows >= self.auto_min_rows else "bf16"

    def run_block(
        self,
        layer_idx: int,
        x: torch.Tensor,
        rope: Any,
        cache: Dict[str, Any],
        tag: str,
        *,
        source_key: str = "default",
        **kwargs: Any,
    ):
        rows = _linear_rows(x)
        effective = self._effective_precision(rows)
        self._event_routes[source_key] = (effective, rows)
        with torch.no_grad():
            if effective == "fp8":
                return self._run_fp8_block(
                    layer_idx,
                    x,
                    rope,
                    cache,
                    tag,
                    **kwargs,
                )
            return self.blocks[layer_idx].approx(x, rope, cache, tag, **kwargs)

    def _run_fp8_block(
        self,
        layer_idx: int,
        x: torch.Tensor,
        rope: Any,
        cache: Dict[str, Any],
        tag: str,
        **kwargs: Any,
    ):
        actual_prefix = f"{tag}_"
        compiled_prefix = f"{_COMPILED_CACHE_TAG}_"
        local_cache: Dict[str, Any] = {}
        for key, value in cache.items():
            if key.startswith(actual_prefix):
                local_cache[f"{compiled_prefix}{key[len(actual_prefix):]}"] = value

        if kwargs.get("appcorr_method", "partial_token") == "partial_token":
            self.blocks[layer_idx]._invalidate_partial_token_derived_caches(cache)

        output, local_cache = self._compiled_fp8_approx[layer_idx](
            x,
            rope,
            local_cache,
            _COMPILED_CACHE_TAG,
            **kwargs,
        )
        # TorchAO's dynamic FP8 Linear preserves a float32 residual stream
        # under autocast. The public approx/correct contract keeps activations
        # and feature caches in BF16 so correction does not inherit FP32 KV
        # storage or residual arithmetic from the FP8 clone.
        output = output.to(dtype=torch.bfloat16)

        for key in list(cache):
            if key.startswith(actual_prefix):
                del cache[key]
        for key, value in local_cache.items():
            if (
                key.endswith("_kv")
                and torch.is_tensor(value)
                and torch.is_floating_point(value)
                and value.dtype != torch.bfloat16
            ):
                value = value.to(dtype=torch.bfloat16)
            if key.startswith(compiled_prefix):
                cache[f"{actual_prefix}{key[len(compiled_prefix):]}"] = value
            else:
                cache[key] = value
        return output, cache

    def event_metadata(self) -> Dict[str, Any]:
        effective_set = {precision for precision, _ in self._event_routes.values()}
        if not effective_set:
            effective = "none"
        elif len(effective_set) == 1:
            effective = next(iter(effective_set))
        else:
            effective = "mixed"
        return {
            "approx_precision_requested": self.precision,
            "approx_precision_effective": effective,
            "approx_precision_rows": sorted(
                {rows for _, rows in self._event_routes.values()}
            ),
            "approx_fp8_sources": sum(
                precision == "fp8" for precision, _ in self._event_routes.values()
            ),
            "approx_bf16_sources": sum(
                precision == "bf16" for precision, _ in self._event_routes.values()
            ),
            "fp8_auto_min_rows": self.auto_min_rows,
        }

    def iter_fp8_linears(self) -> Iterable[tuple[str, nn.Linear]]:
        if self.fp8_blocks is None:
            return ()
        return (
            (name, module)
            for name, module in self.fp8_blocks.named_modules()
            if isinstance(module, nn.Linear)
            and name.endswith(_FP8_LINEAR_SUFFIXES)
        )
