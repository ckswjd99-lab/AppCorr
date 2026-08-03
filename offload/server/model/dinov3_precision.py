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


_LOW_PRECISION_LINEAR_SUFFIXES = (
    "attn.qkv",
    "attn.proj",
    "mlp.w1",
    "mlp.w2",
    "mlp.w3",
)
_COMPILED_CACHE_TAGS = {
    "fp8": "__fp8_layer__",
    "fp4": "__fp4_layer__",
}


def _linear_rows(x: torch.Tensor) -> int:
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError(f"Expected a non-empty Linear input, got shape {tuple(x.shape)}")
    return x.numel() // x.shape[-1]


def _eligible_linears(block: nn.Module) -> list[tuple[str, nn.Linear]]:
    eligible = [
        (name, module)
        for name, module in block.named_modules()
        if isinstance(module, nn.Linear)
        and name.endswith(_LOW_PRECISION_LINEAR_SUFFIXES)
    ]
    if len(eligible) != len(_LOW_PRECISION_LINEAR_SUFFIXES):
        raise RuntimeError(
            "Expected exactly five low-precision Linear layers per DINOv3 "
            f"block, found {[name for name, _ in eligible]}"
        )
    incompatible = [
        (name, tuple(module.weight.shape))
        for name, module in eligible
        if module.in_features % 16 != 0 or module.out_features % 16 != 0
    ]
    if incompatible:
        raise RuntimeError(
            "Low-precision _scaled_mm requires dimensions divisible by 16: "
            f"{incompatible}"
        )
    return eligible


def _eligible_fp4_linears(block: nn.Module) -> list[tuple[str, nn.Linear]]:
    eligible = _eligible_linears(block)
    incompatible = [
        (name, tuple(module.weight.shape))
        for name, module in eligible
        if module.in_features % 32 != 0
    ]
    if incompatible:
        raise RuntimeError(
            "Packed FP4 _scaled_mm requires in_features divisible by 32: "
            f"{incompatible}"
        )
    return eligible


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
    """Select BF16, FP8, or FP4 blocks for DINOv3 approximate execution."""

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
        self.fp4_blocks: nn.ModuleList | None = None
        self._compiled_fp8_approx: list[Any] = []
        self._compiled_fp4_approx: list[Any] = []
        self._fp8_unavailable_reason: str | None = None
        self._fp4_unavailable_reason: str | None = None
        self._event_routes: Dict[str, tuple[str, int]] = {}

        # DINOv3's BF16 correction path also uses local Triton kernels. Keep
        # their compiler discovery symmetric with the FP8 path so the default
        # precision does not depend on machine-specific shell setup.
        if self.device.type == "cuda" and torch.cuda.is_available():
            _configure_compile_environment()

        if precision in {"fp8", "auto"}:
            self._initialize_fp8()
        elif precision == "fp4":
            self._initialize_fp4()

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

    @property
    def fp4_available(self) -> bool:
        return self.fp4_blocks is not None

    @property
    def fp4_unavailable_reason(self) -> str | None:
        return self._fp4_unavailable_reason

    def _handle_fp8_unavailable(self, reason: str) -> None:
        self._fp8_unavailable_reason = reason
        if self.precision == "fp8":
            raise RuntimeError(f"precision='fp8' is unavailable: {reason}")
        warnings.warn(
            f"precision='auto' is falling back to BF16: {reason}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _handle_fp4_unavailable(self, reason: str) -> None:
        self._fp4_unavailable_reason = reason
        raise RuntimeError(f"precision='fp4' is unavailable: {reason}")

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
                eligible = _eligible_linears(fp8_block)

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

        expected_count = len(self.blocks) * len(_LOW_PRECISION_LINEAR_SUFFIXES)
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

    def _initialize_fp4(self) -> None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            self._handle_fp4_unavailable("FP4 requires a CUDA device")
            return

        capability = torch.cuda.get_device_capability(self.device)
        if capability < (10, 0):
            self._handle_fp4_unavailable(
                f"compute capability {capability[0]}.{capability[1]} is below 10.0"
            )
            return

        _configure_compile_environment()
        try:
            from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
            from torchao.quantization import quantize_
        except ImportError as exc:
            self._handle_fp4_unavailable(
                f"TorchAO with prototype NVFP4 support is required ({exc})"
            )
            return

        # This used to be use_triton_kernel=True / use_dynamic_per_tensor_scale=False: the fastest
        # TorchAO 0.15 path, which skipped the tensor-wide amax reduction and outer per-tensor scale,
        # trading accuracy for latency. On the currently installed TorchAO (0.17.0) that combination
        # cannot run at all -- the Triton NVFP4 kernel asserts `per_tensor_scale is not None`, and
        # with the scale enabled it then requires the MSLK package
        # (https://github.com/pytorch/MSLK), which is not installed here. Switched to the eager,
        # accurate path so `precision=fp4` runs on this machine.
        #
        # Consequence: FP4 approx numbers measured now are NOT directly comparable to the historical
        # ImageNet/COCO FP4 figures in docs/memo/dinov3_approx_low_precision_status.md (those used
        # the faster/less accurate mode on the older TorchAO snapshot), and the latency advantage
        # that motivated the old setting is gone with the fused kernel.
        fp4_config = NVFP4DynamicActivationNVFP4WeightConfig(
            use_triton_kernel=False,
            use_dynamic_per_tensor_scale=True,
        )

        fp4_blocks = nn.ModuleList()
        quantized_count = 0
        try:
            for block in self.blocks:
                fp4_block = (
                    copy.deepcopy(block)
                    .to(dtype=torch.bfloat16)
                    .eval()
                    .requires_grad_(False)
                )
                eligible = _eligible_fp4_linears(fp4_block)
                eligible_names = {name for name, _ in eligible}
                quantize_(
                    fp4_block,
                    fp4_config,
                    filter_fn=lambda module, fqn, names=eligible_names: (
                        isinstance(module, nn.Linear) and fqn in names
                    ),
                )
                converted = [
                    name
                    for name, module in fp4_block.named_modules()
                    if name in eligible_names
                    and type(module.weight).__name__ == "NVFP4Tensor"
                ]
                if len(converted) != len(eligible_names):
                    raise RuntimeError(
                        "TorchAO skipped one or more requested FP4 weights: "
                        f"converted={converted}, requested={sorted(eligible_names)}"
                    )
                quantized_count += len(converted)
                fp4_blocks.append(fp4_block)
        except Exception as exc:
            self._handle_fp4_unavailable(str(exc))
            return

        expected_count = len(self.blocks) * len(_LOW_PRECISION_LINEAR_SUFFIXES)
        if quantized_count != expected_count:
            self._handle_fp4_unavailable(
                f"quantized {quantized_count} Linear layers, expected {expected_count}"
            )
            return

        self.fp4_blocks = fp4_blocks
        self._compiled_fp4_approx = [
            torch.compile(block.approx, fullgraph=False, dynamic=False)
            for block in fp4_blocks
        ]
        print(
            "[FP4] Prepared "
            f"{quantized_count} approximate Linear weights across {len(fp4_blocks)} blocks."
        )

    def begin_event(self) -> None:
        self._event_routes = {}

    def _effective_precision(self, rows: int) -> str:
        if self.precision == "bf16":
            return "bf16"
        if self.precision == "fp4":
            return "fp4" if self.fp4_available else "bf16"
        if not self.fp8_available:
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
                return self._run_low_precision_block(
                    "fp8",
                    layer_idx,
                    x,
                    rope,
                    cache,
                    tag,
                    **kwargs,
                )
            if effective == "fp4":
                return self._run_low_precision_block(
                    "fp4",
                    layer_idx,
                    x,
                    rope,
                    cache,
                    tag,
                    **kwargs,
                )
            return self.blocks[layer_idx].approx(x, rope, cache, tag, **kwargs)

    def _run_low_precision_block(
        self,
        precision: str,
        layer_idx: int,
        x: torch.Tensor,
        rope: Any,
        cache: Dict[str, Any],
        tag: str,
        **kwargs: Any,
    ):
        compiled_approx = (
            self._compiled_fp8_approx
            if precision == "fp8"
            else self._compiled_fp4_approx
        )
        actual_prefix = f"{tag}_"
        compiled_tag = _COMPILED_CACHE_TAGS[precision]
        compiled_prefix = f"{compiled_tag}_"
        local_cache: Dict[str, Any] = {}
        for key, value in cache.items():
            if key.startswith(actual_prefix):
                local_cache[f"{compiled_prefix}{key[len(actual_prefix):]}"] = value

        if kwargs.get("appcorr_method", "partial_token") == "partial_token":
            self.blocks[layer_idx]._invalidate_partial_token_derived_caches(cache)

        output, local_cache = compiled_approx[layer_idx](
            x,
            rope,
            local_cache,
            compiled_tag,
            **kwargs,
        )
        # The public approx/correct contract keeps activations and feature
        # caches in BF16 regardless of the low-precision GEMM format.
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
            "approx_fp4_sources": sum(
                precision == "fp4" for precision, _ in self._event_routes.values()
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
            and name.endswith(_LOW_PRECISION_LINEAR_SUFFIXES)
        )

    def iter_fp4_linears(self) -> Iterable[tuple[str, nn.Linear]]:
        if self.fp4_blocks is None:
            return ()
        return (
            (name, module)
            for name, module in self.fp4_blocks.named_modules()
            if isinstance(module, nn.Linear)
            and name.endswith(_LOW_PRECISION_LINEAR_SUFFIXES)
        )


class DINOv3CorrectPrecisionController:
    """Select BF16, FP8, or FP4 blocks for DINOv3 CORRECT execution -- the opposite direction of
    DINOv3ApproxPrecisionController. The coarse `.approx()` pass stays at whatever precision the
    model already runs (bf16 via autocast, or independently controlled by
    DINOv3ApproxPrecisionController); `.correct()` -- which only recomputes a selected subset of
    tokens each round -- runs its 5 eligible Linear layers (qkv/proj/w1/w2/w3) in FP8 or FP4 instead.
    Goal: reduce the (already smaller, selected-token) correction GEMM's theoretical FLOP/byte cost
    further, without touching approx's accuracy contribution.

    Unlike the approx controller, this class does NOT torch.compile the quantized `.correct()` path:
    `.correct()`'s selected-token count varies every round (token_keep_ratio/threshold-dependent), so
    a `dynamic=False` compile would recompile on every distinct shape. TorchAO's quantized nn.Linear
    (tensor-subclass dispatch) is correct in eager mode -- torch.compile there is a pure speed
    optimization on top -- and this controller only targets theoretical compute reduction / accuracy
    right now, not wall-clock latency, so eager-mode quantized `.correct()` is used directly.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        *,
        precision: str,
        device: torch.device,
    ) -> None:
        self.blocks = blocks
        self.precision = precision
        self.device = device
        self.fp8_blocks: nn.ModuleList | None = None
        self.fp4_blocks: nn.ModuleList | None = None
        self._fp8_unavailable_reason: str | None = None
        self._fp4_unavailable_reason: str | None = None
        self._event_routes: Dict[str, tuple[str, int]] = {}

        if precision == "fp8":
            self._initialize_fp8()
        elif precision == "fp4":
            self._initialize_fp4()

    @classmethod
    def from_config(
        cls,
        blocks: nn.ModuleList,
        config: Any,
        device: torch.device,
    ) -> "DINOv3CorrectPrecisionController":
        return cls(
            blocks,
            precision=config.correct_precision,
            device=device,
        )

    @property
    def fp8_available(self) -> bool:
        return self.fp8_blocks is not None

    @property
    def fp8_unavailable_reason(self) -> str | None:
        return self._fp8_unavailable_reason

    @property
    def fp4_available(self) -> bool:
        return self.fp4_blocks is not None

    @property
    def fp4_unavailable_reason(self) -> str | None:
        return self._fp4_unavailable_reason

    def _handle_fp8_unavailable(self, reason: str) -> None:
        self._fp8_unavailable_reason = reason
        if self.precision == "fp8":
            raise RuntimeError(f"correct_precision='fp8' is unavailable: {reason}")

    def _handle_fp4_unavailable(self, reason: str) -> None:
        self._fp4_unavailable_reason = reason
        if self.precision == "fp4":
            raise RuntimeError(f"correct_precision='fp4' is unavailable: {reason}")

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
                eligible = _eligible_linears(fp8_block)
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

        expected_count = len(self.blocks) * len(_LOW_PRECISION_LINEAR_SUFFIXES)
        if quantized_count != expected_count:
            self._handle_fp8_unavailable(
                f"quantized {quantized_count} Linear layers, expected {expected_count}"
            )
            return

        self.fp8_blocks = fp8_blocks
        print(
            "[FP8-correct] Prepared "
            f"{quantized_count} correction Linear weights across {len(fp8_blocks)} blocks."
        )

    def _initialize_fp4(self) -> None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            self._handle_fp4_unavailable("FP4 requires a CUDA device")
            return

        capability = torch.cuda.get_device_capability(self.device)
        if capability < (10, 0):
            self._handle_fp4_unavailable(
                f"compute capability {capability[0]}.{capability[1]} is below 10.0"
            )
            return

        try:
            from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
            from torchao.quantization import quantize_
        except ImportError as exc:
            self._handle_fp4_unavailable(
                f"TorchAO with prototype NVFP4 support is required ({exc})"
            )
            return

        # use_triton_kernel=False: the Triton NVFP4 kernel requires the MSLK package
        # (https://github.com/pytorch/MSLK), which is not installed here, and this controller
        # targets a theoretical compute-reduction / accuracy measurement, not latency, so the
        # (correctness-equivalent, just non-fused) eager dispatch path is used instead.
        # use_dynamic_per_tensor_scale=True (the accurate mode): for the same reason there is no
        # reason to trade the extra approximation error the disabled-scale fast path (used by the
        # approx controller's FP4 config above) accepts for lower latency.
        fp4_config = NVFP4DynamicActivationNVFP4WeightConfig(
            use_triton_kernel=False,
            use_dynamic_per_tensor_scale=True,
        )

        fp4_blocks = nn.ModuleList()
        quantized_count = 0
        try:
            for block in self.blocks:
                fp4_block = (
                    copy.deepcopy(block)
                    .to(dtype=torch.bfloat16)
                    .eval()
                    .requires_grad_(False)
                )
                eligible = _eligible_fp4_linears(fp4_block)
                eligible_names = {name for name, _ in eligible}
                quantize_(
                    fp4_block,
                    fp4_config,
                    filter_fn=lambda module, fqn, names=eligible_names: (
                        isinstance(module, nn.Linear) and fqn in names
                    ),
                )
                converted = [
                    name
                    for name, module in fp4_block.named_modules()
                    if name in eligible_names
                    and type(module.weight).__name__ == "NVFP4Tensor"
                ]
                if len(converted) != len(eligible_names):
                    raise RuntimeError(
                        "TorchAO skipped one or more requested FP4 weights: "
                        f"converted={converted}, requested={sorted(eligible_names)}"
                    )
                quantized_count += len(converted)
                fp4_blocks.append(fp4_block)
        except Exception as exc:
            self._handle_fp4_unavailable(str(exc))
            return

        expected_count = len(self.blocks) * len(_LOW_PRECISION_LINEAR_SUFFIXES)
        if quantized_count != expected_count:
            self._handle_fp4_unavailable(
                f"quantized {quantized_count} Linear layers, expected {expected_count}"
            )
            return

        self.fp4_blocks = fp4_blocks
        print(
            "[FP4-correct] Prepared "
            f"{quantized_count} correction Linear weights across {len(fp4_blocks)} blocks."
        )

    def begin_event(self) -> None:
        self._event_routes = {}

    def _effective_precision(self) -> str:
        if self.precision == "fp8":
            return "fp8" if self.fp8_available else "bf16"
        if self.precision == "fp4":
            return "fp4" if self.fp4_available else "bf16"
        return "bf16"

    def run_block(
        self,
        layer_idx: int,
        x: torch.Tensor,
        dindice: torch.Tensor,
        rope: Any,
        cache: Dict[str, Any],
        tag: str,
        *,
        source_key: str = "default",
        **kwargs: Any,
    ):
        rows = _linear_rows(x)
        effective = self._effective_precision()
        self._event_routes[source_key] = (effective, rows)

        if effective == "fp8":
            blk = self.fp8_blocks[layer_idx]
        elif effective == "fp4":
            blk = self.fp4_blocks[layer_idx]
        else:
            blk = self.blocks[layer_idx]

        with torch.no_grad():
            output, cache = blk.correct(x, dindice, rope, cache, tag, **kwargs)

        # Keep the public approx/correct contract: activations and feature caches stay bf16
        # regardless of the low-precision GEMM format used internally.
        output = output.to(dtype=torch.bfloat16)
        actual_prefix = f"{tag}_"
        for key, value in list(cache.items()):
            if (
                key.startswith(actual_prefix)
                and key.endswith("_kv")
                and torch.is_tensor(value)
                and torch.is_floating_point(value)
                and value.dtype != torch.bfloat16
            ):
                cache[key] = value.to(dtype=torch.bfloat16)
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
            "correct_precision_requested": self.precision,
            "correct_precision_effective": effective,
            "correct_precision_rows": sorted(
                {rows for _, rows in self._event_routes.values()}
            ),
            "correct_fp8_sources": sum(
                precision == "fp8" for precision, _ in self._event_routes.values()
            ),
            "correct_fp4_sources": sum(
                precision == "fp4" for precision, _ in self._event_routes.values()
            ),
            "correct_bf16_sources": sum(
                precision == "bf16" for precision, _ in self._event_routes.values()
            ),
        }
