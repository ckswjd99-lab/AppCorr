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
            # TorchAO renamed this class in 0.17.0. The serving stack (system python) still
            # ships 0.15.0+git, where it is NVFP4InferenceConfig with the same fields, so
            # importing only the new name breaks FP4 on the machine that actually runs evals.
            try:
                from torchao.prototype.mx_formats import (
                    NVFP4DynamicActivationNVFP4WeightConfig as NVFP4Config,
                )
            except ImportError:
                from torchao.prototype.mx_formats import NVFP4InferenceConfig as NVFP4Config
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
        fp4_config = NVFP4Config(
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

    Compilation is **opt-in** via `correct_compile`, because whether it pays depends entirely on the
    workload's selected-token count M. Measured on this stack (MSLK installed,
    docs/memo/dinov3_nvfp4_speedup_gate.md): compiled NVFP4 carries a ~0.54 ms/block fixed cost that
    is flat in M, so it only beats BF16 above M ~= 2300, and it recompiles per distinct shape.
      - ADE20K m2f: M varies per round (median 1028) -> leave compile off; FP4 loses there anyway.
      - ImageNet grid grouping: M is *constant* (69 tokens/image x batch; 2208 @ bs=32, 4416 @ bs=64)
        so exactly one shape is ever compiled and `dynamic=False` is free of recompiles.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        *,
        precision: str,
        device: torch.device,
        compile_enabled: bool = False,
        fp4_calib_events: int = 1,
    ) -> None:
        self.blocks = blocks
        self.precision = precision
        self.device = device
        self.compile_enabled = bool(compile_enabled)
        self.fp4_calib_events = max(0, int(fp4_calib_events))
        self.fp8_blocks: nn.ModuleList | None = None
        self.fp4_blocks: nn.ModuleList | None = None
        self._compiled_correct: dict[int, Any] = {}
        self._fp8_unavailable_reason: str | None = None
        self._fp4_unavailable_reason: str | None = None
        self._event_routes: Dict[str, tuple[str, int]] = {}
        # None once conversion has been attempted (or when the static path is off entirely).
        self._fp4_calib_remaining: int | None = None
        self._fp4_eligible_names: set[str] = set()
        self._fp4_config_factory: Any = None

        # torch.compile of the NVFP4 path dies with a Triton subprocess error on the amax
        # reduction unless these compiler paths are set up first.
        if self.compile_enabled and self.device.type == "cuda" and torch.cuda.is_available():
            _configure_compile_environment()

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
            compile_enabled=bool(getattr(config, "correct_compile", False)),
            fp4_calib_events=int(getattr(config, "correct_fp4_calib_events", 1)),
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
            # TorchAO renamed this class in 0.17.0. The serving stack (system python) still
            # ships 0.15.0+git, where it is NVFP4InferenceConfig with the same fields, so
            # importing only the new name breaks FP4 on the machine that actually runs evals.
            try:
                from torchao.prototype.mx_formats import (
                    NVFP4DynamicActivationNVFP4WeightConfig as NVFP4Config,
                )
            except ImportError:
                from torchao.prototype.mx_formats import NVFP4InferenceConfig as NVFP4Config
            from torchao.quantization import quantize_
        except ImportError as exc:
            self._handle_fp4_unavailable(
                f"TorchAO with prototype NVFP4 support is required ({exc})"
            )
            return

        # The Triton NVFP4 activation-quantization kernel is what makes FP4 viable at all -- eager
        # costs ~2.03 ms/block against BF16's 0.32 ms (0.16x) on the serving stack. It needs
        # ptxas/libcuda on the search path, which is what _configure_compile_environment() sets up;
        # without that call the kernel dies with "Cannot find ptxas" and FP4 falls back to eager.
        #
        # TorchAO 0.17 additionally routes this kernel through MSLK (meta-pytorch/MSLK) and exposes
        # `_mslk_available`; 0.15 -- which is what the serving stack runs -- has neither the symbol
        # nor the dependency, and its Triton path works on its own. Treat a missing symbol as
        # "available" rather than silently dropping to the 6x-slower eager path.
        _configure_compile_environment()
        try:
            from torchao.prototype.mx_formats.kernels import _mslk_available

            use_triton = bool(_mslk_available)
        except ImportError:
            use_triton = True
        calib_events = int(self.fp4_calib_events)

        # Per-tensor scale: `use_dynamic_per_tensor_scale=True` recomputes torch.max(abs(x)) over the
        # whole activation on every call. That scan is memory-bound and grows with M -- 42% of the
        # quantization cost at M=1280, 73% at M=20480 (docs/memo/dinov3_nvfp4_speedup_gate.md). The
        # observer flow replaces it with a scale calibrated once from real correction activations.
        # Calibration is unavoidable at runtime: correction inputs are selected-token activations
        # that do not exist at load time. TorchAO 0.15 has no observer flow, so this is opt-out
        # there rather than a hard failure.
        try:
            from torchao.prototype.mx_formats import NVFP4ObservedLinear
        except ImportError:
            NVFP4ObservedLinear = None
            if calib_events > 0:
                print(
                    "[FP4-correct] TorchAO lacks the NVFP4 observer flow; falling back to a "
                    "dynamic per-tensor scale (correct_fp4_calib_events ignored)."
                )
                calib_events = 0

        def _fp4_config(step: str | None = None):
            if step is None:
                return NVFP4Config(
                    use_triton_kernel=use_triton, use_dynamic_per_tensor_scale=True
                )
            # `step` implies use_dynamic_per_tensor_scale=False (torchao sets it in __post_init__).
            return NVFP4Config(
                use_triton_kernel=use_triton, step=step
            )

        self._fp4_config_factory = _fp4_config
        static = calib_events > 0

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
                self._fp4_eligible_names = eligible_names
                quantize_(
                    fp4_block,
                    _fp4_config("prepare" if static else None),
                    filter_fn=lambda module, fqn, names=eligible_names: (
                        isinstance(module, nn.Linear) and fqn in names
                    ),
                )
                # Verify the swap actually happened. torchao's handlers *return* a replacement
                # module, which quantize_ can only install by assigning into a parent -- on a root
                # Linear the replacement is silently dropped and convert then no-ops, yielding a
                # path that looks quantized in the logs but runs plain BF16.
                if static:
                    converted = [
                        name
                        for name, module in fp4_block.named_modules()
                        if name in eligible_names
                        and isinstance(module, NVFP4ObservedLinear)
                    ]
                else:
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
        self._fp4_calib_remaining = calib_events
        mode = (
            f"static per-tensor scale, calibrating for {calib_events} event(s)"
            if static
            else "dynamic per-tensor scale"
        )
        print(
            "[FP4-correct] Prepared "
            f"{quantized_count} correction Linear weights across {len(fp4_blocks)} blocks "
            f"({mode}, triton={use_triton})."
        )

    def _maybe_convert_fp4_static(self) -> None:
        """Bake the observed activation amax into a static per-tensor scale.

        Until this runs the blocks hold `NVFP4ObservedLinear`, which computes an ordinary BF16
        `F.linear` while recording amax -- so calibration events are numerically exact, just not
        accelerated. After conversion the amax scan is gone from the hot path.
        """
        if self._fp4_calib_remaining is None or self._fp4_calib_remaining > 0:
            return
        self._fp4_calib_remaining = None  # one-shot, even if conversion fails
        try:
            from torchao.quantization import quantize_

            names = self._fp4_eligible_names
            for fp4_block in self.fp4_blocks:
                quantize_(
                    fp4_block,
                    self._fp4_config_factory("convert"),
                    filter_fn=lambda module, fqn, names=names: (
                        isinstance(module, nn.Linear) and fqn in names
                    ),
                )
            bad = [
                f"{idx}.{name}"
                for idx, fp4_block in enumerate(self.fp4_blocks)
                for name, module in fp4_block.named_modules()
                if name in names
                and (
                    type(module.weight).__name__ != "NVFP4Tensor"
                    or getattr(module.weight, "act_per_tensor_scale", None) is None
                )
            ]
            if bad:
                raise RuntimeError(
                    f"convert left {len(bad)} Linear(s) unquantized or without a static "
                    f"activation scale: {bad[:5]}"
                )
        except Exception as exc:
            # Fall back rather than silently serve a half-converted model.
            self._handle_fp4_unavailable(f"static per-tensor scale conversion failed: {exc}")
            return
        print(
            f"[FP4-correct] Converted {len(self.fp4_blocks)} blocks to a static per-tensor "
            "activation scale; the per-call amax scan is now gone."
        )

    def begin_event(self) -> None:
        self._event_routes = {}
        # Convert at the *start of the event after* the last calibration event, so the observers
        # have actually seen data. Converting the moment the counter hits zero would bake in amax=0.
        if self._fp4_calib_remaining is not None:
            if self._fp4_calib_remaining > 0:
                self._fp4_calib_remaining -= 1
            else:
                self._maybe_convert_fp4_static()

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
            fn = blk.correct
            if self.compile_enabled and effective in {"fp8", "fp4"}:
                compiled = self._compiled_correct.get(layer_idx)
                if compiled is None:
                    compiled = torch.compile(blk.correct, fullgraph=False, dynamic=False)
                    self._compiled_correct[layer_idx] = compiled
                fn = compiled
            output, cache = fn(x, dindice, rope, cache, tag, **kwargs)

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
