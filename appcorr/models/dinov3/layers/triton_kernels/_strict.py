"""Make Triton fallbacks impossible to miss.

A silent fallback is the right behaviour for a service and the wrong behaviour for research. This
repository measures kernel latency, so a fallback that quietly substitutes an eager path does not
degrade gracefully -- it produces a number that looks fine and is wrong, with nothing in the log.

That is not hypothetical. The installed triton wheel here shipped without `backends/nvidia/include`
and without `ptxas`, so **no AppCorr kernel had ever compiled in this environment**; the compile
cache was empty. Every correction run had been using the eager fallbacks, and every correction
latency measured here understated the intended system. Nothing said so.

So: fallbacks raise by default. Set `APPCORR_TRITON_FALLBACK` to change that.

    error   (default) raise. A fallback is a bug until proven otherwise.
    warn              print once per distinct reason and continue.
    silent            the old behaviour. Only for deliberately running without Triton.

`verify_triton_runtime()` is the other half: it compiles a trivial kernel at startup so a broken
install fails immediately with a diagnosis, rather than at the first correction, deep in a run.
"""

import os
import sys
import threading

_MODES = ("error", "warn", "silent")
_ENV = "APPCORR_TRITON_FALLBACK"

_lock = threading.Lock()
_seen: set[str] = set()
_runtime_checked: list = []


def mode() -> str:
    m = os.environ.get(_ENV, "error").strip().lower()
    if m not in _MODES:
        raise ValueError(f"{_ENV} must be one of {_MODES}, got {m!r}")
    return m


class TritonFallbackError(RuntimeError):
    """A Triton kernel declined to run and the eager path would have been used silently."""


def note_fallback(kernel: str, reason: str, *, detail: str = "") -> None:
    """Called wherever a Triton kernel gives up. Raises, warns, or does nothing per `mode()`.

    `reason` should say what about the input the kernel could not handle, specifically enough that
    the reader can decide whether it is a real limitation or a bug in the caller.
    """
    m = mode()
    if m == "silent":
        return

    msg = (
        f"[AppCorr] Triton kernel '{kernel}' fell back to the eager path: {reason}."
        f"{(' ' + detail) if detail else ''}\n"
        f"  Eager fallbacks are numerically equivalent but far slower, so any latency measured\n"
        f"  from here understates the system. If this is expected, set {_ENV}=warn (or =silent).\n"
        f"  If Triton itself is broken, run appcorr.models.dinov3.layers.triton_kernels."
        f"verify_triton_runtime()."
    )
    if m == "error":
        raise TritonFallbackError(msg)

    key = f"{kernel}:{reason}"
    with _lock:
        if key in _seen:
            return
        _seen.add(key)
    print(msg, file=sys.stderr, flush=True)


def verify_triton_runtime(raise_on_failure: bool = True) -> bool:
    """Actually JIT-compile a kernel, to prove the toolchain works. Cached after the first call.

    Import success is not evidence: triton imports fine with no `cuda.h` and no `ptxas`, and only
    fails when something is compiled for real -- which in a long run is the first correction, an
    hour in.
    """
    if _runtime_checked:
        ok, err = _runtime_checked[0]
        if not ok and raise_on_failure:
            raise TritonFallbackError(err)
        return ok

    try:
        import torch
        import triton
        import triton.language as tl

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        @triton.jit
        def _probe(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
            o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = o < n
            tl.store(y_ptr + o, tl.load(x_ptr + o, mask=m) * 2.0, mask=m)

        x = torch.arange(8, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        _probe[(1,)](x, y, 8, BLOCK=8)
        torch.cuda.synchronize()
        if not torch.equal(y, x * 2):
            raise RuntimeError("probe kernel produced the wrong result")
        result = (True, "")
    except Exception as exc:  # noqa: BLE001 - the point is to report anything at all
        result = (
            False,
            "[AppCorr] Triton cannot compile kernels in this environment, so every correction\n"
            f"  would silently use the eager path. Underlying error:\n    {type(exc).__name__}: {exc}\n"
            "  A triton wheel missing backends/nvidia/include (no cuda.h) or backends/nvidia/bin/ptxas\n"
            "  is the usual cause. Without write access to the package, point it at the system CUDA:\n"
            "    export CPATH=/usr/local/cuda/include\n"
            "    export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas\n"
            "    export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump\n"
            "    export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm",
        )

    _runtime_checked.append(result)
    ok, err = result
    if not ok and raise_on_failure:
        raise TritonFallbackError(err)
    return ok
