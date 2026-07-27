"""Small, machine-independent CUDA toolchain discovery for Triton experiments."""

from __future__ import annotations

import os
from pathlib import Path


def configure_triton_cuda_environment() -> dict[str, str]:
    """Populate missing CUDA build variables before the first Triton launch.

    Some cluster images expose the driver and PyTorch CUDA runtime but omit the
    unversioned CUDA development paths from the login shell.  Triton's small
    launcher extension then fails on ``cuda.h``, ``-lcuda``, or ``ptxas``.
    Existing user values always win.
    """

    candidates = [Path("/usr/local/cuda")]
    candidates.extend(
        sorted(
            Path("/usr/local").glob("cuda-*"),
            key=lambda path: path.name,
            reverse=True,
        )
    )
    cuda_root = next(
        (
            root
            for root in candidates
            if (root / "bin" / "ptxas").exists()
            and (root / "targets" / "x86_64-linux" / "include" / "cuda.h").exists()
        ),
        None,
    )
    if cuda_root is None:
        return {}

    target = cuda_root / "targets" / "x86_64-linux"
    values = {
        "TRITON_PTXAS_PATH": str(cuda_root / "bin" / "ptxas"),
        "CPATH": str(target / "include"),
        "LIBRARY_PATH": str(target / "lib" / "stubs"),
    }
    compat_lib = cuda_root / "compat" / "lib.real"
    if compat_lib.exists():
        values["TRITON_LIBCUDA_PATH"] = str(compat_lib)

    configured: dict[str, str] = {}
    for name, value in values.items():
        if name in {"CPATH", "LIBRARY_PATH"} and os.environ.get(name):
            value = f"{value}{os.pathsep}{os.environ[name]}"
        if name not in os.environ:
            os.environ[name] = value
            configured[name] = value
    return configured
