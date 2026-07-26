"""Shared host bootstrap for standalone NORA LIBERO experiments."""

import importlib.util
import os
from pathlib import Path
import sys


APPCORR_ROOT = Path(__file__).resolve().parents[2]


def configure_nora_libero_runtime() -> Path:
    root_text = str(APPCORR_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    openvla_root = APPCORR_ROOT.parent / "openvla"
    if openvla_root.is_dir() and str(openvla_root) not in sys.path:
        sys.path.insert(0, str(openvla_root))

    candidates = []
    if os.environ.get("LIBERO_ROOT"):
        candidates.append(Path(os.environ["LIBERO_ROOT"]).expanduser())
    candidates.append(APPCORR_ROOT.parent / "openvla_deps" / "LIBERO")
    if importlib.util.find_spec("libero") is None:
        for candidate in candidates:
            if (candidate / "libero" / "libero").is_dir():
                sys.path.insert(0, str(candidate.resolve()))
                break
        else:
            raise RuntimeError(
                "LIBERO is not importable; set LIBERO_ROOT to its checkout"
            )

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "2")
    os.environ.setdefault("MUJOCO_EGL_ALLOW_ANY_DEVICE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(
        "[nora-runtime] "
        f"MUJOCO_GL={os.environ['MUJOCO_GL']} "
        f"EGL_DEVICE={os.environ['MUJOCO_EGL_DEVICE_ID']}",
        flush=True,
    )
    return APPCORR_ROOT
