"""Shared runtime bootstrap for pi0-FAST LIBERO evaluation scripts.

The local B200 host needs EGL device 2 plus ALLOW_ANY_DEVICE, and LIBERO is installed as a sibling
of the AppCorr checkout rather than in the pi0fast environment. Centralizing those requirements
prevents individual launch commands from silently omitting one of them.
"""

import importlib.util
import os
from pathlib import Path
import sys


APPCORR_ROOT = Path(__file__).resolve().parents[2]


def _candidate_libero_roots() -> list[Path]:
    candidates = []
    configured = os.environ.get("LIBERO_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    # Standard layout on the AppCorr evaluation host:
    #   <workspace>/AppCorr
    #   <workspace>/openvla_deps/LIBERO
    candidates.append(APPCORR_ROOT.parent / "openvla_deps" / "LIBERO")
    return candidates


def configure_pi0fast_libero_runtime() -> Path:
    """Configure import/render defaults before importing LeRobot's LIBERO environment."""
    root_text = str(APPCORR_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    libero_root = None
    if importlib.util.find_spec("libero") is None:
        for candidate in _candidate_libero_roots():
            if (candidate / "libero" / "libero").is_dir():
                libero_root = candidate.resolve()
                libero_text = str(libero_root)
                if libero_text not in sys.path:
                    sys.path.insert(0, libero_text)
                break
        if libero_root is None:
            checked = ", ".join(str(path) for path in _candidate_libero_roots())
            raise RuntimeError(
                "LIBERO is not importable. Set LIBERO_ROOT to the checkout containing "
                f"'libero/libero'. Checked: {checked}"
            )
    else:
        spec = importlib.util.find_spec("libero")
        libero_root = (
            Path(next(iter(spec.submodule_search_locations))).resolve().parent
            if spec is not None and spec.submodule_search_locations
            else Path("<installed>")
        )

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "2")
    os.environ.setdefault("MUJOCO_EGL_ALLOW_ANY_DEVICE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    print(
        "[libero-runtime] "
        f"root={libero_root} "
        f"MUJOCO_GL={os.environ['MUJOCO_GL']} "
        f"EGL_DEVICE={os.environ['MUJOCO_EGL_DEVICE_ID']} "
        f"EGL_ALLOW_ANY={os.environ['MUJOCO_EGL_ALLOW_ANY_DEVICE']} "
        f"TORCHDYNAMO_DISABLE={os.environ['TORCHDYNAMO_DISABLE']}",
        flush=True,
    )
    return libero_root
