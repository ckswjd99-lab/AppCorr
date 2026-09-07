"""vLLM general-plugin entry point.

To have the hooks applied in every vLLM process (front-end, engine core, workers) without an
explicit `install()`, register the plugin in the environment that runs vLLM:

    [project.entry-points."vllm.general_plugins"]
    appcorr_stream = "appcorr.vllm_stream.plugin:register"

(vLLM calls every `vllm.general_plugins` entry point from `load_general_plugins()`, which the
engine core and each worker invoke at start-up.) The in-process prototype does not need this --
`StreamingLLM` calls `install()` itself.
"""
from . import register  # noqa: F401
