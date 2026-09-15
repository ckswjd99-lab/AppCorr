"""Worker-side half of AppCorr's streaming/correct path under tensor parallelism.

At `tensor_parallel_size == 1` vLLM uses `UniProcExecutor` and the `GPUModelRunner` lives in the
caller's process, so `StreamingLLM.runner.appcorr_*(...)` is a plain method call. At TP > 1 vLLM
uses `MultiProcExecutor` -- one worker PROCESS per rank
(`vllm/config/parallel.py:966-1007`; `vllm/v1/executor/multiproc_executor.py:114-200`) -- and
there is no in-process runner at all. This module is the two things that gap needs:

1. **`AppcorrWorkerExtension`**, passed to the engine as
   `worker_extension_cls="appcorr.vllm_stream.tp_worker.AppcorrWorkerExtension"`. vLLM resolves
   it by qualified name inside `WorkerWrapperBase.init_worker`
   (`vllm/v1/worker/worker_base.py:285-310`) and dynamically adds it to the worker class's bases.
   Resolving the name IMPORTS THIS MODULE in every worker process, and this module calls
   `appcorr.vllm_stream.install()` at import -- which is how the runner patch and the correct
   patch get into each rank. That happens inside `init_worker`, i.e. before `init_device` and
   `load_model`, and therefore long before the first `execute_model`.

   Why not the two alternatives (`docs/memo/glm53_tp_plan.md` §1): a `vllm.general_plugins` entry
   point is correct but needs the package pip-installed, and `bin/pip` is broken in the served
   env; relying on `fork` to inherit the parent's already-patched class is silently wrong,
   because `_maybe_force_spawn` (`vllm/utils/system_utils.py:125-164`) switches the start method
   to `spawn` whenever CUDA is already initialised in the parent -- which it is by the time the
   executor is built.

2. **RPC entry points**, called through `Executor.collective_rpc`. `run_method`
   (`vllm/v1/serial_utils.py:486-510`) resolves a `str` with `getattr(worker, name)`, so every
   public name here must be `appcorr_`-prefixed: `init_worker` asserts that the extension shares
   no attribute name with the worker class (`worker_base.py:293-298`).

Argument discipline. A `collective_rpc` argument tuple is pickled into the shared-memory
broadcast queue, which has a torch reducer for **CPU** tensors only
(`vllm/distributed/device_communicators/shm_broadcast.py:421-449, 823-855`). So the CALLER sends
CPU tensors and the entry points below move them onto the rank's own device -- reproducing
exactly what the TP=1 in-process caller hands the runner. Every rank receives byte-identical
arguments (one enqueue, one shared buffer, every reader), which is what makes a replicated
rewrite -- the MLA latent, the sparse indexer's caches -- identical across ranks by construction
rather than by check. Side buffers are module state in `correct.py` inside each worker process,
so they are per-rank and rank-local for free; that is the right shape, because the KDA conv and
recurrent states are sharded by head (`vllm/model_executor/layers/mamba/abstract.py:59-61`) and a
re-scan is therefore rank-local.

Nothing here runs at TP=1: `StreamingLLM` keeps the direct in-process call so the Qwen3.5 /
GLM-4.6V campaigns are byte-identical to what they measured.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

WORKER_EXTENSION_CLS = "appcorr.vllm_stream.tp_worker.AppcorrWorkerExtension"

# Importing this module installs the hooks in THIS process. In a worker that happens inside
# `init_worker`; in the parent it is a no-op repeat of what `StreamingLLM.__init__` already did.
# Guarded so the module stays importable from a process without vllm (the CPU dispatch test
# imports `to_cpu` / `runner_call` with a fake executor and no engine at all).
try:                                   # pragma: no cover - exercised only where vllm exists
    from . import install as _install

    _install()
    INSTALLED = True
except Exception as _e:                # noqa: BLE001
    INSTALLED = False
    INSTALL_ERROR = repr(_e)


def to_cpu(obj: Any) -> Any:
    """Recursively detach torch tensors onto the CPU, leaving everything else alone.

    Applied to every `collective_rpc` argument at TP > 1. A CUDA tensor would fall through to
    torch's default reducer (CUDA IPC), which is not something to rely on across vLLM's worker
    processes; and the receiving rank's device is not the sender's anyway.
    """
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj


def _to_device(obj: Any, device) -> Any:
    """The inverse of `to_cpu` on the worker side: tensors onto this rank's device, dtypes kept."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def runner_call(worker, fn: Callable, args: tuple = (), kwargs: Optional[dict] = None):
    """`collective_rpc` adapter for an arbitrary callable over the rank's `GPUModelRunner`.

    `collective_rpc(runner_call, args=(fn, args, kwargs))` runs `fn(model_runner, *args, **kwargs)`
    on every rank and returns the per-rank results. `fn` is cloudpickled -- a function defined in
    a gate script's `__main__` is pickled BY VALUE, which is exactly what lets a gate ship its own
    comparison to the ranks without that comparison living in this package.

    Return values travel back through the same pickle path, so they must be CPU tensors or plain
    Python. `correct.appcorr_snapshot` already returns CPU tensors; a gate body that builds its
    own must do the same.
    """
    return fn(worker.model_runner, *args, **(kwargs or {}))


def _patch_flags(runner) -> dict:
    """Patch flags off the runner's OWN class, not the imported V1 symbol.

    vLLM main instantiates a V2 GPUModelRunner our hooks never touch, so reading the flags off
    `vllm.v1.worker.gpu_model_runner.GPUModelRunner` answers about a class that may not be the one
    serving -- False while the live runner is patched, or True while it is not. This function is a
    gate's only evidence that the patch is in on this rank, and a gate that can lie is worse than
    no gate (see docs/memo, "patch target must be the instantiated class"). Walks the MRO because
    install() sets the flag on whichever class it patched, which may be a base of the instance's.
    """
    cls = type(runner)
    out = {}
    for key, attr in (("stream_patched", "_appcorr_stream_patched"),
                      ("correct_patched", "_appcorr_correct_patched")):
        out[key] = any(bool(getattr(k, attr, False)) for k in cls.__mro__)
    out["runner_class"] = f"{cls.__module__}.{cls.__qualname__}"
    return out


class AppcorrWorkerExtension:
    """Mixed into the worker class at `init_worker`. `self` is the vLLM `Worker`."""

    # --- generic runner method dispatch -------------------------------------------------------
    def appcorr_rpc(self, method: str, args: tuple = (), kwargs: Optional[dict] = None):
        """Call `GPUModelRunner.<method>(*args, **kwargs)` on this rank.

        Only `appcorr_`-prefixed runner methods are reachable: the RPC name arrives over a queue,
        and a generic `getattr` on the runner would turn the correction channel into an arbitrary
        method-call channel into the worker.
        """
        if not str(method).startswith("appcorr_"):
            raise ValueError(f"appcorr_rpc refuses {method!r}: only appcorr_* runner methods")
        runner = self.model_runner
        fn = getattr(runner, method, None)
        if fn is None:
            raise AttributeError(
                f"{type(runner).__name__} has no {method!r} -- appcorr.vllm_stream.install() did "
                f"not run in this worker (INSTALLED={INSTALLED})")
        dev = runner.device
        return fn(*_to_device(tuple(args), dev), **_to_device(dict(kwargs or {}), dev))

    # --- ops that need the rank's own device / state, not a value from the caller --------------
    def appcorr_open_buffer(self, req_id: str, n: int, *, lo: int = 0, hi: int = 0,
                            capture_out: bool = False, open_walk: int = 0) -> dict:
        """`correct.open_buffer` on this rank. The device is the RANK's, never the caller's."""
        from . import correct as _correct

        _correct.check_gdn_path(self.model_runner.vllm_config)
        sb = _correct.open_buffer(req_id, int(n), self.model_runner.device, lo=int(lo), hi=int(hi),
                                  capture_out=bool(capture_out), open_walk=int(open_walk))
        return {"n": int(sb.n), "device": str(sb.device)}

    def appcorr_scatter_prompt(self, req_id: str, pos_cpu, rows_cpu) -> int:
        """Mirror a corrected band into this rank's `CachedRequestState.prompt_embeds`.

        The worker keeps its own CPU copy of the prompt; a preempted request re-prefills from it,
        so it must carry the corrected rows on every rank. At TP=1 `StreamingLLM.correct` writes
        this directly; the arithmetic here is the same `_cpu_scatter_rows` for the same reason
        (torch's CPU `index_put_` fans out over the OpenMP pool).
        """
        from .client import _cpu_scatter_rows

        st = self.model_runner.requests.get(req_id)
        if st is None:
            return 0
        rows = rows_cpu.to(st.prompt_embeds.dtype)
        _cpu_scatter_rows(st.prompt_embeds, pos_cpu, rows)
        return int(pos_cpu.numel())

    def appcorr_free(self, req_id: str) -> bool:
        """Drop this rank's side buffers for a finished request.

        At TP=1 `StreamingScheduler._free_request` calls `correct.free` in the same process. At
        TP>1 the scheduler runs in the engine-core process and frees nothing in the workers, so
        `StreamingLLM.step` dispatches this when a request finishes. Without it every corrected
        request leaks its side buffer (1.06 MiB per token per rank on GLM-5.3) until the worker
        dies.
        """
        from . import correct as _correct

        had = _correct.get_buffer(req_id) is not None
        _correct.free(req_id)
        return had

    # --- introspection for gates --------------------------------------------------------------
    def appcorr_worker_info(self) -> dict:
        """Enough to prove (a) from the driver: the patch is in, on this rank, on this device."""
        runner = self.model_runner
        pc = runner.vllm_config.parallel_config
        return {
            "rank": int(getattr(self, "rank", -1)),
            "local_rank": int(getattr(self, "local_rank", -1)),
            "device": str(runner.device),
            "installed": bool(INSTALLED),
            **_patch_flags(runner),
            "has_correct_step": hasattr(runner, "appcorr_correct_step"),
            "tensor_parallel_size": int(pc.tensor_parallel_size),
            # SP inserts collectives INSIDE the decoder layer around attention
            # (`vllm/models/glm5next/nvidia/model.py:366-369, 473-474`), which a per-rank row
            # rewrite is not aligned to. It is off at DP=1 by construction
            # (`vllm/config/parallel.py:714-730`); asserted rather than assumed.
            "sequence_parallel_moe": bool(pc.use_sequence_parallel_moe),
            "data_parallel_size": int(pc.data_parallel_size),
        }


class OpDispatchMixin:
    """Driver-side op dispatch, mixed into `StreamingLLM`.

    One contract at any TP: a runner op goes through `_dispatch`, a worker-extension op through
    `_dispatch_worker`, and a gate's own callable through `run_on_ranks`. TP=1 takes the
    in-process branch of each, which is byte-for-byte the call the code made before this existed
    -- the Qwen3.5 / GLM-4.6V campaigns run there and their numbers must not move.

    It lives in this module rather than in `client.py` so it can be imported, and its
    every-rank / identical-args / per-rank-buffer properties tested against a fake executor, in a
    process with no vllm (`tests/test_glm53_tp_dispatch.py`). The host class must provide
    `tensor_parallel_size`, `executor` and `runner`.
    """

    @property
    def executor(self):
        """The engine's `Executor` -- `UniProcExecutor` at TP=1, `MultiProcExecutor` above
        (`vllm/v1/engine/core.py:134`). Both implement `collective_rpc`."""
        return self.core.model_executor

    def _dispatch(self, method: str, *args, **kwargs):
        """Run `GPUModelRunner.<method>` on every rank; return rank 0's result.

        TP=1 is the EXACT call the code made before this method existed -- the in-process runner,
        the caller's own (GPU) tensors, no copy, no executor. That is deliberate: the Qwen3.5 and
        GLM-4.6V campaigns run at TP=1 and their numbers must not move.

        TP>1 goes through `collective_rpc`, which pickles the arguments into the shared-memory
        broadcast queue -- CPU tensors only (`shm_broadcast.py:421-449`) -- and every rank reads
        the same buffer, so all ranks get byte-identical arguments and the worker moves them onto
        its own device (`tp_worker.AppcorrWorkerExtension.appcorr_rpc`). Ranks disagreeing is
        therefore not a thing that can happen here; what CAN differ is rank-local sharded state
        (the KDA conv/recurrent slices), which is exactly the state a rank-local re-scan owns.
        """
        if self.tensor_parallel_size == 1:
            return getattr(self.runner, method)(*args, **kwargs)
        out = self.executor.collective_rpc(
            "appcorr_rpc", args=(method, to_cpu(tuple(args)), to_cpu(dict(kwargs))))
        return out[0] if isinstance(out, (list, tuple)) else out

    def _dispatch_worker(self, method: str, *args, **kwargs):
        """As `_dispatch` but for an op that lives on the WORKER extension rather than on the
        runner (it needs the rank's own device or its own module state). At TP=1 there is no
        extension, so the caller passes an `inproc=` callable for that branch."""
        inproc = kwargs.pop("inproc")
        if self.tensor_parallel_size == 1:
            return inproc()
        out = self.executor.collective_rpc(
            method, args=to_cpu(tuple(args)), kwargs=to_cpu(dict(kwargs)))
        return out[0] if isinstance(out, (list, tuple)) else out

    def run_on_ranks(self, fn, *args, **kwargs) -> list:
        """Run `fn(model_runner, *args, **kwargs)` on EVERY rank; return the per-rank results.

        The gate tooling's hook: a gate ships its own comparison (K's KDA re-scan-vs-stock, M's
        indexer/latent snapshots) as a cloudpickled callable instead of reaching for a runner that
        does not exist in this process. A function defined in a gate script's `__main__` is
        pickled by value, so the gate keeps its logic; only the transport is here.

        Returns a LIST, length 1 at TP=1 and `tensor_parallel_size` above, so a caller aggregates
        rank 0 and asserts agreement across ranks with the same code at any TP. `fn` must return
        CPU tensors / plain Python (the reply crosses the same pickle path).
        """
        if self.tensor_parallel_size == 1:
            return [fn(self.runner, *args, **kwargs)]
        out = self.executor.collective_rpc(
            runner_call, args=(fn, to_cpu(tuple(args)), to_cpu(dict(kwargs))))
        return list(out) if isinstance(out, (list, tuple)) else [out]

    def worker_info(self) -> list:
        """Per-rank `{rank, device, installed, *_patched, sequence_parallel_moe, ...}`.

        This is the (a) check of `docs/memo/glm53_tp_plan.md` reduced to one call: if the
        extension did not import our package in a worker, that rank is missing from the list or
        reports `correct_patched=False`."""
        if self.tensor_parallel_size == 1:
            r = self.runner
            pc = r.vllm_config.parallel_config
            return [{"rank": 0, "device": str(r.device), "installed": True,
                     **_patch_flags(r),
                     "has_correct_step": hasattr(r, "appcorr_correct_step"),
                     "tensor_parallel_size": 1,
                     "sequence_parallel_moe": bool(pc.use_sequence_parallel_moe),
                     "data_parallel_size": int(pc.data_parallel_size)}]
        return list(self.executor.collective_rpc("appcorr_worker_info"))


class LLMRanks(OpDispatchMixin):
    """`run_on_ranks` around a PLAIN `vllm.LLM` (no `StreamingLLM`, no streaming request).

    The two correctness gates that predate TP -- `glm53_kda_gate.py` (K) and `glm53_mla_gate.py`
    (M) -- reach the runner as `…engine_core.model_executor.driver_worker.worker.model_runner`,
    which exists only under `UniProcExecutor`. GLM-5.3-Flash does not fit on one B200, so that
    path is unreachable for this model and the gates need the same TP dispatch `StreamingLLM`
    has. This adapter gives it to them without either gate taking a dependency on the streaming
    client.

    Usage: `ranks = LLMRanks(llm, tp); rows = ranks.run_on_ranks(my_body, *args)` -> one entry
    per rank. State a body wants to keep BETWEEN two calls (a temporary monkeypatch armed before
    a `generate` and harvested after) must be stashed on the `runner` object: the body itself is
    cloudpickled by value each time, so two calls do not share a module namespace in the worker.
    """

    def __init__(self, llm, tensor_parallel_size: int = 1):
        self.tensor_parallel_size = int(tensor_parallel_size)
        core = llm.llm_engine.engine_core
        self._core = getattr(core, "engine_core", core)

    @property
    def executor(self):
        return self._core.model_executor

    @property
    def runner(self):
        if self.tensor_parallel_size > 1:
            raise RuntimeError("no in-process runner at TP>1; use run_on_ranks")
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        r = self._core.model_executor.driver_worker.worker.model_runner
        assert isinstance(r, GPUModelRunner), type(r)
        return r


def agree_across_ranks(rows: list, keys, *, tol: float = 0.0) -> dict:
    """Check that every rank reported the same value for `keys`, and say where it did not.

    Use ONLY for quantities vLLM replicates across TP ranks -- the MLA latent cache (one KV head)
    and the sparse indexer's `k_cache` / `tail_cache`
    (`vllm/models/glm5next/nvidia/attention.py:250-266`). Do NOT use it on KDA conv/recurrent
    state or on any `o_proj`-sharded quantity: those are sharded by head
    (`vllm/model_executor/layers/mamba/abstract.py:59-61`) and the ranks are SUPPOSED to differ,
    so an equality assert there would be a false alarm that hides the real check.
    """
    import math

    out = {"n_ranks": len(rows), "agree": True, "disagreements": {}}
    if len(rows) < 2:
        return out
    for k in keys:
        vals = [r.get(k) for r in rows]
        base = vals[0]
        for i, v in enumerate(vals[1:], start=1):
            same = (v == base) if not isinstance(base, float) else (
                math.isclose(float(v), float(base), rel_tol=tol, abs_tol=tol))
            if not same:
                out["agree"] = False
                out["disagreements"].setdefault(k, []).append({"rank": i, "value": v,
                                                               "rank0": base})
    return out
