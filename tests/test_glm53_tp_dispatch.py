"""CPU gate for the TP>1 op dispatch (`appcorr/vllm_stream/tp_worker.py` + `StreamingLLM`'s
`_dispatch` / `_dispatch_worker` / `run_on_ranks`).

No CUDA, no engine, no model. A fake two-rank executor stands in for `MultiProcExecutor`: it
implements the same `collective_rpc(method, args=, kwargs=)` contract (`run_method` semantics,
`vllm/v1/serial_utils.py:486-510`) over two fake workers, each with its own fake
`GPUModelRunner`, and returns the per-rank result list. What that lets us assert without a GPU:

  1. **every rank is called, with identical args** -- the property that makes a replicated
     rewrite (the MLA latent, the sparse indexer's caches) identical across ranks by
     construction rather than by check. Asserted bytewise on the tensors each rank received,
     including that rank 1 did NOT receive an alias of rank 0's tensor;
  2. **arguments arrive as CPU tensors** -- the broadcast queue's torch reducer handles CPU
     tensors only (`shm_broadcast.py:421-449`), so `to_cpu` must have run on the way out and the
     worker must have moved them onto its own device on the way in;
  3. **side buffers are per-rank objects** -- `appcorr_open_buffer` builds one on each rank's own
     device, and the two are distinct objects, which is what makes a KDA re-scan rank-local
     (the conv/recurrent state is sharded by head, `mamba/abstract.py:59-61`);
  4. **only `appcorr_*` runner methods are reachable** through the RPC name channel;
  5. **`run_on_ranks` returns one entry per rank** and ships a `__main__`-defined callable;
  6. **the TP=1 path is unchanged** -- `_dispatch` calls the in-process runner directly, with the
     caller's own tensor objects (identity, not equality) and no executor involvement at all.

The end-to-end TP=1-is-unchanged claim is the bitwise `vllm_interleaved_axis_gate.py --tiny`
runs for `glm53` and `glm46v`; this file is the dispatch half of it.
"""
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from appcorr.vllm_stream.tp_worker import (  # noqa: E402
    AppcorrWorkerExtension, OpDispatchMixin, runner_call, to_cpu)


# --- the fakes --------------------------------------------------------------------------------- #

class _FakeRunner:
    """Just enough `GPUModelRunner` for the dispatch layer: a device, a request table, and the
    `appcorr_*` methods recording what they were handed."""

    def __init__(self, device: str):
        self.device = torch.device(device)
        self.requests = {}
        self.calls = []
        self.vllm_config = types.SimpleNamespace(
            compilation_config=types.SimpleNamespace(static_forward_context={}))

    def appcorr_correct_step(self, req_id, positions, embeds, window, final, replay=False):
        self.calls.append(("appcorr_correct_step", req_id, positions, embeds, window, final, replay))
        return {"num_rows": int(positions.numel()), "device": str(positions.device),
                "embeds_device": str(embeds.device)}

    def appcorr_open_walk(self, req_id):
        self.calls.append(("appcorr_open_walk", req_id))
        return {"req_id": req_id}

    def appcorr_take_final_info(self, req_id):
        self.calls.append(("appcorr_take_final_info", req_id))
        return {"req_id": req_id}

    def load_model(self):            # a NON-appcorr method: must stay unreachable over RPC
        self.calls.append(("load_model",))
        return "loaded"


class _FakeWorker(AppcorrWorkerExtension):
    """A vLLM `Worker` as far as the extension is concerned: `.model_runner`, `.rank`."""

    def __init__(self, rank: int, device: str):
        self.rank = rank
        self.local_rank = rank
        self.model_runner = _FakeRunner(device)


class _FakeExecutor:
    """`MultiProcExecutor.collective_rpc`'s contract, in-process, over N fake workers.

    Mirrors `run_method`: a `str` is `getattr`'d on the worker, a callable is `partial(fn,
    worker)`. Each rank is handed its OWN deserialised copy of the arguments -- the real thing
    pickles them through a shared-memory buffer, so a rank can never mutate another's tensor, and
    a test that passed only because the ranks shared one object would be testing nothing."""

    def __init__(self, world_size: int = 2):
        self.workers = [_FakeWorker(r, "cpu") for r in range(world_size)]
        self.rpcs = []

    @staticmethod
    def _roundtrip(obj):
        import pickle
        return pickle.loads(pickle.dumps(obj))

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False):
        kwargs = kwargs or {}
        self.rpcs.append((method if isinstance(method, str) else "<callable>", args, kwargs))
        out = []
        for w in self.workers:
            a, k = self._roundtrip((args, kwargs))
            fn = getattr(w, method) if isinstance(method, str) else (lambda *x, **y: method(w, *x, **y))
            out.append(fn(*a, **k))
        return out


class _FakeLLM(OpDispatchMixin):
    """`StreamingLLM`'s dispatch surface without the engine. The mixin under test is the REAL
    one `StreamingLLM` inherits -- only `executor` / `runner` are stubbed."""

    def __init__(self, tp: int):
        self.tensor_parallel_size = tp
        self._executor = _FakeExecutor(tp) if tp > 1 else None
        self._runner = _FakeRunner("cpu")

    @property
    def executor(self):
        assert self._executor is not None, "TP=1 must not touch the executor"
        return self._executor

    @property
    def runner(self):
        assert self.tensor_parallel_size == 1, "TP>1 must not touch the in-process runner"
        return self._runner


# --- 1 / 2: every rank, identical args, CPU on the wire ------------------------------------------ #

def test_dispatch_calls_every_rank_with_identical_args():
    llm = _FakeLLM(tp=2)
    pos = torch.tensor([3, 7, 11], dtype=torch.int64)
    emb = torch.randn(3, 8)
    info = llm._dispatch("appcorr_correct_step", "req-a", pos, emb, (0, 16), False, replay=False)

    assert info["num_rows"] == 3                      # rank 0's reply is what the caller gets
    assert len(llm.executor.rpcs) == 1
    assert llm.executor.rpcs[0][0] == "appcorr_rpc"

    per_rank = [w.model_runner.calls for w in llm.executor.workers]
    assert [len(c) for c in per_rank] == [1, 1], [len(c) for c in per_rank]
    a0, a1 = per_rank[0][0], per_rank[1][0]
    assert a0[0] == a1[0] == "appcorr_correct_step"
    assert a0[1] == a1[1] == "req-a"
    assert torch.equal(a0[2], a1[2]) and torch.equal(a0[3], a1[3])   # identical VALUES
    assert a0[2].data_ptr() != a1[2].data_ptr()                      # but not the same OBJECT
    assert a0[4] == a1[4] == (0, 16) and a0[5] is False and a0[6] is False
    # what crossed the wire was CPU, and each rank put it on its own device (both "cpu" here)
    assert info["device"] == "cpu" and info["embeds_device"] == "cpu"
    assert torch.equal(a0[2], pos) and torch.allclose(a0[3], emb)


def test_to_cpu_moves_tensors_and_leaves_the_rest():
    d = {"t": torch.zeros(2), "n": 3, "s": "x", "l": [torch.ones(1), None], "tup": (torch.ones(2),)}
    out = to_cpu(d)
    assert out["t"].device.type == "cpu" and out["n"] == 3 and out["s"] == "x"
    assert out["l"][1] is None and isinstance(out["tup"], tuple)


# --- 3: side buffers are per-rank ---------------------------------------------------------------- #

def test_open_buffer_is_one_object_per_rank(monkeypatch):
    """`appcorr_open_buffer` must build the buffer on the RANK's device out of the RANK's own
    `correct` module state. Patched here because the real `correct.py` needs vllm; what is under
    test is the extension's plumbing (device source, per-rank call), not `SideBuffer` itself."""
    import appcorr.vllm_stream.tp_worker as tw

    made = []

    class _SB:
        def __init__(self, req_id, n, device):
            self.n, self.device, self.req_id = n, device, req_id

    fake_correct = types.SimpleNamespace(
        check_gdn_path=lambda cfg: [],
        open_buffer=lambda req_id, n, device, **kw: (made.append(_SB(req_id, n, device)) or made[-1]),
        get_buffer=lambda req_id: None,
        free=lambda req_id: None)
    # BOTH, and this is the whole trick: `from . import correct` inside the extension resolves
    # the PACKAGE ATTRIBUTE first and only falls back to `sys.modules`, so patching `sys.modules`
    # alone works in isolation and silently does nothing once another test in the same session
    # has already imported the real module (which is how this failed only in the full suite).
    import appcorr.vllm_stream as _pkg
    monkeypatch.setitem(sys.modules, "appcorr.vllm_stream.correct", fake_correct)
    monkeypatch.setattr(_pkg, "correct", fake_correct, raising=False)
    monkeypatch.setattr(tw, "_install", lambda: None, raising=False)

    ex = _FakeExecutor(2)
    ex.workers[1].model_runner.device = torch.device("cpu")   # distinct object, same type
    outs = [w.appcorr_open_buffer("req-b", 512, lo=4, hi=260) for w in ex.workers]
    assert [o["n"] for o in outs] == [512, 512]
    assert len(made) == 2, made
    assert made[0] is not made[1], "the two ranks must not share one side buffer"
    assert made[0].device is ex.workers[0].model_runner.device
    assert made[1].device is ex.workers[1].model_runner.device


# --- 4: the RPC name channel is not a generic method channel -------------------------------------- #

def test_appcorr_rpc_refuses_a_non_appcorr_method():
    w = _FakeWorker(0, "cpu")
    with pytest.raises(ValueError, match="only appcorr_"):
        w.appcorr_rpc("load_model")
    assert w.model_runner.calls == []
    with pytest.raises(AttributeError, match="install"):
        w.appcorr_rpc("appcorr_does_not_exist")


# --- 5: run_on_ranks ships a callable -------------------------------------------------------------- #

def _gate_body(runner, scale: float):
    """Stands in for K's / M's per-rank comparison: sees the rank's runner, returns CPU data."""
    return {"device": str(runner.device), "scale": scale,
            "probe": (torch.arange(4, dtype=torch.float32) * scale)}


def test_run_on_ranks_returns_one_result_per_rank():
    llm = _FakeLLM(tp=2)
    res = llm.run_on_ranks(_gate_body, 2.0)
    assert len(res) == 2, res
    assert all(r["scale"] == 2.0 for r in res)
    assert torch.equal(res[0]["probe"], res[1]["probe"])          # replicated -> ranks agree
    assert llm.executor.rpcs[0][0] == "<callable>"
    # the transport adapter itself, called the way collective_rpc calls it
    assert runner_call(llm.executor.workers[0], _gate_body, (3.0,))["scale"] == 3.0


def test_run_on_ranks_at_tp1_is_the_in_process_runner():
    llm = _FakeLLM(tp=1)
    res = llm.run_on_ranks(_gate_body, 1.0)
    assert len(res) == 1 and res[0]["device"] == "cpu"


# --- 6: TP=1 is untouched -------------------------------------------------------------------------- #

def test_tp1_dispatch_is_a_direct_call_with_the_callers_own_tensors():
    llm = _FakeLLM(tp=1)
    pos = torch.tensor([1, 2], dtype=torch.int64)
    emb = torch.randn(2, 4)
    info = llm._dispatch("appcorr_correct_step", "req-c", pos, emb, (0, 4), True, replay=False)
    assert info["num_rows"] == 2
    call = llm._runner.calls[0]
    # IDENTITY, not equality: TP=1 must not copy, must not go to the CPU, must not serialise
    assert call[2] is pos and call[3] is emb
    assert llm._executor is None


def test_tp1_dispatch_worker_uses_the_inproc_branch():
    llm = _FakeLLM(tp=1)
    seen = []
    out = llm._dispatch_worker("appcorr_scatter_prompt", "req-d", torch.tensor([0]),
                               torch.zeros(1, 2), inproc=lambda: (seen.append(1) or 42))
    assert out == 42 and seen == [1] and llm._executor is None


def test_worker_info_shape_at_tp1():
    w = _FakeWorker(1, "cpu")
    info = _FakeExecutor(2).workers[1]
    assert isinstance(info, _FakeWorker) and w.rank == 1
