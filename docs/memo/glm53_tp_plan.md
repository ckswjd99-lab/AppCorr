# TP=2 for the AppCorr stream server -- plan (agent V, 2026-09-13)

Scope: what has to change in `appcorr/vllm_stream/` for our streaming-prefill server to run a
model under `tensor_parallel_size=2`, and the smallest experiment that proves the first two
steps. Written for GLM-5.3-Flash on B200-8 GPUs 2-3 (the box already runs a stock TP=2 arm there:
world size 2, FLASHINFER all-reduce for TP, PYNCCL for EP), but nothing below is GLM-specific --
it is the same for the 122B Qwen row if that ever needs two GPUs.

Every path below is into the vLLM main clone at HEAD **658c813**
(`$V = /tmp/claude-3092/.../scratchpad/vllm-main`), which is the nightly the served env
(`/NHNHOME/share/cjpark/backup/env/appcorr-vllm-main`, vllm `0.1.1.dev65+g658c8131c`) was built
from. **Nothing here has been run.**

---

## 0. What actually changes at TP>1

`VLLM_ENABLE_V1_MULTIPROCESSING=0` and `tensor_parallel_size` are **orthogonal knobs on
different processes**, which is the whole source of confusion:

| knob | what it controls | where it is read |
|---|---|---|
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | the **engine core** runs in the caller's process (`InprocClient`) instead of its own | `$V/vllm/v1/engine/llm_engine.py:162,179` |
| `tensor_parallel_size > 1` | the **executor** becomes `MultiProcExecutor`, one worker PROCESS per rank | `$V/vllm/config/parallel.py:966-1007` (world_size > 1 and no ray -> backend `"mp"`), `:1010-1011` (world_size == 1 -> `"uni"`) |

So at TP=2 we keep everything we rely on in-process -- `self.core.scheduler.stream_append` by
reference, `StreamingScheduler`, `EngineCore.preprocess_add_request` -- and lose exactly one
thing: `self.core.model_executor.driver_worker.worker.model_runner`
(`client.py:203-211`) no longer exists, because the runner lives in the worker processes.
`ExecutorWithExternalLauncher` is the one backend that keeps a single in-process worker per
engine, but it requires a torchrun-style launcher and N copies of the engine
(`$V/vllm/v1/executor/uniproc_executor.py:168-200`); it is not a shortcut for us.

`async_scheduling` is a **scheduler-side** flag (`$V/vllm/config/scheduler.py:190,214`) and is
unaffected by TP: it selects `AsyncScheduler` + a 2-deep batch queue with sampled ids kept on the
GPU. Our `StreamingScheduler` subclasses the **sync** `Scheduler`
(`appcorr/vllm_stream/scheduler.py:55`) and the hooks were gated on the sync engine, so
`client.py` keeps forcing `async_scheduling=False` at any TP. Under TP that has one extra
consequence worth knowing: a sync step is one `collective_rpc("execute_model", ...)` broadcast
plus a reply from the output rank (`$V/vllm/v1/executor/multiproc_executor.py:340-360`), so the
scheduler's step latency now includes an inter-process round trip -- it does NOT change any
value, only the wall clock the latency probe reads. Turning async scheduling on to hide that is a
separate, unported piece of work.

---

## 1. (a) Getting the runner patch into each worker

`runner_patch.install()` and `correct.install()` monkey-patch the `GPUModelRunner` **class**
(`appcorr/vllm_stream/runner_patch.py:88-101`). At TP=1 that is done by `StreamingLLM.__init__`
before `LLM(...)`. At TP=2 the class has to be patched **inside each worker process**, before its
first `execute_model`. Three mechanisms exist; take the third.

1. **Rely on `fork`.** `VLLM_WORKER_MULTIPROC_METHOD` defaults to `"fork"`
   (`$V/vllm/envs.py:68`), and a forked child inherits the parent's already-patched class object
   for free. **Do not rely on it**: `_maybe_force_spawn` (`$V/vllm/utils/system_utils.py:125-164`,
   called from `get_mp_context()` at `:168-181`, used by `MultiProcExecutor._init_executor` at
   `$V/vllm/v1/executor/multiproc_executor.py:172`) silently overrides to `spawn` when **CUDA is
   already initialized** in the parent -- which it usually is by the time the executor is built
   -- and also under Ray, WSL and `--numa-bind`. A gate that passes under fork and fails in
   production under spawn is the worst outcome available.
2. **A `vllm.general_plugins` entry point.** Each worker calls `load_general_plugins()` in
   `WorkerWrapperBase.init_worker` (`$V/vllm/v1/worker/worker_base.py:269-271`), and
   `appcorr/vllm_stream/plugin.py` already exists for exactly this. It needs the package
   INSTALLED to register the entry point, and `bin/pip` is broken in the served env, so this is
   correct but not available tonight.
3. **`--worker-extension-cls` (take this one).** `ParallelConfig.worker_extension_cls`
   (`$V/vllm/config/parallel.py:276-280`) is resolved by qualified name and dynamically mixed
   into the worker class in `init_worker`
   (`$V/vllm/v1/worker/worker_base.py:285-310`) -- i.e. **our module is imported in every worker
   process**, before `init_device`/`load_model`, with no packaging. Put

   ```python
   # appcorr/vllm_stream/tp_worker.py
   from . import install
   install()                      # patches GPUModelRunner in THIS process, idempotent

   class AppcorrWorkerExtension:
       """Methods callable via executor.collective_rpc(name, args=...)."""
       def appcorr_open_buffer(self, ...): ...
       def appcorr_correct(self, ...): ...
   ```

   and pass `worker_extension_cls="appcorr.vllm_stream.tp_worker.AppcorrWorkerExtension"` through
   `LLM(...)` (it is a real `EngineArgs` field, `$V/vllm/engine/arg_utils.py:702, 2370`, CLI
   `--worker-extension-cls` at `:1245`). Two constraints from `init_worker`: every public
   attribute name must not already exist on `Worker` (it asserts, `:293-298`), and the class must
   be importable by qualified name in the worker -- so `PYTHONPATH` must carry the repo, which
   the server command already sets.

   Note `install()` refuses any vllm outside `SUPPORTED_VLLM = ("0.11.2", "0.28.0")`
   (`appcorr/vllm_stream/__init__.py:22,30-35`) and the served env reports
   `0.1.1.dev65+g658c8131c`. **This blocks TP=2 and TP=1 alike in that env** and must be settled
   before any gate there -- either add the nightly's version string to `SUPPORTED_VLLM` or give
   `install()` an explicit override. It is a one-line decision but it is not mine to make
   silently, so the gate script (§5) checks it first and stops with this message.

## 2. (b) Dispatching `open` / `correct` / `append` to all ranks

`Executor.collective_rpc(method, args=, kwargs=)` takes **either a method name or a callable**;
a callable is `cloudpickle`d and each worker runs `func(self.worker)`
(`$V/vllm/v1/executor/multiproc_executor.py:375-420` enqueues `(send_method, args, kwargs,
output_rank)` on `rpc_broadcast_mq`; `$V/vllm/v1/executor/multiproc_executor.py:1029-1059`
`worker_busy_loop` -> `_execute_worker_rpc` resolves it with `getattr(self.worker, method)` or
`partial(cloudpickle.loads(method), self.worker)` and returns the result). The executor is
reachable from our client as `self.core.model_executor` (`$V/vllm/v1/engine/core.py:134`), and
`UniProcExecutor` implements the same method -- so ONE code path serves TP=1 and TP=2.

Per op:

* **`append` needs nothing.** It goes through `EngineCore.add_request` ->
  `StreamingScheduler.stream_append` (`engine_patch.py:44-51`), all inside the engine-core
  process. The grown prompt reaches the workers through the SchedulerOutput -- see (c).
* **`open(correct=True)`** currently calls `correct.open_buffer(..., self.runner.device, ...)`
  in the driver process (`client.py:178-185`). Becomes
  `executor.collective_rpc("appcorr_open_buffer", args=(core_id, n, lo, hi, ...))`. Identical
  args on every rank; the device is each worker's own.
* **`correct`** currently calls `self.runner.appcorr_correct_step(...)` /
  `appcorr_staged_correct(...)` / `appcorr_arm_final(...)` (`client.py:248-265`). Becomes the
  same three as `collective_rpc` names. **Arguments must be CPU tensors**: the RPC is pickled
  into the shared-memory queue with a torch dispatch entry that handles CPU tensors out-of-band
  (`$V/vllm/distributed/device_communicators/shm_broadcast.py:421-449, 823-855`); a CUDA tensor
  would fall through to torch's default reducer and is not something to rely on across
  processes. So `correct(positions, embeds, ...)` has to `.cpu()` on the way out and each rank
  re-uploads -- a real added cost on the correction critical path (a band's embeds at 4096 dim
  bf16 is ~8 KB/row) that the latency probe must be told about, not a correctness issue.
* **Return values** come back as a list, one per rank, unless `unique_reply_rank` is passed
  (`multiproc_executor.py:382, 405-420`). The correct step's `info` dict is picklable; take
  rank 0's and assert the others agree on `num_rows` / `n_sub` (cheap, and it is the only
  mechanical check that the ranks did the same work).
* **Per-rank side buffers.** `correct.py`'s buffers are keyed by request id in module state
  inside the worker process, so at TP they are automatically per-rank and rank-local. That is
  the right shape: the KDA conv/recurrent state is **sharded by head**
  (`is_kv_cache_tp_replicated=False`, `$V/vllm/model_executor/layers/mamba/abstract.py:59-61`;
  shapes at `$V/vllm/model_executor/layers/mamba/mamba_utils.py:298-321`), so a re-scan is
  rank-local and needs no communication; the `o_proj` all-reduce after it is the stock layer's
  (`$V/vllm/models/glm5next/nvidia/kda.py:256`). The MLA latent and the whole sparse indexer are
  **replicated** (`$V/vllm/models/glm5next/nvidia/attention.py:250-266`), so every rank must
  perform the identical rewrite -- which a broadcast `collective_rpc` with identical args gives
  by construction, as long as nothing in the op reads a rank-local random source.

## 3. (c) Keeping the pushed embeddings and positions consistent per rank

**This needs no change, and the reason is worth writing down because the scheduler's own
docstring guesses otherwise** (`appcorr/vllm_stream/scheduler.py:19-20`: "a multi-process core
would need them as real (msgpack) fields").

`SchedulerOutput` and `NewRequestData` are plain `@dataclass`es
(`$V/vllm/v1/core/sched/output.py:36-37, 224-225`), and the broadcast queue serialises them with
**pickle protocol 5**, not msgspec: `MessageQueue.enqueue` builds a `pickle.Pickler` with
`dispatch_table[torch.Tensor] = _reduce_tensor` and a `buffer_callback`, so CPU tensor bytes ride
as out-of-band buffers (`$V/vllm/distributed/device_communicators/shm_broadcast.py:823-855`,
reducer at `:421-449`, rebuild at `:389-418`). Pickling a plain dataclass pickles its `__dict__`,
so the attributes `StreamingScheduler.schedule` attaches -- `nr.appcorr_stream`
(`scheduler.py:78`) and `out.appcorr_stream_updates` (`scheduler.py:87`) -- **do** cross to the
workers, together with the `StreamChunk`'s CPU `embeds`. Requirements that follow:

1. `StreamNewInfo` and `StreamChunk` must be importable by qualified name in the worker. They
   are (`appcorr.vllm_stream.scheduler` / `.request`), given `PYTHONPATH`.
2. Chunk tensors must be on the **CPU** at that point. They are: `client.py` keeps
   `req.prompt_embeds` / `st.prompt_embeds` on the CPU and the runner patch does `cpu_cat`
   (`runner_patch.py:42-48`).
3. One enqueue is read by every local reader out of the same shared-memory buffer, so the ranks
   see byte-identical content -- consistency is structural, not something to verify per rank.
4. **Cost, not correctness**: at TP=1 in-process the SchedulerOutput was passed by reference and
   the prompt embeds cost nothing per step. Now the first step after each append copies the
   chunk through the queue (a 3000-row 4096-dim bf16 prompt is ~24 MB; buffers over 1 MiB go out
   of band, and over `VLLM_MQ_MAX_CHUNK_BYTES_MB` through a socket, `:848-856`). Expect a
   per-push cost the TP=1 numbers do not have, and read `t_open`/`t_final` accordingly.
5. **Positions.** For GLM-5.3-Flash the chunks carry `mrope_positions=None`
   (`appcorr/models/glm53/axis.py`, `client.py::Glm53Composer`), so nothing position-shaped
   crosses at all and each rank derives the same 1-D counter from `num_computed_tokens`. For an
   M-RoPE model the `[3, T]` CPU tensor crosses the same way as the embeds.

## 4. (d) Sequence parallelism

**It is already off at TP=2 and there is nothing to disable.** `use_sequence_parallel_moe` is a
property that requires `data_parallel_size > 1` in addition to expert parallelism and TP>1
(`$V/vllm/config/parallel.py:714-730`): the full condition is an all2all backend in
{`allgather_reducescatter`, `deepep_*`, `flashinfer_nvlink_one_sided`, `mori_*`, `nixl_ep`}
**and** `enable_expert_parallel` **and** `tensor_parallel_size > 1` **and**
`data_parallel_size > 1`. The B200-8 configuration is TP=2, DP=1, EP on -> `False`.

Why it matters that it stays off: when SP is on, GLM-5.3's decoder layer chunks the token axis
across ranks and inserts `sp_all_gather` / `sp_reduce_scatter` **inside** the layer around
attention (`$V/vllm/models/glm5next/nvidia/model.py:366-369, 473-474`, MoE at `:253-269`), so a
per-rank row rewrite would have to be aligned to the SP shard and a rank-local KDA re-scan would
straddle a collective. So: **assert it rather than assume it.** The check is one line in the
worker extension --
`assert not vllm_config.parallel_config.use_sequence_parallel_moe` -- and it belongs in step (a)'s
module, because the day someone adds `--data-parallel-size 2` it is the only thing standing
between us and silently wrong corrected rows.

## 5. The smallest experiment that proves (a) + (b)

`analysis/experiments/glm53_tp_gate.sh` -- **arm B of `vllm_stream_gate.py` under TP=2 with no
correction**: the same prompt composed by `Glm53Composer` and pushed as `prompt_embeds` in one
message, against arm A (vLLM's own image request) on the same engine. It exercises exactly the
two things the plan changes and nothing else:

* (a) is proved by the run **starting at all**: `runner_patch._update_states` /
  `_init_mrope_positions` must be installed in both workers or the streaming request's
  `NewRequestData` handling diverges from stock; and `_preprocess`'s absence (0.28+) is the same
  either way.
* (b) is proved by the **result**: arm B exact-matching arm A 8/8 means both ranks consumed the
  same pushed rows through the same code path. Since B needs no `correct`, it needs no
  `collective_rpc` of our own -- so a PASS isolates the executor/serialisation half from the
  correction half, which is the point of gating it first.

Arm C (chunked streaming, still no correction) is the natural follow-up in the same script and
is included as an optional second invocation; it adds the append path of §3 to the same check.

Pass rule, the same as every prior `vllm_stream_gate` run: **exact 8/8** on the generated token
ids, and `max_dlogprob_common` reported. A non-exact B under TP=2 that is exact under TP=1 is an
all-reduce non-determinism question (FLASHINFER all-reduce is the TP backend on that box), not
an AppCorr bug -- run TP=1 first in the same script for that reason.

## 6. Order of work after the gate

1. `appcorr/vllm_stream/tp_worker.py` (§1 mechanism 3) + `--worker-extension-cls` plumbed from
   `StreamingLLM.__init__`. The guard that already exists: `StreamingLLM.runner` raises at
   TP>1 with a pointer here rather than walking a `driver_worker` attribute that is not there.
2. `StreamingLLM._dispatch(op, *args)` -> `collective_rpc`, used by `open`/`correct` at ANY TP
   (at TP=1 `UniProcExecutor.collective_rpc` runs it in-process, so there is one path to test).
   This is the only change `correct.py` / `runner_patch.py` need, and it is additive: the
   `appcorr_*` methods they already expose on the runner become the worker extension's targets.
3. Only then the correction arms. `--interleaved` + TP>1 is refused today
   (`appcorr/vllm_stream/server.py::main`).

## 6b. STATUS: steps (a) and (b) are IMPLEMENTED (2026-09-13, agent V)

Written, CPU-gated, not yet run on a GPU.

* `appcorr/vllm_stream/tp_worker.py` -- `AppcorrWorkerExtension` (the `--worker-extension-cls`
  target; importing it runs `install()` in the worker), `OpDispatchMixin` (`_dispatch` /
  `_dispatch_worker` / `run_on_ranks` / `worker_info`, mixed into `StreamingLLM`), `LLMRanks`
  (the same dispatch around a plain `vllm.LLM`, for the two correctness gates), `to_cpu`,
  `runner_call`, `agree_across_ranks`.
* `client.py` passes `worker_extension_cls` at TP>1 ONLY, and every op that used to call
  `self.runner.appcorr_*` now goes through `_dispatch` / `_dispatch_worker`:
  `appcorr_open_buffer`, `appcorr_open_walk`, `appcorr_arm_final`, `appcorr_correct_step`,
  `appcorr_staged_correct`, `appcorr_take_final_info`, plus the worker-side CPU prompt mirror
  (`appcorr_scatter_prompt`) and the per-rank side-buffer free on finish (`appcorr_free`).
* `server.py` no longer refuses `--interleaved` with TP>1.
* Two bugs found on the way, both invisible at TP=1 and fatal at TP>1:
  1. `StreamingScheduler._free_request` calls `correct.free` in the ENGINE-CORE process; at TP>1
     the side buffers are in the workers, so every corrected request leaked its buffer
     (1.06 MiB/token/rank). `StreamingLLM.step` now dispatches `appcorr_free` on finish.
  2. The hold-back-release chunk was built with `mrope_positions=torch.empty((3, 0))`, which is
     "not None", so `runner_patch`'s `assert st.mrope_positions is not None` would have fired on
     the LAST correct round of every GLM-5.3 request -- a non-M-RoPE model whose
     `CachedRequestState.mrope_positions` is never filled. It is `None` now when the request has
     no M-RoPE.

TP=1 is untouched by construction: `_dispatch` takes the in-process branch, with the caller's own
GPU tensor OBJECTS (asserted by identity in the dispatch test), no CPU copy and no executor call,
and `worker_extension_cls` is not set. Evidence in the appendix.

What is still GPU-only: everything the legs of `glm53_tp_gate.sh` measure. In particular nobody
has yet observed a worker process import `tp_worker`, so (a) is argued from
`worker_base.py:285-310` and checked by `worker_info()`, not observed.

## 7. What I changed already (this session)

* `appcorr/vllm_stream/client.py`: `tensor_parallel_size` is a real parameter (it was hardcoded
  `1` in the `LLM(...)` call); `StreamingLLM.runner` raises at TP>1 instead of walking a
  `driver_worker` attribute that does not exist there; `EmbeddedPrompt` accepts
  `mrope_positions=None`.
* `appcorr/vllm_stream/server.py`: `--tensor-parallel-size`, and `--interleaved` + TP>1 refused.
* `analysis/experiments/vllm_stream_gate.py`: `--tensor-parallel-size`, and `composer_for` now
  raises on an unknown model instead of silently returning the Qwen composer.

No change to `correct.py` or `runner_patch.py`.

---

## Appendix: CPU gate results behind this plan (2026-09-13, agent V)

The vision/axis half is gated; the TP half is NOT (nothing in §5 has been run).

| gate | how | result |
|---|---|---|
| stock port vs transformers `Glm5NextVisionModel` | `glm53_tower_gate.py --mode port`, served env (tf 5.16.1), fp32 CPU, real checkpoint weights | **rel-L2 0.0, max_abs 0.0** -- bitwise. `reference_forward` is bitwise the port too. Both towers at norm 1e-6 / qk_norm 1e-5 |
| fork `reference_forward` vs stock tower | `tests/test_glm53_tower.py` | bitwise, max_abs 0 |
| 24 block stages (4 ranges) vs reference | same | bitwise, max_abs 0 |
| merger band slicing, bf16 | same, g1c fp32-referenced criterion | 0 non-bitwise bands of 17; err_band max 9.565e-4 == err_all max, ulp@scale 9.766e-4 |
| `correct_rows` vs `correct_forward[rows]` | same, 1 thread | bitwise, max_abs 0 |
| g=1 correction == full-resolution forward | same | rel-L2 0.0 |
| q/k RMSNorm is wired into every path | same, perturbation | output moves by 0.0339 |
| GLM-4.6V G5 regression after the `uses_mrope` edit | `vllm_interleaved_axis_gate.py --tiny --family glm46v` | G5_PASS, bitwise at keep 1.0 and 0.5 |
| prompt half (18 checks: template shape, the `</think>` edit on both sides, sentinels, expansion, positions, low-res, driver/gate plumbing) | `glm53_tower_gate.py --mode prompt`, served env | all true, `fails: []`. 414 -> 63 image tokens at `max_image_tokens=64` (grid 36x46 -> 14x18) |
| GLM-5.3 streaming/interleaved axis, tiny random model | `vllm_interleaved_axis_gate.py --tiny --family glm53`, served env | G5_PASS, bitwise (0/32 bad rows) at keep 1.0 and 0.5; prefill streaming 32 / interleaved 59 and 47 |

**Practical note for whoever runs the TP gate**: cap the CPU thread pool. The tower's per-layer
GEMMs are small and at this box's default 72 OMP threads a full CPU pass was still running at 15
minutes; at `OMP_NUM_THREADS=8` the same four tests took 86 s. Same lesson as the serving hot
path (`reference/cpu_torch_ops_omp_72_threads`).

---

## Appendix B: TP dispatch, CPU evidence (2026-09-13, agent V)

| gate | how | result |
|---|---|---|
| dispatch layer, fake two-rank executor | `tests/test_glm53_tp_dispatch.py` (appcorr env, no vllm, no CUDA) | 9/9. Every rank called; args byte-equal but distinct objects; args arrive as CPU tensors; side buffers are distinct per-rank objects; `appcorr_rpc` refuses a non-`appcorr_` method; `run_on_ranks` returns one entry per rank; TP=1 passes the caller's own tensor OBJECTS (identity) and never touches the executor |
| TP=1 unchanged, GLM-5.3 | `vllm_interleaved_axis_gate.py --tiny --family glm53` after the refactor | G5_PASS, bitwise (0/32 bad rows) at keep 1.0 and 0.5 |
| TP=1 unchanged, GLM-4.6V | same, `--family glm46v` | G5_PASS, bitwise (0/32 bad rows) at keep 1.0 and 0.5 |
| whole suite after the refactor | `pytest tests/ -q` (appcorr env) | **144 passed, 8 skipped, 0 failed**, 11 subtests passed, 162 s (K's and M's 135+8 plus the 9 new dispatch tests) |
| served-env preflight | `glm53_tp_gate.sh 0`, CPU only, vllm `0.1.1.dev65+g658c8131c` | importing `tp_worker` gives `INSTALLED=True / stream=True / correct=True`; **zero** attribute clashes with vLLM's `Worker`; `resolve_obj_by_qualname` (the exact call `init_worker` makes) resolves the class; all 6 dispatched runner methods present; all 7 rank-side gate bodies cloudpicklable |

GPU-only, i.e. everything `glm53_tp_gate.sh` legs 1-3 measure: that a worker process actually
imports `tp_worker` and comes up patched (`worker_info()` reports it, but nobody has watched it
happen); that arm C survives the append path across the process boundary; that the KDA re-scan
matches the stock kernel per rank; that the replicated MLA/indexer snapshots agree across ranks.
