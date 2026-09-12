"""In-process client: `StreamingLLM` (open / append / step) and Qwen2-VL-family prompt composition.

The prototype runs the engine core in the caller's process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`,
TP=1) so the scheduler/runner hooks can pass tensors by reference. A chunk is submitted as an
ordinary `EngineCoreRequest` built by vLLM's own `Processor` (so max_tokens/eos/stop handling is
stock) with the `x-appcorr-stream` headers of `request.py`; `open` also registers the request
with the `OutputProcessor` (`LLMEngine.add_request`), `append`/`final` go straight to the engine
core (the output processor already knows the id). `step()` is one engine step; the caller
interleaves `append` and `step` however the chunks arrive.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import torch

from .request import StreamChunk, make_stream_headers


@dataclass
class PromptParts:
    """A Qwen2-VL-family prompt as vLLM would tokenize it, split into what the embeds are made of."""
    input_ids: torch.Tensor          # [N] int64 (image_pad tokens included)
    image_start: int                 # first image_pad row
    image_len: int                   # number of image_pad rows (= merged vision tokens)
    pixel_values: torch.Tensor       # HF processor output [num_patches, C*T*P*P]
    image_grid_thw: torch.Tensor     # [1, 3]
    text: str

    @property
    def num_tokens(self) -> int:
        return int(self.input_ids.shape[0])


@dataclass
class EmbeddedPrompt:
    embeds: torch.Tensor             # [N, D] CPU, model dtype
    mrope_positions: torch.Tensor    # [3, N] int64
    mrope_delta: int
    parts: PromptParts

    def chunk(self, lo: int, hi: int, final: bool) -> StreamChunk:
        return StreamChunk(embeds=self.embeds[lo:hi], final=final,
                           mrope_positions=self.mrope_positions[:, lo:hi], mrope_delta=self.mrope_delta)

    def chunks(self, bounds: list[int]) -> list[StreamChunk]:
        """bounds = [0, b1, ..., N] -> consecutive chunks, the last one final."""
        assert bounds[0] == 0 and bounds[-1] == self.embeds.shape[0], bounds
        return [self.chunk(lo, hi, final=(hi == bounds[-1])) for lo, hi in zip(bounds[:-1], bounds[1:])]


class _DuckItem:
    """Quacks like MultiModalKwargsItem for `MultiModalFeatureSpec.gather_kwargs` (0.11.2) and
    for `iter_mm_grid_thw`'s `data["image_grid_thw"].data` / `data.get(...)` (0.28.0)."""
    def __init__(self, **fields):
        self._f = {k: type("E", (), {"data": v})() for k, v in fields.items()}

    def __contains__(self, k):
        return k in self._f

    def __getitem__(self, k):
        return self._f[k]

    def get(self, k, default=None):
        return self._f.get(k, default)


class _DuckFeature:
    """Quacks like MultiModalFeatureSpec: `.data` (both releases), `.modality` and
    `.mm_position.offset` (0.28.0's `iter_mm_grid_thw` walks features by placeholder offset
    instead of scanning the token ids)."""
    def __init__(self, modality: str, offset: int, length: int, **fields):
        from vllm.multimodal.inputs import PlaceholderRange
        self.data = _DuckItem(**fields)
        self.modality = modality
        self.mm_position = PlaceholderRange(offset=offset, length=length)
        self.identifier = f"appcorr-{modality}-{offset}"


def _cpu_scatter_rows(dst: torch.Tensor, pos: torch.Tensor, rows: torch.Tensor) -> None:
    """``dst[pos] = rows`` for CPU tensors through numpy: torch's CPU ``index_put_`` on a few
    hundred rows fans out over the OpenMP pool (72 threads here) and costs milliseconds; a numpy
    memcpy does not.  bf16 has no numpy dtype, so both sides are viewed as int16."""
    assert dst.device.type == "cpu" and rows.device.type == "cpu" and rows.dtype == dst.dtype
    if dst.dtype == torch.bfloat16:
        d, r = dst.view(torch.int16), rows.contiguous().view(torch.int16)
    else:
        d, r = dst, rows.contiguous()
    d.numpy()[pos.numpy()] = r.numpy()


class StreamingLLM:
    # `correct(final=True)`: run the final step inside the engine step that releases the
    # hold-back (correct.py: fused hold-back) rather than before it. Env APPCORR_DEFER_FINAL=0
    # restores the synchronous form (gate reference).
    defer_final: bool = os.environ.get("APPCORR_DEFER_FINAL", "1") == "1"

    def __init__(self, model: str, *, gpu_memory_utilization: float = 0.35, max_model_len: int = 8192,
                 enforce_eager: bool = False, dtype: str = "bfloat16", **kw):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0", "in-process engine core required"
        from . import install
        install()
        from vllm import LLM
        self.model_name = model
        from . import vllm_version
        self.vllm_version = vllm_version()
        if self.vllm_version != "0.11.2":
            # 0.28.0 normalises pixel values on the device by default (HF processor run with
            # do_normalize/do_rescale off); the composer feeds the HF processor's normalised
            # output straight to the tower, so keep the stock arm on the same (CPU) path.
            kw.setdefault("mm_device_do_normalize", False)
            # 0.28.0 turns async scheduling on by default (AsyncScheduler + a 2-deep batch queue,
            # sampled ids kept on the GPU). The streaming hooks subclass the sync Scheduler and
            # were gated on the sync engine; keep the engine sync until the async path is ported.
            kw.setdefault("async_scheduling", False)
        self.llm = LLM(model=model, enable_prompt_embeds=True, enable_prefix_caching=False,
                       scheduler_cls="appcorr.vllm_stream.scheduler.StreamingScheduler",
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                       enforce_eager=enforce_eager, dtype=dtype, tensor_parallel_size=1, **kw)
        self.engine = self.llm.llm_engine
        self.max_model_len = int(self.engine.model_config.max_model_len)
        # 0.11.2: `LLMEngine.processor`; 0.28.0: `LLMEngine.input_processor` (+ `supported_tasks`)
        self.processor = getattr(self.engine, "input_processor", None) or self.engine.processor
        core = self.engine.engine_core  # InprocClient
        self.core = core.engine_core
        assert hasattr(self.core.scheduler, "stream_append"), type(self.core.scheduler)
        self._open: dict[str, float] = {}
        # 0.28.0 randomises request ids on `add_request` (external id -> "<id>-<8 hex>" inside the
        # engine; outputs carry the external id). appends must address the internal id.
        self._internal: dict[str, str] = {}

    # -- streaming API ---------------------------------------------------------------------
    def _ecr(self, request_id: str, chunk: StreamChunk, sp, mode: str):
        extra = {}
        if self.vllm_version != "0.11.2":
            extra["supported_tasks"] = self.engine.get_supported_tasks()
        return self.processor.process_inputs(request_id, {"prompt_embeds": chunk.embeds}, sp,
                                             arrival_time=time.time(),
                                             trace_headers=make_stream_headers(mode, chunk), **extra)

    def _core_id(self, request_id: str) -> str:
        return self._internal.get(request_id, request_id)

    def open(self, request_id: str, chunk: StreamChunk, sampling_params, *,
             correct: bool = False, capture_out: bool = False, image_start: int = 0,
             image_end: int = 0, open_walk: int = 0) -> None:
        """`correct=True` opens the request for interleaved correction: the runner keeps a side
        buffer of every GDN layer's pre-conv inputs for the whole prompt (see `correct.py`), which
        `self.correct(...)` then rewrites band by band. The whole (approximate) prompt must be in
        `chunk` and `chunk.final` must be False (the hold-back is released by the last correct).
        `image_start`/`image_end` = the image rows [lo, hi) -- `image_end` is needed by the
        depth-staged form only (its frontier walks cover exactly those rows).
        `open_walk` = b_0 > 0 (unified axis, memo §7.12): the engine does NOT run its stock
        full-depth prefill on this prompt (the scheduled prefill steps are no-ops for it);
        once they have been scheduled it walks rows [0, image_end) through layers [0, b_0)
        from the prompt embeddings (`correct.appcorr_open_walk`, run from `step` /
        `_drain_until_prefilled`), which is exactly the walk the first staged round used to
        run inside itself -- same state, off the first round's critical path, and no full
        prefill whose deep layers the walks re-do anyway."""
        assert sampling_params.max_tokens is not None, "set max_tokens: the prompt length is not known at open"
        mode = "oneshot" if chunk.final else "open"
        ecr = self._ecr(request_id, chunk, sampling_params, mode)
        ret = self.engine.add_request(request_id, ecr, sampling_params)
        self._internal[request_id] = ret if isinstance(ret, str) else request_id
        self._open[request_id] = time.perf_counter()
        if correct:
            assert not chunk.final, "a correcting request is opened with the whole approx prompt, not final"
            from . import correct as _correct
            _correct.check_gdn_path(self.engine.vllm_config)
            _correct.open_buffer(self._core_id(request_id), chunk.num_tokens,
                                 self.runner.device, lo=image_start, hi=image_end,
                                 capture_out=capture_out, open_walk=open_walk)
            if open_walk:
                assert image_end > image_start, "open_walk needs the image rows (image_end)"
                if self._open_walks is None:
                    self._open_walks = set()
                self._open_walks.add(self._core_id(request_id))
        else:
            assert not open_walk, "open_walk is an option of correct=True"

    _open_walks = None       # core ids whose open walk has not run yet

    def _run_open_walks(self) -> None:
        """Run the open walk of every request whose (no-op) prefill has been scheduled."""
        if not self._open_walks:
            return
        for core_id in list(self._open_walks):
            st = self.core.scheduler.stream_state(core_id)
            if st is None:
                self._open_walks.discard(core_id)
                continue
            if st["num_computed_tokens"] == st["num_prompt_tokens"] - 1:
                self.runner.appcorr_open_walk(core_id)
                self._open_walks.discard(core_id)

    # -- interleaved correction ---------------------------------------------------------------
    @property
    def runner(self):
        """The in-process `GPUModelRunner` (UniProcExecutor, VLLM_ENABLE_V1_MULTIPROCESSING=0)."""
        r = getattr(self, "_runner", None)
        if r is None:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner
            r = self.core.model_executor.driver_worker.worker.model_runner
            assert isinstance(r, GPUModelRunner), type(r)
            self._runner = r
        return r

    def _drain_until_prefilled(self, request_id: str, *, max_steps: int = 256,
                               timeout_s: float = 300.0, step_fn=None) -> dict:
        """Step the engine until the approx prefill is complete (computed == num_prompt - 1).
        `step_fn` replaces `self.step` so an owner that books outputs per step (the server's
        `_step`: first-token times, finished results) does not lose the other requests' outputs
        produced while draining."""
        step = step_fn or self.step
        t0 = time.perf_counter()
        for i in range(max_steps + 1):
            st = self.stream_state(request_id)
            if st is None:
                raise RuntimeError(f"{request_id}: request gone before the correct step")
            if not st["open"]:
                raise RuntimeError(f"{request_id}: correct on a closed request ({st})")
            if st["num_computed_tokens"] == st["num_prompt_tokens"] - 1:
                self._run_open_walks()
                return st
            if i == max_steps:
                break
            if time.perf_counter() - t0 > timeout_s:
                raise RuntimeError(f"{request_id}: approx prefill did not finish in {timeout_s}s ({st})")
            step()
        raise RuntimeError(f"{request_id}: approx prefill did not finish in {max_steps} steps "
                           f"({self.stream_state(request_id)})")

    def correct(self, request_id: str, positions, embeds, window, final: bool,
                *, replay: bool = False, step_fn=None, stage=None) -> dict:
        """Rewrite prompt rows `positions` with `embeds` and re-run the decoder on them.

        positions int64 [P] (sorted, all < N-1), embeds [P, D], window = (start, end) prompt
        positions of this round's DeltaNet re-scan. `final=True` also commits the recurrent state
        to the request's mamba block and releases the hold-back (empty final chunk).
        `stage=(r, g)` selects the depth-staged form (`correct.appcorr_staged_correct`): the
        rows are corrected over the first `b_r` layers only and the image rows are walked
        through the next layer band with the corrected context. `stage=(r, g, bounds)` is the
        same with explicit per-round depths (the unified vision+decoder axis, memo §7.12)."""
        assert request_id in self._open, request_id
        core_id = self._core_id(request_id)
        t_recv = time.perf_counter()
        self._drain_until_prefilled(request_id, step_fn=step_fn)
        deferred = final and self.defer_final
        if deferred:
            # the final step runs inside the engine step that computes the released row N-1
            # (correct.py: fused hold-back), armed here and run after the release below
            self.runner.appcorr_arm_final(core_id, positions, embeds, window, stage=stage,
                                          replay=replay)
            info = None
        elif stage is None:
            info = self.runner.appcorr_correct_step(core_id, positions, embeds, window, final,
                                                    replay=replay)
        else:
            info = self.runner.appcorr_staged_correct(core_id, positions, embeds, window, final,
                                                      stage, replay=replay)
        # keep both CPU copies of the prompt in sync so a preempted request re-prefills the
        # corrected rows (scheduler-side `StreamingRequest` and worker-side `CachedRequestState`)
        t_sc = time.perf_counter()
        pos_cpu = positions.detach().cpu().to(torch.int64)
        req = self.core.scheduler.requests[core_id]
        # clone: the caller's rows may alias the tensor `open` was given (same storage)
        rows = embeds.detach().to("cpu", req.prompt_embeds.dtype).clone()
        _cpu_scatter_rows(req.prompt_embeds, pos_cpu, rows)
        st = self.runner.requests[core_id]
        _cpu_scatter_rows(st.prompt_embeds, pos_cpu, rows.to(st.prompt_embeds.dtype))
        t_sc = time.perf_counter() - t_sc
        if final:
            d = req.prompt_embeds.shape[1]
            empty = StreamChunk(embeds=torch.empty((0, d), dtype=req.prompt_embeds.dtype),
                                final=True,
                                mrope_positions=torch.empty((3, 0), dtype=torch.int64),
                                mrope_delta=req.mrope_delta)
            # straight to the scheduler: vLLM's Processor rejects a zero-length prompt, and the
            # engine-core append path would only forward this to `stream_append` anyway.
            self.core.scheduler.stream_append(core_id, empty)
        if deferred:
            (step_fn or self.step)()
            info = self.runner.appcorr_take_final_info(core_id)
            if info is None:       # not scheduled in that step (c>1 budget): runs when it is
                info = {"num_rows": int(positions.numel()), "n_sub": 0, "deferred": True}
        info["t_recv"] = t_recv
        info["t_scatter_ms"] = t_sc * 1e3
        return info

    def append(self, request_id: str, chunk: StreamChunk) -> None:
        assert request_id in self._open, request_id
        core_id = self._core_id(request_id)
        sp = self.core.scheduler.requests[core_id].sampling_params
        ecr = self._ecr(request_id, chunk, sp, "final" if chunk.final else "append")
        ecr.request_id = core_id
        self.engine.engine_core.add_request(ecr)

    def step(self):
        """One engine step; returns the finished/streamed RequestOutputs of that step."""
        outs = self.engine.step()
        for o in outs:
            if o.finished:
                self._open.pop(o.request_id, None)
                self._internal.pop(o.request_id, None)
        self._run_open_walks()
        return outs

    def run_until_done(self, request_ids, max_steps: int = 100000):
        want = set([request_ids] if isinstance(request_ids, str) else request_ids)
        done = {}
        for _ in range(max_steps):
            for o in self.step():
                if o.finished and o.request_id in want:
                    done[o.request_id] = o
            if want <= set(done):
                return done
        raise RuntimeError(f"requests {want - set(done)} did not finish in {max_steps} steps")

    def stream_state(self, request_id: str):
        return self.core.scheduler.stream_state(self._core_id(request_id))

    # -- stock reference --------------------------------------------------------------------
    def generate(self, prompts, sampling_params):
        return self.llm.generate(prompts, sampling_params, use_tqdm=False)

    def apply_model(self, fn):
        return self.llm.apply_model(fn)[0]


class Qwen25VLComposer:
    """Builds the exact prompt vLLM would run for (image, question) and turns it into embeddings
    using the vLLM model's own embedding table and vision tower, plus stock M-RoPE positions.

    Covers the whole Qwen2-VL family as vLLM models it: Qwen2.5-VL and Qwen3.5 (`<|image_pad|>`
    span, `get_mrope_input_positions` over placeholder features, tower -> [T, D]). Qwen3-VL proper
    ships deepstack (`vision_config.deepstack_visual_indexes` non-empty: extra per-level features
    added at early LM layers through a side buffer the embeds path never fills); the Qwen3.5
    checkpoints ship it EMPTY, so they take the plain path -- `embed` asserts that."""

    def __init__(self, llm: StreamingLLM):
        from transformers import AutoProcessor
        self.llm = llm
        # the engine's resolved (local snapshot) path: transformers' tokenizer loader otherwise
        # queries the hub for a repo id even under HF_HUB_OFFLINE
        self.hf = AutoProcessor.from_pretrained(llm.engine.model_config.model)
        self.image_pad_id = self.hf.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def messages(self, question: str):
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]

    def prompt_text(self, question: str) -> str:
        return self.hf.apply_chat_template(self.messages(question), tokenize=False, add_generation_prompt=True)

    def parts(self, image, question: str) -> PromptParts:
        text = self.prompt_text(question)
        enc = self.hf(text=[text], images=[image], return_tensors="pt")
        ids = enc["input_ids"][0]
        pad = (ids == self.image_pad_id).nonzero().flatten()
        assert pad.numel() > 0 and int(pad[-1] - pad[0]) == pad.numel() - 1, "one contiguous image span expected"
        return PromptParts(input_ids=ids, image_start=int(pad[0]), image_len=int(pad.numel()),
                           pixel_values=enc["pixel_values"], image_grid_thw=enc["image_grid_thw"], text=text)

    def embed(self, parts: PromptParts, image_embeds: Optional[torch.Tensor] = None) -> EmbeddedPrompt:
        """Full-prompt embeddings: text rows from the LM embedding table, image rows from
        `image_embeds` [image_len, D] if given (e.g. AppCorr's progressive tower) else from the
        vLLM model's own vision tower."""
        ids, pv, thw = parts.input_ids, parts.pixel_values, parts.image_grid_thw
        i0, L = parts.image_start, parts.image_len
        vllm_config = self.llm.engine.vllm_config  # not every model class keeps `.vllm_config`

        def fn(model):
            from vllm.forward_context import set_forward_context
            dev = next(model.parameters()).device
            assert getattr(model, "deepstack_num_level", 0) == 0, (
                "deepstack model: the [T, D] embeds wire format cannot carry the per-level features")
            with torch.no_grad():
                emb = model.embed_input_ids(ids.to(dev))
                if image_embeds is None:
                    with set_forward_context(None, vllm_config):
                        vis = model.visual(pv.to(dev, model.visual.dtype), grid_thw=thw.tolist())
                else:
                    vis = image_embeds.to(dev, emb.dtype)
                assert vis.shape[0] == L, (vis.shape, L)
                emb = emb.clone()
                emb[i0:i0 + L] = vis.to(emb.dtype)
                feat = _DuckFeature("image", i0, L, image_grid_thw=thw[0])
                pos, delta = model.get_mrope_input_positions(ids.tolist(), [feat])
            return emb.cpu(), pos.cpu().to(torch.int64), int(delta)

        emb, pos, delta = self.llm.apply_model(fn)
        return EmbeddedPrompt(embeds=emb, mrope_positions=pos, mrope_delta=delta, parts=parts)

    def image_bounds(self, parts: PromptParts, num_chunks: int) -> list[int]:
        """Chunk boundaries that split the image span into `num_chunks` (row-aligned to the merged
        grid) with the leading text in the first chunk and the trailing text in the last."""
        i0, L, N = parts.image_start, parts.image_len, parts.num_tokens
        w = int(parts.image_grid_thw[0, 2]) // 2  # merged grid width
        rows = L // w
        cuts = [i0 + (rows * k // num_chunks) * w for k in range(1, num_chunks)]
        return [0, *cuts, N]


QwenVLComposer = Qwen25VLComposer  # Qwen2.5-VL / Qwen3.5 (see class docstring)
