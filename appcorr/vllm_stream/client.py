"""In-process client: `StreamingLLM` (open / append / step) and Qwen2.5-VL prompt composition.

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
    """Quacks like MultiModalKwargsItem for `MultiModalFeatureSpec.gather_kwargs`."""
    def __init__(self, **fields):
        self._f = {k: type("E", (), {"data": v})() for k, v in fields.items()}

    def __contains__(self, k):
        return k in self._f

    def __getitem__(self, k):
        return self._f[k]


class _DuckFeature:
    def __init__(self, **fields):
        self.data = _DuckItem(**fields)


class StreamingLLM:
    def __init__(self, model: str, *, gpu_memory_utilization: float = 0.35, max_model_len: int = 8192,
                 enforce_eager: bool = False, dtype: str = "bfloat16", **kw):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0", "in-process engine core required"
        from . import install
        install()
        from vllm import LLM
        self.model_name = model
        self.llm = LLM(model=model, enable_prompt_embeds=True, enable_prefix_caching=False,
                       scheduler_cls="appcorr.vllm_stream.scheduler.StreamingScheduler",
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                       enforce_eager=enforce_eager, dtype=dtype, tensor_parallel_size=1, **kw)
        self.engine = self.llm.llm_engine
        self.processor = self.engine.processor
        core = self.engine.engine_core  # InprocClient
        self.core = core.engine_core
        assert hasattr(self.core.scheduler, "stream_append"), type(self.core.scheduler)
        self._open: dict[str, float] = {}

    # -- streaming API ---------------------------------------------------------------------
    def _ecr(self, request_id: str, chunk: StreamChunk, sp, mode: str):
        return self.processor.process_inputs(request_id, {"prompt_embeds": chunk.embeds}, sp,
                                             arrival_time=time.time(),
                                             trace_headers=make_stream_headers(mode, chunk))

    def open(self, request_id: str, chunk: StreamChunk, sampling_params) -> None:
        assert sampling_params.max_tokens is not None, "set max_tokens: the prompt length is not known at open"
        mode = "oneshot" if chunk.final else "open"
        ecr = self._ecr(request_id, chunk, sampling_params, mode)
        self.engine.add_request(request_id, ecr, sampling_params)
        self._open[request_id] = time.perf_counter()

    def append(self, request_id: str, chunk: StreamChunk) -> None:
        assert request_id in self._open, request_id
        sp = self.core.scheduler.requests[request_id].sampling_params
        ecr = self._ecr(request_id, chunk, sp, "final" if chunk.final else "append")
        self.engine.engine_core.add_request(ecr)

    def step(self):
        """One engine step; returns the finished/streamed RequestOutputs of that step."""
        outs = self.engine.step()
        for o in outs:
            if o.finished:
                self._open.pop(o.request_id, None)
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
        return self.core.scheduler.stream_state(request_id)

    # -- stock reference --------------------------------------------------------------------
    def generate(self, prompts, sampling_params):
        return self.llm.generate(prompts, sampling_params, use_tqdm=False)

    def apply_model(self, fn):
        return self.llm.apply_model(fn)[0]


class Qwen25VLComposer:
    """Builds the exact prompt vLLM would run for (image, question) and turns it into embeddings
    using the vLLM model's own embedding table and vision tower, plus stock M-RoPE positions."""

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

        def fn(model):
            from vllm.forward_context import set_forward_context
            dev = next(model.parameters()).device
            with torch.no_grad():
                emb = model.embed_input_ids(ids.to(dev))
                if image_embeds is None:
                    with set_forward_context(None, model.vllm_config):
                        vis = model.visual(pv.to(dev, model.visual.dtype), grid_thw=thw.tolist())
                else:
                    vis = image_embeds.to(dev, emb.dtype)
                assert vis.shape[0] == L, (vis.shape, L)
                emb = emb.clone()
                emb[i0:i0 + L] = vis.to(emb.dtype)
                feat = _DuckFeature(image_grid_thw=thw[0])
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
