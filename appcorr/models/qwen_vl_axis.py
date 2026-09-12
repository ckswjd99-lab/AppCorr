"""
qwen_vl_axis.py

The progressive-arrival axis shared by the Qwen2-VL family (Qwen2.5-VL, Qwen3.5): vision tower
approximates-then-corrects per band, the LLM streams (chunked prefill per band). The streaming
loop here is `appcorr/models/qwen35/unified.py`'s, lifted verbatim once a second model needed it;
the per-model subclasses (`Qwen35Axis`, `Qwen25VLAxis`) only supply the tower and the four places
where the two vision forks differ:

    _approx_base      the tower's full-depth base approx (attention-mean collection kwarg differs)
    _attn_layermean   where the received-attention mean lives afterwards (key name differs)
    _rows_of_groups   merge-group index -> the tower's ROW indices for that group. Qwen3.5's rows
                      are in natural order (identity); Qwen2.5-VL's tower permutes rows by window
                      (`window_index`) before its block loop, so a group's rows are wherever the
                      permutation put them. Everything the loop touches at row granularity (the
                      arrived-rows mask, the band the merger sees, the keep<1 row mask, the
                      attention pooled to groups) goes through this one mapping.
    build_inputs      chat-template kwargs (Qwen3.5 has a thinking switch)

The LLM side is identical: same `get_rope_index` contract (3, B, T), same `inputs_embeds +
position_ids + DynamicCache` chunked prefill, same `mm_token_type_ids` marking of the image run.

**Two LLM backends, one vision path.** `streaming_forward(..., sink=None)` runs the LLM in
process (HF, the path every table number so far went through). With `sink=` (a
`appcorr.vllm_stream.bridge.StreamSink`), the same loop sends each chunk -- rows of `emb_all`
plus their M-RoPE positions -- to the vLLM streaming server instead of calling the HF model, and
returns no logits: the answer comes back from the server (`sink.result()`), decoded by vLLM's
greedy loop. The vision work, the band schedule and the chunk boundaries are byte-for-byte the
same in both modes; only who consumes the chunks changes. `oneshot_embeds` gives the floor and
ceiling arms the same one-request path (stock tower, one chunk), so all arms of a vLLM campaign
decode through the identical engine -- the decode-mechanism-consistency rule the HF driver
enforces with its shared greedy loop, restated for the two-process form.

**Two LLM schedules over that one vision path.** `llm_schedule="streaming"` (default) appends each
band to the prompt; `llm_schedule="interleaved"` pushes the whole approximate prompt at t=0 and
rewrites each band's rows in place through `StreamSink.correct` (docs/memo/
vllm_interleaved_design.md). The vision work and the band boundaries are identical in both -- only
what the sink is told changes -- so at keep=1 the per-position embeddings the LLM ends up holding
are bitwise equal (gate G5, `analysis/experiments/vllm_interleaved_axis_gate.py`).
`llm_schedule="interleaved_staged"` sends the same messages tagged with the round index so the
engine runs the depth-staged form (memo §7.11): round r's rows are corrected over the first
`b_r` decoder layers and every image row is then walked through the next layer band with the
corrected context -- the ProgVFM §3.3 schedule, whose k<1 result differs from the unstaged one.
`llm_schedule="unified_staged"` puts the VISION TOWER inside that staging (memo §7.12): the
tower's layers and the decoder's form ONE axis, cut into `groups` rounds of equal COST, so the
early rounds fall inside the tower and the LLM is opened only when a bound crosses the projector
-- the whole prompt then goes out with the bands corrected so far already corrected. It is the
gemma3 `interleaved_forward` walk (`appcorr/models/gemma3/unified.py`) on the Qwen3.5 pair.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# PyTorch's cuDNN SDPA backend returns a WRONG output for some attention heads on the B200 in
# this env (torch 2.12.1+cu130, cuDNN 9.20): found 2026-09-07 through the vLLM bridge gate --
# the HF twin decoded "ribbeded" / "dotteded" where vLLM, eager, the flash and the math
# backends all decode "ribbed" / "dotted". On the captured decode-step call (q [1,28,1,128],
# kv [1,4,382,128], real Qwen2.5-VL-7B activations) one head is off by 1.56 (flash: 4e-3);
# random tensors of the same shape do not trigger it, so it is data-dependent. Eligibility is
# head_dim <= 128 (the backend refuses head_dim 256, so Qwen3.5 / Gemma never hit it); the
# error is confined to q_len=1 decode steps in every case measured (prefill logits within bf16
# noise). Repro tensors: analysis/results/vllm_stream/sdpa_cudnn_repro.pt. Every HF-side
# consumer imports this module, so the switch lives here.
torch.backends.cuda.enable_cudnn_sdp(False)


class QwenVLStreamingAxis(nn.Module):
    # Family default: the text model consumes M-RoPE (3, T) positions. GLM-5.3-Flash sets
    # this False (see the instance attribute's comment in `__init__`).
    uses_mrope = True

    def __init__(self, model: nn.Module, processor: Any, flop_counter: Optional[Any] = None):
        super().__init__()
        self.model = model
        self.processor = processor
        self.flops = flop_counter
        self.tower = self._make_tower(model)
        self.lm = model.model.language_model
        self.cfg = model.config
        self.image_token_id = model.config.image_token_id
        # Throughput knobs (2026-09-08, docs/memo/qwen_correct_forward_profile.md). All default to
        # the fast form; every one of them is bitwise on image_embeds + positions (gated by
        # analysis/experiments/qwen_axis_snapshot.py), the slow forms stay as the in-process
        # reference. correct_rows_only: carry only the corrected rows through the tower
        # (`tower.correct_rows`) instead of the full [T, D] stream (`correct_forward`).
        # positions_mode: "fast" (closed-form single-image M-RoPE, no host sync), "reference"
        # (transformers' get_rope_index), "check" (both, raise on mismatch).
        # image_embeds_with_sink: keep the fp32 image-row copy in `stats` even when the chunks
        # left the process (only the in-process gates read it; 250 MB at 15k tokens).
        self.correct_rows_only = True
        # Unified axis only: once the frontier has crossed the projector no approximate range
        # reads the vision stream again, so the rounds after the crossing satisfy the
        # `correct_rows` contract (every corrected row read from layer 0, non-corrected rows
        # never read: the merge takes them from the crossing stream `x_base_out`) and can drop
        # the [T, D] reconstruction + rule-3 write-back of `correct_forward`. Rows are bitwise
        # the same (gated: analysis/experiments/vllm_unified_hybrid_gate.py); the last band's
        # vision correct was 62 vs 34 ms on V*Bench with the full-stream form (2026-09-11).
        self.unified_rows_after_crossing = True
        # Unified axis only: the opening push asks the engine to SKIP its stock full-depth
        # prefill and instead walk the prompt rows through the first LLM stage `[0, b_0)` at
        # open (`open_walk`), so the served work is the closed form's (approx pass spread over
        # the rounds) and the first LLM round does not carry a full prefill on its critical path.
        # State-identical to the stock-prefill form (the same walk ran inside the first round
        # before); gated served (vllm_unified_gate.py).
        self.engine_open_walk = True
        self.positions_mode = "fast"
        # `uses_mrope`: does the TEXT model consume the (3, T) t/h/w position tensor?
        # True for every Qwen2-VL-family model and for GLM-4.6V. FALSE for GLM-5.3-Flash,
        # whose decoder has no rotary at all (34 KDA layers ignore `positions`, the 11
        # MLA layers are built with `skip_rope=config.mla_nope`), so vLLM's
        # `model_config.uses_mrope` is False and the engine never calls
        # `_init_mrope_positions`. A streaming chunk MUST then carry `mrope=None`: the
        # runner patch asserts `st.mrope_positions is not None` before extending it
        # (`appcorr/vllm_stream/runner_patch.py:45-48`) and that field is only ever filled
        # by `_init_mrope_positions`, which the runner skips for a non-M-RoPE model
        # (vLLM main `v1/worker/gpu_model_runner.py:1343-1345, 1676-1677`). Pushing a
        # broadcast 1-D tensor instead would crash on the first appended chunk.
        self.uses_mrope = type(self).uses_mrope
        self.image_embeds_with_sink = False
        # keep<1 selection score: True = the attention term is computed after the first band's
        # push (band 0 selects on the residual-energy hint alone), False = the eager score
        # (attention before band 0; the arms measured up to 2026-09-09). See streaming_forward.
        self.pscore_defer = True

    # --- per-model hooks (subclasses) ----------------------------------------------------------- #

    def _make_tower(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError

    # Whether the tower can stash the base pass's queries and compute the received-attention
    # term of the selection score later (`_attn_layermean_deferred`). Off = the eager path: the
    # score is complete before the first band and delays the first push by the whole column
    # sum (60 ms on the 35B RWQA probe, O(T^2) memory traffic over 27 layers).
    supports_deferred_pscore = False

    def _approx_base(self, ctx_base: Dict[str, Any], cache: Dict[str, Any],
                     collect_attn) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Full-depth approx of the base image; with `collect_attn` True, also leave the per-row
        received-attention mean where `_attn_layermean` finds it; with "defer" (only where
        `supports_deferred_pscore`), leave what `_attn_layermean_deferred` needs instead."""
        raise NotImplementedError

    def _attn_layermean(self, cache: Dict[str, Any]) -> torch.Tensor:
        """[n_rows] received-attention mean, in the tower's row order."""
        raise NotImplementedError

    def _attn_layermean_deferred(self, cache: Dict[str, Any], ctx_base: Dict[str, Any]) -> torch.Tensor:
        """The same vector as `_attn_layermean`, computed now from what a deferred base pass left."""
        raise NotImplementedError

    def _rows_of_groups(self, ctx: Dict[str, Any], group_idx: torch.Tensor) -> torch.Tensor:
        """[G * unit] row indices (tower row order) of the given ORIGINAL merge-group indices,
        group-major, so `x[rows].reshape(G, unit, D)` is group g's rows in order."""
        unit = self.tower.spatial_merge_unit
        return (group_idx.unsqueeze(1) * unit
                + torch.arange(unit, device=group_idx.device)).flatten()

    # --- unified stage axis (vision layers then decoder layers) --------------------------------- #
    #
    # `llm_schedule="unified_staged"` walks ONE axis of `n_vision + n_llm` stages, split into
    # rounds by equal COST (gemma3's `layer_bounds`, ported). Counting stages would be wrong: a
    # vision layer runs 4 x n_image_tokens rows at width 1152 and a decoder layer N tokens at
    # width 2048-3072 with three layers in four recurrent -- not interchangeable units. The
    # bounds are therefore a function of the REQUEST (n_rows, N), computed per request.

    supports_unified_axis = False        # subclasses that implement the two cost hooks

    def _vision_stage_cost(self, n_rows: int) -> float:
        """FLOPs of ONE vision-tower layer over `n_rows` patch rows (full attention)."""
        raise NotImplementedError

    def _llm_stage_costs(self, n_prompt: int) -> List[float]:
        """FLOPs of EACH decoder layer over an `n_prompt`-token prefill. A list, not one number:
        the Qwen3.5 decoder is hybrid (softmax vs Gated DeltaNet layers cost differently), so the
        bound must fall where the cumulative cost says, not where a layer count says."""
        raise NotImplementedError

    def _approx_range(self, ctx: Dict[str, Any], cache: Dict[str, Any], start_l: int, end_l: int,
                      x: Optional[torch.Tensor] = None, collect_attn: bool = False):
        """Approximate walk of vision layers `[start_l, end_l)` on `x` (the layer-0 stream of
        `ctx` when None). The unified axis's replacement for `_approx_base`'s full-depth pass."""
        raise NotImplementedError

    def _attn_layermean_prefix(self, cache: Dict[str, Any], n_layers: int) -> torch.Tensor:
        """[n_rows] received-attention mean over the first `n_layers` vision layers."""
        raise NotImplementedError

    def unified_stage_costs(self, n_rows: int, n_prompt: int) -> List[float]:
        return ([self._vision_stage_cost(n_rows)] * len(self.tower.blocks)
                + self._llm_stage_costs(n_prompt))

    def unified_bounds(self, groups: int, n_rows: int, n_prompt: int) -> List[int]:
        """Round boundaries over the unified axis, split by equal cumulative COST.

        `bounds[r]` is the depth round r corrects at (and the depth its own approximate frontier
        sits at when it starts); the frontier then advances to `bounds[r+1]`. The last bound is
        always the whole axis, so the final round is full depth -- which is what makes `groups=1`
        the identity against today's `interleaved` and keeps the critical path unchanged.
        """
        costs = self.unified_stage_costs(n_rows, n_prompt)
        n = len(costs)
        if groups <= 1:
            return [n]
        if groups > n:
            # The de-duplication loop below can only invent bounds that exist; asking for more
            # rounds than stages would spin forever looking for one (gemma3's `layer_bounds`
            # has the same loop and the same hole -- it never bites there because 61 stages are
            # never split 61 ways, and 67/75 here are not either).
            raise ValueError(f"{groups} rounds over a {n}-stage axis: at most one round per stage")
        total = sum(costs)
        cum, acc = [], 0.0
        for c in costs:
            acc += c
            cum.append(acc / total)
        bounds = []
        for r in range(1, groups):
            target = r / groups
            b = next((i + 1 for i, c in enumerate(cum) if c >= target), n)
            bounds.append(max(1, min(b, n - 1)))
        bounds = sorted(set(bounds))
        while len(bounds) < groups - 1:            # keep `groups` distinct rounds
            for cand in range(1, n):
                if cand not in bounds:
                    bounds.append(cand)
                    break
            bounds = sorted(set(bounds))
        return bounds + [n]

    # --- flop scopes (same shape as gemma3/ov2) ------------------------------------------------- #

    def _arrival(self, index: int):
        return self.flops.arrival(index) if self.flops is not None else nullcontext()

    def _stage(self, name: str):
        return self.flops.stage(name) if self.flops is not None else nullcontext()

    # --- shared prep ---------------------------------------------------------------------------- #

    def _chat_template_kwargs(self, **kw) -> Dict[str, Any]:
        return {}

    @torch.no_grad()
    def build_inputs(self, image, question: str, **kw) -> Dict[str, Any]:
        """Chat-template + pixel preprocessing for one (image, question) request. Guarantees
        `mm_token_type_ids` (1 on the image run) is present -- the M-RoPE contract every path here
        relies on -- computing it from `input_ids` if the processor did not."""
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": question}]}]
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", **self._chat_template_kwargs(**kw),
        )
        if "mm_token_type_ids" not in inputs:
            inputs["mm_token_type_ids"] = (inputs["input_ids"] == self.image_token_id).long()
        return inputs

    def _image_token_run(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """(start, count) of the single contiguous image-token run. Raises on anything else --
        multiple images or a fragmented run would make a band not be one prefill chunk."""
        pos = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            raise ValueError("no image tokens in input_ids")
        if not bool((pos[1:] - pos[:-1] == 1).all()):
            raise ValueError("image tokens are not one contiguous run; streaming bands need one")
        return int(pos[0]), int(pos.numel())

    def _bands(self, groups: int, n_merge_groups: int) -> List[Tuple[int, int]]:
        """Split [0, n_merge_groups) into `groups` contiguous bands (sequential grouping)."""
        edges = [round(k * n_merge_groups / groups) for k in range(groups + 1)]
        return [(a, b) for a, b in zip(edges[:-1], edges[1:])]

    def _positions_reference(self, inputs: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        """M-RoPE positions (3, 1, T) for the whole request + the rope delta (an int: what the
        model adds to a plain counter for every generated token), through transformers' own
        `get_rope_index`. The model's own fallback when `position_ids is None` (and any
        hand-rolled `arange`) replicates a 1D counter across the axes, silently destroying the
        image grid -- the M-RoPE trap hit on Qwen2.5-VL, where BOTH arms of an A/B shared the
        wrong positions and agreed with each other. Shape (3, B, T) is the t/h/w axes only,
        exactly what stock's own forward hands its text model. Costs a `.tolist()` of the whole
        prompt, a Python groupby over it and two `.item()`s -- the reference `_positions_fast`
        is checked against, not the campaign path."""
        ids = inputs["input_ids"]
        mm_ttids = inputs.get("mm_token_type_ids")
        if mm_ttids is None:
            mm_ttids = (ids == self.image_token_id).long()
        pos_3d, rope_deltas = self.model.model.get_rope_index(
            ids, mm_ttids, image_grid_thw=inputs["image_grid_thw"])
        seq = int(ids.shape[1])
        if pos_3d.shape[0] != 3 or pos_3d.shape[-1] != seq:
            raise ValueError(f"get_rope_index returned {tuple(pos_3d.shape)}, expected (3, 1, {seq})")
        return pos_3d, int(rope_deltas.flatten()[0].item())

    def _positions_fast(self, inputs: Dict[str, Any], image_run: Optional[Tuple[int, int]] = None,
                        grid_thw: Optional[Tuple[int, int, int]] = None) -> Tuple[torch.Tensor, int]:
        """Closed form of `get_rope_index` for the one layout this axis serves -- one image, one
        contiguous run, text on both sides: text before the run counts 0..lo-1 on all three
        axes; the image rows sit at t = lo, h = lo + row, w = lo + col of the MERGED grid; text
        after it resumes at lo + max(h_m, w_m). Integer-exact, so bitwise against the reference
        (checked by `positions_mode="check"`); no host sync when `image_run` and `grid_thw`
        come in as Python ints (the driver's CPU-side prep has both)."""
        ids = inputs["input_ids"]
        grid = inputs["image_grid_thw"]
        if grid.shape[0] != 1:
            raise ValueError(f"{grid.shape[0]} images; the streaming axis serves exactly one")
        seq = int(ids.shape[1])
        lo, n_tok = image_run if image_run is not None else self._image_token_run(ids)
        if grid_thw is None:
            grid_thw = tuple(int(v) for v in grid[0].tolist())
        t, h, w = (int(v) for v in grid_thw)
        m = int(self.cfg.vision_config.spatial_merge_size)
        h_m, w_m = h // m, w // m
        if t != 1 or h_m * w_m != n_tok:
            raise ValueError(f"grid {(t, h, w)} / merge {m} does not give {n_tok} image tokens")
        trailing = seq - lo - n_tok
        if lo < 0 or trailing < 0:
            raise ValueError(f"image run [{lo}, {lo + n_tok}) does not fit in {seq} tokens")
        after = lo + max(h_m, w_m)
        max_pos = (after + trailing - 1) if trailing > 0 else (lo + max(h_m, w_m) - 1)
        dev = ids.device
        pos = torch.empty(3, 1, seq, dtype=ids.dtype, device=dev)
        ar = torch.arange(seq, dtype=ids.dtype, device=dev)
        pos[:, 0, :lo] = ar[:lo]
        img = torch.arange(n_tok, dtype=ids.dtype, device=dev)
        pos[0, 0, lo:lo + n_tok] = lo
        pos[1, 0, lo:lo + n_tok] = img // w_m + lo
        pos[2, 0, lo:lo + n_tok] = img % w_m + lo
        pos[:, 0, lo + n_tok:] = ar[:trailing] + after
        return pos, max_pos + 1 - seq

    def _positions(self, inputs: Dict[str, Any], image_run: Optional[Tuple[int, int]] = None,
                   grid_thw: Optional[Tuple[int, int, int]] = None) -> Tuple[torch.Tensor, int]:
        """See `positions_mode`: the fast closed form by default, the transformers reference on
        request, or both with an equality check (the snapshot gate's mode)."""
        if self.positions_mode == "reference":
            return self._positions_reference(inputs)
        pos, delta = self._positions_fast(inputs, image_run, grid_thw)
        if self.positions_mode == "check":
            pos_ref, delta_ref = self._positions_reference(inputs)
            if delta != delta_ref or not torch.equal(pos, pos_ref.to(pos.device)):
                raise RuntimeError(f"fast M-RoPE positions differ from get_rope_index "
                                   f"(delta {delta} vs {delta_ref})")
        elif self.positions_mode != "fast":
            raise ValueError(f"positions_mode {self.positions_mode!r}")
        return pos, delta

    # --- the reference -------------------------------------------------------------------------- #

    @torch.no_grad()
    def full_forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Stock forward, final-position logits. The ceiling and the gates' reference."""
        with self._arrival(0), self._stage("full"):
            out = self.model(input_ids=inputs["input_ids"],
                             pixel_values=inputs["pixel_values"].to(self.model.dtype),
                             image_grid_thw=inputs["image_grid_thw"],
                             mm_token_type_ids=inputs["mm_token_type_ids"], use_cache=False)
        return out.logits[:, -1]

    # --- the arm -------------------------------------------------------------------------------- #

    @torch.no_grad()
    def streaming_forward(self, inputs: Dict[str, Any], px_base: torch.Tensor,
                          groups: int, keep: float = 1.0,
                          sink: Optional[Any] = None,
                          image_run: Optional[Tuple[int, int]] = None,
                          grid_thw: Optional[Tuple[int, int, int]] = None,
                          llm_schedule: str = "streaming",
                          ) -> Tuple[Optional[torch.Tensor], Any, Dict[str, Any]]:
        """Progressive arrival: vision approximates-then-corrects per band, the LLM streams.

        Args:
            inputs: `build_inputs` output built with the FULL-resolution image (its pixel_values
                are the ground truth the bands converge to).
            px_base: pixel_values of the DEGRADED base image, same grid (the transmission's level-2
                base). Same shape as inputs["pixel_values"].
            groups: arrival rounds. groups=1 must reproduce `full_forward` exactly (in exact
                arithmetic): one band corrected after everything arrived = no staleness anywhere.
            keep: fraction of image tokens corrected in total (the standard 0.25/0.50 arms;
                1.0 = the original streaming arm and the identity-gate case). Band r selects its
                quota among arrived-and-uncorrected tokens by residual energy x received
                attention; the attention term rides the base approx pass this arm already runs
                (full depth -- the base approx is not frontier-chunked, so no extra pass exists to
                duplicate). Unselected tokens enter the LLM at their approximate reconstruction
                and, this being streaming, are never revisited -- that permanence is the knob's
                cost, and what the accuracy arms price.
                With `pscore_defer` (default where the tower supports it) the attention column
                sum runs AFTER the first band's push instead of inside the base pass: band 0
                selects on the energy hint alone, bands 1.. on energy x attention. The budget,
                the quota split and the per-band ranking are unchanged (the score's global
                normalisations are per-request scalars, so the within-band ranking never
                depended on them); only band 0's ranking differs from the eager arm. Motivation:
                the sum is O(T^2) memory traffic over every layer, excluded from the FLOP columns
                (`PSCORE`), yet it sat on the first push's critical path -- +63 ms of first-push
                latency on the 35B RWQA probe, and with the engine's per-step floor that became
                the whole keep<1 vs keep=1 TTFT gap (2026-09-09).
            sink: optional `StreamSink`. When given, every chunk goes to the vLLM server instead of
                the in-process HF model and the return is `(None, None, stats)`; read the answer
                from `sink.result()`.
            image_run: optional (start, count) of the image-token run, and grid_thw the (t, h, w)
                patch grid, both as Python ints from the CPU side of the driver's prep; when
                given, nothing about the prompt layout is read back from the GPU.
            llm_schedule: how the LLM consumes the bands (docs/memo/vllm_interleaved_design.md).
                "streaming" (default) prefills each band once, append-only: band r's chunk is
                rows [lo+g0_r, lo+g1_r) and no row is ever revisited. "interleaved" pushes the
                WHOLE prompt once at t=0 with every image row at its approximate (base) merge,
                lets the engine prefill it, and then REWRITES each band's corrected rows in
                place as they are produced -- so the critical path after the last arrival is
                k/g of the image rows plus the text suffix, not the whole prompt. Sink path
                only: rewriting rows of a KV cache the HF model already built is exactly the
                engine-side work `StreamingLLM.correct` exists for. "interleaved_staged" is
                the same message sequence with `stage=(r, g)` on every correct (depth-staged
                rounds on the engine; g = number of non-empty bands). "unified_staged" walks
                the tower and the decoder as ONE cost-split axis: the approximate pass is
                chunked by layer range too, band r is corrected only over the stages walked so
                far, and the opening push happens when the frontier crosses the projector, so
                it carries the earlier bands already corrected. Fewer LLM rounds than `groups`,
                at explicit bounds -- `stage=(r, n_llm_rounds, bounds)` on the wire.

        Returns (final_position_logits, kv_cache, stats).
        """
        from transformers.cache_utils import DynamicCache

        if llm_schedule not in ("streaming", "interleaved", "interleaved_staged",
                                "unified_staged"):
            raise ValueError(f"llm_schedule {llm_schedule!r}")
        unified = llm_schedule == "unified_staged"
        interleaved = llm_schedule.startswith("interleaved") or unified
        staged = llm_schedule in ("interleaved_staged", "unified_staged")
        if unified and not self.supports_unified_axis:
            raise NotImplementedError(
                f"{type(self).__name__} has no unified stage axis (the two cost hooks and the "
                "chunked approx walk); it is implemented for Qwen3.5 only")
        if interleaved and sink is None:
            raise NotImplementedError(
                "llm_schedule='interleaved' is sink-only: rewriting rows an HF DynamicCache has "
                "already prefilled is the engine-side `StreamingLLM.correct` step (memo §2)")
        ids = inputs["input_ids"]
        grid = inputs["image_grid_thw"]
        px_full = inputs["pixel_values"].to(self.model.dtype)
        px_base = px_base.to(device=px_full.device, dtype=px_full.dtype)
        if px_base.shape != px_full.shape:
            raise ValueError(f"base/full pixel grids differ: {tuple(px_base.shape)} vs "
                             f"{tuple(px_full.shape)} -- degrade content, never geometry")

        lo, n_tok = image_run if image_run is not None else self._image_token_run(ids)
        seq = int(ids.shape[1])
        unit = self.tower.spatial_merge_unit
        dev = px_full.device
        # `correct_rows` is a full-depth shortcut that skips the rule-3 write-back and the
        # [T, D] reconstruction; the unified axis needs both while its later approx ranges READ
        # the stream a correction produced, i.e. up to the projector crossing -- after it the
        # per-round `rows_r` below switches to the shortcut (`unified_rows_after_crossing`).
        rows_only = self.correct_rows_only and not unified

        # Arrival 0: everything that needs only the base image. The grid-only prep (positions,
        # rotary tables, segment ranges) is shared by the two embeds -- it is grid-shape-exact,
        # and the two images share the grid; the base approx pass gives every patch row a value
        # and every layer a K/V.
        cache: Dict[str, Any] = {}
        with self._arrival(0):
            with self._stage("prepare"):
                gctx = self.tower.prepare_grid(grid, dev)
                ctx_full = self.tower.prepare_full_tokens(px_full, grid, gctx)
                ctx_base = self.tower.prepare_full_tokens(px_base, grid, gctx)
            n_rows = ctx_full["seq_len"]
            n_groups_total = n_rows // unit
            if n_tok != n_groups_total:
                raise ValueError(f"{n_tok} image tokens vs {n_groups_total} merge groups")
            # The deferred score buys nothing on the unified axis: it hides the attention column
            # sum behind the first PUSH, and the unified schedule has not pushed anything when
            # band 0 must be selected (the LLM opens only when the frontier crosses the tower).
            defer = (keep < 1.0 and self.pscore_defer and self.supports_deferred_pscore
                     and not unified)
            n_vis = len(self.tower.blocks)
            if unified:
                # One axis: tower layers 0..n_vis-1 then decoder layers 0..n_llm-1, cut into
                # `groups` rounds of equal COST. Round r corrects at depth bounds[r] and then
                # advances the frontier to bounds[r+1] (gemma3/unified.py's walk).
                n_llm = int(self.cfg.text_config.num_hidden_layers)
                bounds = self.unified_bounds(groups, int(n_rows), seq)
                assert len(bounds) == groups and bounds[-1] == n_vis + n_llm, bounds
                v_front = min(bounds[0], n_vis)
                with self._stage("vision_base"):
                    x_v, cache = self._approx_range(ctx_base, cache, 0, v_front,
                                                    collect_attn=keep < 1.0)
                x_base_out = x_v          # the frontier stream; becomes the crossing merge below
            else:
                with self._stage("vision_base"):
                    x_base_out, cache = self._approx_base(
                        ctx_base, cache, collect_attn=("defer" if defer else keep < 1.0))

        emb_all = self.lm.embed_tokens(ids)
        bands = self._bands(groups, n_groups_total)
        stats = {"prefill_tokens": 0, "corrected_groups": 0, "chunks": [],
                 "group_idx": []}  # decode_start_pos added below
        if unified:
            # The cost script replays the vision half from these (it is no longer common across
            # arms): one record per approximate layer range and per band correction.
            stats["chunks"].append(("vapprox", 0, v_front, int(n_rows)))

        pos_3d, rope_delta = self._positions(inputs, image_run=(lo, n_tok), grid_thw=grid_thw)
        # Where a decode loop on top of the returned cache must continue from: stock advances all
        # three axes together for generated (text) tokens. rope_delta = max_pos + 1 - seq by
        # definition, so this is CPU arithmetic, not a read-back of pos_3d.
        stats_decode_pos = rope_delta + seq
        all_groups = torch.arange(n_groups_total, device=dev)
        rows_all = self._rows_of_groups(ctx_full, all_groups)   # group-major row order

        if keep < 1.0:
            # Per-merge-group score. Energy is pixel-level (the client hint in deployment): mean
            # squared residual between full and base patch rows, pooled to merge groups.
            # pixel_values rows are in patch_embed's native (group) order; the attention mean is
            # in the tower's row order and is gathered into group order through `rows_all`.
            resid = (px_full.float() - px_base.float()).pow(2).mean(dim=-1)      # [n_rows]
            energy = resid.reshape(n_groups_total, unit).mean(dim=1)
            energy = energy / energy.mean().clamp_min(1e-12)

            def attn_term(vec: torch.Tensor) -> torch.Tensor:
                vec = vec.to(energy.device)
                vec = vec[rows_all.to(vec.device)].reshape(n_groups_total, unit).mean(dim=1)
                return vec / vec.mean().clamp_min(1e-12)

            # Deferred: band 0 ranks on the energy hint; the attention term joins after push 0.
            # Unified: PROGRESSIVE -- band r ranks on the layers walked so far (recomputed at the
            # top of each round below), because the full-tower mean does not exist yet. That is a
            # different SIGNAL, not only a different schedule: at keep<1 the unified arm selects a
            # different set from the streaming / interleaved arms, so contract rule 5 says the
            # k<1 arms are not directly comparable across schedules (k=1 is: everything is
            # selected). Same trade gemma3's `interleaved_forward_progressive` makes, and the
            # alternative -- a full-depth scoring pass before round 0 -- runs the tower twice.
            if unified:
                score = energy * attn_term(self._attn_layermean_prefix(cache, v_front))
            else:
                score = energy if defer else energy * attn_term(self._attn_layermean(cache))
            n_sel = max(1, int(round(keep * n_groups_total)))
            quota = [n_sel // groups + (1 if r_ < n_sel % groups else 0) for r_ in range(groups)]
            selected = torch.zeros(n_groups_total, dtype=torch.bool, device=score.device)

        kv = None if sink is not None else DynamicCache(config=self.cfg.text_config)
        pos_done = 0
        last_logits = None

        def prefill(end: int):
            nonlocal pos_done, last_logits
            if end <= pos_done:
                return
            if sink is not None:
                # The chunk leaves the process: rows of the (partially corrected) embedding
                # sequence plus their M-RoPE positions; `final` closes the prompt on the server.
                sink.push(emb_all[0, pos_done:end],
                          pos_3d[:, 0, pos_done:end] if self.uses_mrope else None,
                          rope_delta if self.uses_mrope else None,
                          final=(end == seq))
            else:
                out = self.model(
                    inputs_embeds=emb_all[:, pos_done:end], past_key_values=kv,
                    position_ids=(pos_3d[:, :, pos_done:end] if self.uses_mrope else None),
                    use_cache=True)
                last_logits = out.logits[:, -1]
            stats["chunks"].append((pos_done, end))
            stats["prefill_tokens"] += end - pos_done
            pos_done = end

        sent_in_round = True     # arrival 0 counts as "spaced": band 0 is never delayed

        def open_prompt(x_stream):
            """Open the LLM with the WHOLE prompt, every image row at the merge of `x_stream`.

            The engine prefills it (hold-back: row seq-1 waits for `final`) while the next band
            is still being corrected; each band then rewrites its own rows in place. Merging in
            band-sized slices bounds the merger's working set to what the streaming schedule
            already gives it. The interleaved schedules call this at t=0 with the base pass's
            output; the unified axis calls it the moment its approximate frontier crosses out of
            the tower, so the bands already corrected go out CORRECTED and the rest approximate.
            """
            nonlocal sent_in_round
            sent_in_round = True
            with self._stage("merge"):
                for g0_, g1_ in bands:
                    if g1_ <= g0_:
                        continue
                    b_rows = self._rows_of_groups(ctx_full, torch.arange(g0_, g1_, device=dev))
                    emb_all[:, lo + g0_:lo + g1_] = \
                        self.tower.merger(x_stream[b_rows]).unsqueeze(0).to(emb_all.dtype)
            with self._stage("llm_prefill"):
                # clone: the push's device->host copy is ordered after this point on a side
                # stream, and the band loop overwrites these very rows -- without the
                # snapshot the approx prompt could carry rows corrected later, which is the
                # contract's rule-2 leak (data from the future) in wire form.
                sink.push(emb_all[0, :seq].clone(),
                          pos_3d[:, 0, :seq] if self.uses_mrope else None,
                          rope_delta if self.uses_mrope else None, final=False,
                          correct_from=lo,
                          correct_to=(lo + n_groups_total if staged else None),
                          **({"open_walk": int(llm_bounds[0])}
                             if unified and self.engine_open_walk else {}))
            stats["chunks"].append(("approx", 0, seq - 1))
            stats["prefill_tokens"] += seq - 1

        if interleaved and not unified:
            # t=0: the WHOLE approximate prompt, every image row at its base-resolution merge.
            with self._arrival(0):
                open_prompt(x_base_out)

        # Rows arrived so far -- the residual-stream restart mixes full rows (arrived) with base
        # rows (not yet), which is the in-process equivalent of the executor path's "reconstructed
        # canvas": the stream the correction restarts from is exactly what has been received.
        # With `correct_rows_only` the mixed stream is never materialised: band r corrects only
        # band r's groups (keep<1 selects among them), the bands are disjoint, so every corrected
        # row is an ARRIVED row and its layer-0 value is `ctx_full["hidden_states"][row]`; the
        # base rows of the mix were only ever carried, never read (see `block.correct_rows`).
        if not rows_only:
            arrived_rows = torch.zeros(n_rows, dtype=torch.bool, device=dev)
        last_arrival = 0
        last_band = max(r for r, (g0, g1) in enumerate(bands) if g1 > g0)
        pscore_pending = keep < 1.0 and defer
        # Bands whose message has already left. `pos_done > 0` used to stand in for this, but the
        # interleaved branch never advances `pos_done` (its prompt went out whole at t=0), so the
        # deferred score would never have completed there and the two schedules would have
        # SELECTED DIFFERENT GROUPS -- the interleaved contract's rule 5, and unobservable in any
        # gate that compares the two arms' embeddings alone.
        bands_done = 0
        if unified:
            # Only the rounds whose bound has crossed out of the tower send a `correct`: before
            # that the LLM has not been opened at all. So the engine sees FEWER rounds than
            # `groups`, at bounds that are neither `L(r+1)/g` nor equally spaced -- hence the
            # explicit-bounds form of the wire's `stage` field (memo §7.12).
            if any(g1_ <= g0_ for g0_, g1_ in bands):
                raise ValueError(f"unified_staged needs one non-empty band per round: groups="
                                 f"{groups} over {n_groups_total} merge groups")
            llm_rounds = [r_ for r_ in range(groups) if bounds[r_] > n_vis]
            assert llm_rounds and llm_rounds[-1] == groups - 1, (bounds, n_vis)
            llm_bounds = tuple(bounds[r_] - n_vis for r_ in llm_rounds)
            assert llm_bounds[-1] == n_llm, (llm_bounds, n_llm)
            stage_of = {r_: (j, len(llm_rounds), llm_bounds) for j, r_ in enumerate(llm_rounds)}
            stats["unified_bounds"] = list(bounds)
            stats["unified_llm_bounds"] = list(llm_bounds)
            stats["open_walk"] = int(llm_bounds[0]) if self.engine_open_walk else 0
            opened = bounds[0] > n_vis
            if opened:
                # The first bound already crosses the projector (always at groups=1, and on
                # prompts whose decoder half dominates): the tower has just been walked to full
                # depth on the base image, so this push IS the interleaved schedule's t=0 push
                # -- which is what makes groups=1 an identity against it.
                with self._arrival(0):
                    x_base_out = x_v
                    open_prompt(x_v)

        def advance(r_: int) -> None:
            """End of round r_: push the approximate frontier from bounds[r_] to bounds[r_+1].

            Tower stages first; the moment the next bound reaches past the last tower layer the
            axis crosses the projector -- the vision stream is merged AS IT STANDS (bands
            0..r_ corrected and carried, the rest approximate) and the whole prompt opens the
            LLM. After that the tower frontier is at `n_vis` and only corrections change it, so
            `x_base_out` (the merge's reference for rows a later band does NOT correct) is
            pinned to the crossing stream -- the unified analogue of the interleaved arm's
            base-resolution merge, and what keeps an uncorrected row's LLM input the value the
            opening push actually carried.
            """
            nonlocal x_v, v_front, opened, x_base_out, cache
            nxt = bounds[r_ + 1] if r_ + 1 < groups else bounds[-1]
            v_nxt = min(nxt, n_vis)
            if v_front < v_nxt:
                with self._stage("vision_approx"):
                    x_v, cache = self._approx_range(ctx_base, cache, v_front, v_nxt, x=x_v,
                                                    collect_attn=keep < 1.0)
                stats["chunks"].append(("vapprox", v_front, v_nxt, int(n_rows)))
                v_front = v_nxt
            if nxt > n_vis and not opened:
                x_base_out = x_v
                open_prompt(x_v)
                opened = True

        for r, (g0, g1) in enumerate(bands):
            if g1 <= g0:
                continue
            last_arrival = r + 1
            if unified and not sent_in_round and hasattr(sink, "band_gap"):
                # The previous round crossed nothing and sent nothing, so the bridge's
                # band-spacing sleep never ran: apply it here, or this band's pixels would
                # "arrive" the instant the last one was processed (memo §7.12, latency).
                sink.band_gap()
            sent_in_round = False
            if hasattr(sink, "band_start"):
                sink.band_start()
            with self._arrival(last_arrival):
                if pscore_pending and bands_done:
                    # First band is out: complete the score for the bands still to be selected.
                    # PSCORE stage = the FLOP counter's excluded scope (a bare column sum the
                    # hooks never saw anyway; the label keeps the split honest if that changes).
                    with self._stage("PSCORE"):
                        score = energy * attn_term(self._attn_layermean_deferred(cache, ctx_base))
                    pscore_pending = False
                if unified and keep < 1.0:
                    # Progressive: rank on every tower layer walked SO FAR. Costs nothing extra
                    # -- each layer's received-attention vector was collected inside its own
                    # approximate range (uncounted; see PSCORE above) -- and is the best signal
                    # that exists at the moment this band's groups must be chosen.
                    score = energy * attn_term(self._attn_layermean_prefix(cache, v_front))
                band_groups = torch.arange(g0, g1, device=dev)
                band_rows_idx = self._rows_of_groups(ctx_full, band_groups)
                # Unified, after the crossing: the tower frontier is at full depth for good and
                # this round's correct is full depth too -- the rows-only shortcut applies.
                rows_r = rows_only or (unified and opened and v_front == n_vis
                                       and self.correct_rows_only
                                       and self.unified_rows_after_crossing)
                if not rows_r:
                    arrived_rows[band_rows_idx] = True
                    stream = torch.where(arrived_rows.unsqueeze(-1),
                                         ctx_full["hidden_states"], ctx_base["hidden_states"])
                if keep < 1.0:
                    band_mask = torch.zeros(n_groups_total, dtype=torch.bool, device=score.device)
                    band_mask[g0:g1] = True
                    cand = band_mask & ~selected
                    # Bands are disjoint and `selected` only ever gains groups of the current
                    # band, so every group of this band is still a candidate: the count is
                    # g1 - g0 by construction (the former `int(cand.sum())` was a host sync
                    # returning exactly that).
                    q = min(quota[r], g1 - g0)
                    if q > 0:
                        group_idx = score.masked_fill(~cand, float("-inf")).topk(q).indices.sort().values
                        selected[group_idx] = True
                    else:
                        group_idx = torch.empty(0, dtype=torch.long, device=score.device)
                else:
                    group_idx = band_groups
                if group_idx.numel():
                    with self._stage("vision_correct"):
                        if rows_r:
                            # keep=1.0: the band IS the contiguous group range [g0, g1), so the
                            # tower can slice rows instead of gathering them (span, no sync).
                            unit = self.tower.spatial_merge_unit
                            x_rows, cache = self.tower.correct_rows(
                                ctx_full["hidden_states"], group_idx, ctx_full, cache, "v",
                                span=(g0 * unit, g1 * unit) if keep >= 1.0 else None)
                            if unified:
                                stats["chunks"].append(
                                    ("vcorrect", 0, n_vis, int(group_idx.numel()) * unit))
                        else:
                            # Unified: over the tower layers walked so far, not the full depth.
                            # `v_front` is this round's bound (bounds[r]) clamped to the tower --
                            # correcting deeper than the approximate pass has reached would read
                            # K/V that does not exist yet.
                            depth = v_front if unified else len(self.tower.blocks)
                            x_v, cache = self.tower.correct_forward(stream, group_idx, 0,
                                                                    depth, ctx_full,
                                                                    cache, "v")
                            if unified:
                                stats["chunks"].append(
                                    ("vcorrect", 0, depth, int(group_idx.numel()) * unit))
                stats["corrected_groups"] += int(group_idx.numel()) if keep < 1.0 else (g1 - g0)
                stats["group_idx"].append(group_idx)  # device tensors, no sync (gates read them)
                # Merge ONLY this band. The merger is per-merge-group (norm -> reshape(unit) ->
                # MLP), so slicing at group granularity is exact. Under keep<1, UNCORRECTED rows
                # take the PURE approx output -- the same convention gemma3's progressive walk
                # uses (mixed = corrected rows from the walk, everything else feats_appr), not the
                # stream+increment reconstruction, which mixes refined layer-0 with degraded
                # increments (the self-inconsistent combination the CLIP memo measured below floor).
                # Unified: nothing to merge or send before the LLM has been opened -- the rounds
                # up to the crossing are pure tower rounds and their corrections reach the LLM
                # through the opening push's merge of the stream as it then stands.
                if unified and not opened:
                    advance(r)
                    bands_done += 1
                    continue
                with self._stage("merge"):
                    if rows_r:
                        if keep < 1.0:
                            # Gather is a fresh tensor; the corrected rows (group-major in
                            # group_idx order) overwrite their groups' slots within the band.
                            band_rows = x_base_out[band_rows_idx]
                            if group_idx.numel():
                                band_rows.view(g1 - g0, unit, -1)[group_idx.to(dev) - g0] = \
                                    x_rows.view(-1, unit, x_rows.shape[-1])
                        else:
                            band_rows = x_rows
                    elif keep < 1.0:
                        row_mask = torch.zeros(n_rows, dtype=torch.bool, device=dev)
                        if group_idx.numel():
                            row_mask[self._rows_of_groups(ctx_full, group_idx.to(dev))] = True
                        src = torch.where(row_mask.unsqueeze(-1), x_v, x_base_out) \
                            if group_idx.numel() else x_base_out
                        band_rows = src[band_rows_idx]
                    else:
                        band_rows = x_v[band_rows_idx]
                    merged = self.tower.merger(band_rows)
                emb_all[:, lo + g0:lo + g1] = merged.unsqueeze(0).to(emb_all.dtype)
                with self._stage("llm_prefill"):
                    if interleaved:
                        # Round r rewrites ONLY this band's corrected rows (contract rule 1);
                        # rows this band did not select keep the approximate value the t=0 push
                        # already carried, and are simply absent from P_r. The last round also
                        # carries the text suffix -- it must see the fully corrected image -- and
                        # since [lo+g0, lo+G) and [lo+G, seq-1) are ADJACENT, one re-scan window
                        # [lo+g0, seq-1) covers both, so it is one message, not two.
                        pos_r = lo + group_idx.to(dev)
                        end = lo + g1
                        if r == last_band:
                            end = seq - 1        # row seq-1 is the hold-back, never rewritten
                            pos_r = torch.cat([pos_r, torch.arange(
                                lo + n_groups_total, end, device=dev, dtype=pos_r.dtype)])
                        if r == last_band and pos_r.numel() == 0:
                            raise RuntimeError(
                                "interleaved: the final round corrects nothing and carries no "
                                "text suffix -- nothing would release the held-back last row")
                        if pos_r.numel():
                            # Gathered from emb_all, which this band's merge has just been
                            # written into: bitwise the rows the streaming schedule pushes for
                            # the same positions (gate G5).
                            # staged: rounds are the non-empty bands 0..last_band; a band
                            # that selected no row is simply not sent (the engine's next
                            # round walks the frontier over the skipped layer range).
                            # unified: the LLM rounds are only the bands after the crossing,
                            # and their bounds come from the cost split, so the round index,
                            # the round COUNT and the bounds all go on the wire explicitly.
                            rec = ("correct", lo + g0, end, int(pos_r.numel()))
                            if unified:
                                stage = stage_of[r]
                                rec = rec + (stage[0], stage[1], int(stage[2][stage[0]]))
                            elif staged:
                                stage = (r, last_band + 1)
                                rec = rec + stage
                            else:
                                stage = None
                            sink.correct(pos_r, emb_all[0, pos_r], (lo + g0, end),
                                         final=(r == last_band), stage=stage)
                            sent_in_round = True
                            stats["chunks"].append(rec)
                            stats["prefill_tokens"] += int(pos_r.numel())
                    else:
                        # r=0 also carries the leading text; the last band carries the trailing
                        # text (question + generation prompt) in the SAME chunk: it waits on the
                        # last band anyway (same arrival -- charging it later would invent an
                        # arrival the transmission never had), and one prefill of image tail +
                        # text is one engine step instead of two (2026-09-09, was a separate
                        # trailing push).
                        prefill(seq if r == last_band else lo + g1)
                if unified:
                    advance(r)
            bands_done += 1
        if interleaved:
            assert pos_done == 0, "the streaming prefill closure ran in the interleaved branch"
        else:
            assert pos_done == seq, (pos_done, seq)
        if unified:
            assert opened and v_front == n_vis, (opened, v_front, n_vis)
        stats["llm_schedule"] = llm_schedule
        stats["decode_start_pos"] = stats_decode_pos
        stats["rope_delta"] = rope_delta
        if keep < 1.0:
            stats["pscore"] = "progressive" if unified else ("deferred" if defer else "eager")
        # The image-row embeddings the LLM actually consumed, for feature-space gating. Task
        # metrics are not monotone in fidelity (the interleaved contract says this in as many
        # words), and Qwen3.5's first generated token is CoT boilerplate that ignores the image
        # entirely -- measured TV(floor, stock) = 0.0005 at that position, i.e. no logit-level gate
        # can see the vision mechanism at all. The embeddings can. Skipped on the sink path
        # unless asked for: the chunks already left with these rows, and the fp32 copy is 250 MB
        # at 15k tokens.
        if sink is None or self.image_embeds_with_sink:
            stats["image_embeds"] = emb_all[:, lo:lo + n_groups_total].float()
        return last_logits, kv, stats

    # --- the floor ------------------------------------------------------------------------------ #

    @torch.no_grad()
    def approx_only_forward(self, inputs: Dict[str, Any], px_base: torch.Tensor) -> torch.Tensor:
        """The floor: the degraded base image through the STOCK path, one-shot. 100% critical."""
        with self._arrival(0), self._stage("floor"):
            out = self.model(input_ids=inputs["input_ids"],
                             pixel_values=px_base.to(self.model.dtype),
                             image_grid_thw=inputs["image_grid_thw"],
                             mm_token_type_ids=inputs["mm_token_type_ids"], use_cache=False)
        return out.logits[:, -1]

    # --- one-shot prompt embeddings (floor / ceiling through the vLLM server) ------------------- #

    @torch.no_grad()
    def oneshot_embeds(self, inputs: Dict[str, Any], pixel_values: torch.Tensor,
                       stage: str = "full", image_run: Optional[Tuple[int, int]] = None,
                       grid_thw: Optional[Tuple[int, int, int]] = None,
                       ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """The stock path's prompt embeddings for ONE image: text rows from `embed_tokens`, image
        rows from the STOCK vision tower on `pixel_values` (the full image for the ceiling, the
        degraded base for the floor), spliced at the image run -- the same `inputs_embeds` stock's
        forward builds internally before its first decoder layer. Returns (embeds [T, D],
        mrope positions [3, T], rope_delta), i.e. one chunk for `StreamSink.push(final=True)`."""
        ids = inputs["input_ids"]
        lo, n_tok = image_run if image_run is not None else self._image_token_run(ids)
        with self._arrival(0), self._stage(stage):
            feats = self.model.model.get_image_features(
                pixel_values.to(self.model.dtype), inputs["image_grid_thw"])
            feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
            if isinstance(feats, (list, tuple)):
                feats = torch.cat(list(feats), dim=0)
            if feats.shape[0] != n_tok:
                raise ValueError(f"tower produced {feats.shape[0]} rows for {n_tok} image tokens")
            emb = self.lm.embed_tokens(ids)[0]
            emb[lo:lo + n_tok] = feats.to(emb.dtype)
        pos_3d, rope_delta = self._positions(inputs, image_run=(lo, n_tok), grid_thw=grid_thw)
        if not self.uses_mrope:
            return emb, None, None
        return emb, pos_3d[:, 0], rope_delta
