"""
vla_interleaved_static.py

Static scheduling for the OpenVLA progressive-prefill executor. One policy class, three schedules
(selected by scheduler_kwargs['schedule']) so all baselines share the same instruction/monitoring
path:

  interleaved (default): the static counterpart of ADE20KInterleavedDynamicPolicy. Group 0 (the
      base layer) immediately advances the LLM approx frontier to f_1 = total_layers / num_groups
      -- mirroring ADE20KInterleavedDynamicPolicy's group 0 handler, which starts queuing
      layer-by-layer approx progress right away rather than waiting for the first residual group.
      Each subsequently-arriving residual group g (1..num_groups) is corrected through the CURRENT
      frontier (0, f_g) -- reached by the PRECEDING step -- then the frontier advances to
      f_{g+1} = g * total_layers / num_groups (or stays at total_layers for the last group, which
      needs no further approx). Frontiers default to uniform spacing: f_g = g * total_layers /
      num_groups for g=1..num_groups. Every residual group gets a real CORRECT_FORWARD this way --
      an earlier version of this policy left group 0 at frontier 0, which meant group 1 always
      found nothing yet approximated to correct (a bug, not intended AppCorr semantics; fixed by
      removing group 0's special-cased "stay at zero" frontier).

  sequential: classic approx-then-correct (ADE20KApproxCorrectPolicy shape): full approx on the
      base layer, residual groups only accumulate into the canvas, one full-depth CORRECT at the
      end covering everything that arrived.

  full: no approximation -- accumulate all groups, then run the stock full inference
      (FULL_INFERENCE) on the final canvas. Baseline with identical monitoring.
"""

from typing import Any, List, Optional

from offload.common.protocol import ExperimentConfig, Instruction, OpType, Patch, Task

from ..interface import ISchedulingPolicy


class VLAInterleavedStaticPolicy(ISchedulingPolicy):
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.current_request_id = None
        self.frontier = 0  # LLM layers approximated so far

    @staticmethod
    def _sched(config: ExperimentConfig) -> str:
        return str(config.scheduler_kwargs.get("schedule", "interleaved"))

    @staticmethod
    def _total_layers(config: ExperimentConfig) -> int:
        return int(config.scheduler_kwargs.get("total_layers", 32))

    @staticmethod
    def _num_groups(config: ExperimentConfig) -> int:
        return max(int(config.transmission_kwargs.get("num_groups", 4)), 1)

    def _frontiers(self, config: ExperimentConfig) -> List[int]:
        explicit = config.scheduler_kwargs.get("frontiers")
        if explicit:
            return [int(f) for f in explicit]
        G, L = self._num_groups(config), self._total_layers(config)
        return [round(g * L / G) for g in range(1, G + 1)]

    @staticmethod
    def _finish(instructions: List[Instruction]) -> None:
        instructions.append(Instruction(OpType.HEAD_INFERENCE))
        instructions.append(Instruction(OpType.SEND_RESPONSE))
        instructions.append(Instruction(OpType.FREE_SESSION))

    def decide(self, buffer: List[Patch], config: ExperimentConfig, task_id_gen: Any, **kwargs) -> Optional[Task]:
        if not buffer:
            return None
        head = buffer[0]
        gid, target = int(head.group_id), int(head.batch_group_total)
        if len(buffer) < target:
            return None

        task_id = next(task_id_gen)
        if gid == 0 or self.current_request_id is None:
            self.current_request_id = task_id
            self.frontier = 0
        payload = buffer[:target]

        sched = self._sched(config)
        G = self._num_groups(config)
        L = self._total_layers(config)
        frontiers = self._frontiers(config)
        is_last = gid >= G
        instructions: List[Instruction] = [Instruction(OpType.LOAD_INPUT)]

        if sched == "full":
            if is_last:
                # No HEAD_INFERENCE: FULL_INFERENCE runs the stock predict_action() directly.
                instructions.append(Instruction(OpType.FULL_INFERENCE))
                instructions.append(Instruction(OpType.SEND_RESPONSE))
                instructions.append(Instruction(OpType.FREE_SESSION))

        elif sched == "approx":
            # Approx-only baseline: base layer gets a full-depth approx prefill; residual groups
            # are ignored (canvas accumulates but is never corrected); decode from the approx state
            # at the last group so the pipeline still yields exactly one InferenceResult per request.
            if gid == 0:
                instructions.append(Instruction(OpType.PREPARE_TOKENS))
                instructions.append(Instruction(OpType.APPROX_FORWARD, {"layers": (0, L)}))
                self.frontier = L
            elif is_last:
                self._finish(instructions)
            # middle groups: LOAD only (ignored).

        elif sched == "chunked":
            # True chunked causal prefill: base does vision approx + LLM cache init + BOS prefill
            # (in PREPARE_TOKENS); each residual group prefills only its own vision-token positions
            # once, and the last group additionally prefills the text suffix, then decodes. No LLM
            # approx pass, no per-group text re-correction.
            instructions.append(Instruction(OpType.PREPARE_TOKENS))
            if gid > 0:
                instructions.append(
                    Instruction(OpType.CORRECT_FORWARD,
                                {"layers": (0, L), "group_id": gid, "include_text": is_last})
                )
                if is_last:
                    self._finish(instructions)

        elif sched == "sequential":
            if gid == 0:
                instructions.append(Instruction(OpType.PREPARE_TOKENS))
                instructions.append(Instruction(OpType.APPROX_FORWARD, {"layers": (0, L)}))
                self.frontier = L
            elif is_last:
                instructions.append(Instruction(OpType.PREPARE_TOKENS))
                instructions.append(Instruction(OpType.CORRECT_FORWARD, {"layers": (0, L), "group_id": gid}))
                self._finish(instructions)
            # middle groups: LOAD only (canvas accumulates; corrected at the end)

        else:  # interleaved (static frontiers)
            if gid == 0:
                instructions.append(Instruction(OpType.PREPARE_TOKENS))
                first_frontier = frontiers[0] if frontiers else 0
                if first_frontier > 0:
                    instructions.append(Instruction(OpType.APPROX_FORWARD, {"layers": (0, first_frontier)}))
                    self.frontier = first_frontier
            else:
                instructions.append(Instruction(OpType.PREPARE_TOKENS))
                if self.frontier > 0:
                    instructions.append(
                        Instruction(OpType.CORRECT_FORWARD, {"layers": (0, self.frontier), "group_id": gid})
                    )
                next_frontier = L if is_last else frontiers[gid] if gid < len(frontiers) else L
                if next_frontier > self.frontier:
                    instructions.append(
                        Instruction(OpType.APPROX_FORWARD, {"layers": (self.frontier, next_frontier)})
                    )
                    self.frontier = next_frontier
                if is_last:
                    self._finish(instructions)

        return Task(
            task_id=task_id,
            request_id=self.current_request_id,
            payload=payload,
            instructions=instructions,
        )
