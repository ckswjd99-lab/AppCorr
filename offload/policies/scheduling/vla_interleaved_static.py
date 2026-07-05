"""
vla_interleaved_static.py

Static scheduling for the OpenVLA progressive-prefill executor. One policy class, three schedules
(selected by scheduler_kwargs['schedule']) so all baselines share the same instruction/monitoring
path:

  interleaved (default): the static counterpart of ADE20KInterleavedDynamicPolicy. LLM approximation
      advances to a fixed frontier per group; each arriving residual group g is corrected only
      through LLM layers (0, frontier_g); layers above the frontier absorb the correction as part
      of their single normal approx execution. Frontiers default to uniform spacing:
      f_g = (g-1) * total_layers / num_groups (group 1 arrives before any LLM layer ran -> its
      correction enters purely via PREPARE_TOKENS' x0 rebuild).

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
        return [round((g - 1) * L / G) for g in range(1, G + 1)]

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
