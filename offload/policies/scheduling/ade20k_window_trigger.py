from typing import Any, List, Optional

from offload.common.protocol import ExperimentConfig, Instruction, OpType, Task

from ..interface import ISchedulingPolicy


class ADE20KWindowInterleavedPolicy(ISchedulingPolicy):
    """
    ADE20K m2f crop-cover interleaved schedule.

    The number of correction groups N = per-image sliding-crop count, read from
    `patch.num_correction_groups` (set by ADE20KWindowProgressiveLaplacianPolicy). Layers are split
    into N even chunks; group g is corrected up to the current approx frontier, then the backbone is
    advanced by one chunk. The final residual group (g == N) corrects the whole model and runs the
    head. Structurally identical to GroupTrigger, but N is dynamic per image instead of a fixed config
    value.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.current_request_id = None
        self.num_groups = 1

    @staticmethod
    def _total_layers(config: ExperimentConfig) -> int:
        return int(
            config.scheduler_kwargs.get(
                "total_layers",
                config.transmission_kwargs.get("total_layers", 40),
            )
        )

    @classmethod
    def _layer_boundaries(cls, config: ExperimentConfig, num_groups: int) -> List[int]:
        total = cls._total_layers(config)
        n = max(int(num_groups), 1)
        base = total // n
        remainder = total % n
        boundaries = [0]
        cursor = 0
        for idx in range(n):
            cursor += base + (1 if idx < remainder else 0)
            boundaries.append(cursor)
        boundaries[-1] = total
        return boundaries

    def decide(self, buffer, config: ExperimentConfig, task_id_gen: Any, **kwargs) -> Optional[Task]:
        if not buffer:
            return None
        head = buffer[0]
        current_group = head.group_id
        target_count = head.batch_group_total
        if len(buffer) < target_count:
            return None

        task_id = next(task_id_gen)
        if current_group == 0 or self.current_request_id is None:
            self.current_request_id = task_id
            self.num_groups = 1

        n = int(getattr(head, "num_correction_groups", 0) or 0)
        if n > 0:
            self.num_groups = n

        payload = buffer[:target_count]
        instructions = self._get_pipeline_instructions(current_group, config)
        return Task(
            task_id=task_id,
            request_id=self.current_request_id,
            payload=payload,
            instructions=instructions,
        )

    def _get_pipeline_instructions(self, group_id: int, config: ExperimentConfig) -> List[Instruction]:
        total = self._total_layers(config)
        n = max(self.num_groups, 1)
        boundaries = self._layer_boundaries(config, n)

        instructions = [Instruction(OpType.LOAD_INPUT), Instruction(OpType.PREPARE_TOKENS)]

        if group_id < n:
            chunk_start = boundaries[group_id]
            chunk_end = boundaries[group_id + 1]
            if group_id > 0 and chunk_start > 0:
                instructions.append(
                    Instruction(OpType.CORRECT_FORWARD, {"layers": (0, chunk_start), "group_id": group_id})
                )
            instructions.append(Instruction(OpType.APPROX_FORWARD, {"layers": (chunk_start, chunk_end)}))
            return instructions

        # Final residual group (group_id == n): correct the whole model, then head.
        instructions.append(Instruction(OpType.CORRECT_FORWARD, {"layers": (0, total), "group_id": group_id}))
        instructions.append(Instruction(OpType.HEAD_INFERENCE))
        instructions.append(Instruction(OpType.EXIT_ALL))
        instructions.append(Instruction(OpType.SEND_RESPONSE))
        instructions.append(Instruction(OpType.FREE_SESSION))
        return instructions
