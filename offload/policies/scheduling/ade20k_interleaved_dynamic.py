from typing import Any, List, Optional

from offload.common.protocol import ExperimentConfig, Instruction, OpType, Patch, Task

from ..interface import ISchedulingPolicy


class ADE20KInterleavedDynamicPolicy(ISchedulingPolicy):
    """
    Dynamic ADE20K interleaved appcorr schedule.

    Approximation advances layer-by-layer while residual groups are still in
    flight. When a residual group arrives, that group is corrected up to the
    latest queued approximate layer, mirroring the COCO dynamic window policy.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.current_request_id = None
        self.latest_approx_layer_queued = 0
        self.current_group_id = -1
        self.ahead_layers = 2
        # N correction groups: fixed from config for grid/block_grid; overridden per
        # image by patch.num_correction_groups for crop_cover (variable crop count).
        self.num_groups = self._num_groups(config) if config is not None else 4
        if config is not None:
            self.ahead_layers = int(
                config.scheduler_kwargs.get(
                    "ahead_layers",
                    config.transmission_kwargs.get("ahead_layers", self.ahead_layers),
                )
            )

    @staticmethod
    def _total_layers(config: ExperimentConfig) -> int:
        return int(
            config.scheduler_kwargs.get(
                "total_layers",
                config.transmission_kwargs.get("total_layers", 40),
            )
        )

    @staticmethod
    def _num_groups(config: ExperimentConfig) -> int:
        return int(
            config.scheduler_kwargs.get(
                "num_groups",
                config.transmission_kwargs.get("num_groups", 4),
            )
        )

    @staticmethod
    def _append_finish(instructions: List[Instruction]) -> None:
        instructions.append(Instruction(OpType.HEAD_INFERENCE))
        instructions.append(Instruction(OpType.EXIT_ALL))
        instructions.append(Instruction(OpType.SEND_RESPONSE))
        instructions.append(Instruction(OpType.FREE_SESSION))

    def decide(
        self,
        buffer: List[Patch],
        config: ExperimentConfig,
        task_id_gen: Any,
        **kwargs,
    ) -> Optional[Task]:
        feedback_events = kwargs.get("feedback_events", [])
        max_completed_layer = max(feedback_events) if feedback_events else None

        if buffer:
            head_patch = buffer[0]
            current_group = int(head_patch.group_id)
            target_count = int(head_patch.batch_group_total)
            if len(buffer) >= target_count:
                task_id = next(task_id_gen)
                n = int(getattr(head_patch, "num_correction_groups", 0) or 0)
                if n > 0:
                    self.num_groups = n
                if current_group == 0 or self.current_request_id is None:
                    self.current_request_id = task_id
                    self.latest_approx_layer_queued = 0
                    self.current_group_id = 0
                else:
                    self.current_group_id = current_group

                payload = buffer[:target_count]
                instructions = self._get_payload_instructions(
                    current_group,
                    config,
                    max_completed_layer=max_completed_layer,
                )
                return Task(
                    task_id=task_id,
                    request_id=self.current_request_id,
                    payload=payload,
                    instructions=instructions,
                )

        if max_completed_layer is not None:
            instructions: List[Instruction] = []
            self._append_more_approx(instructions, config, max_completed_layer)
            if instructions:
                task_id = next(task_id_gen)
                return Task(
                    task_id=task_id,
                    request_id=self.current_request_id,
                    payload=[],
                    instructions=instructions,
                )

        return None

    def _append_more_approx(
        self,
        instructions: List[Instruction],
        config: ExperimentConfig,
        max_completed_layer: int,
    ) -> None:
        total_layers = self._total_layers(config)
        num_groups = self.num_groups
        if self.current_group_id == -1 or self.current_group_id >= num_groups:
            return

        diff = self.latest_approx_layer_queued - int(max_completed_layer)
        while diff < self.ahead_layers and self.latest_approx_layer_queued < total_layers:
            start_l = self.latest_approx_layer_queued
            end_l = start_l + 1
            instructions.append(
                Instruction(
                    OpType.APPROX_FORWARD,
                    {
                        "layers": (start_l, end_l),
                        "source_kind": "local",
                    },
                )
            )
            self.latest_approx_layer_queued = end_l
            diff += 1

    def _get_payload_instructions(
        self,
        group_id: int,
        config: ExperimentConfig,
        max_completed_layer: int | None = None,
    ) -> List[Instruction]:
        total_layers = self._total_layers(config)
        num_groups = self.num_groups
        instructions = [Instruction(OpType.LOAD_INPUT), Instruction(OpType.PREPARE_TOKENS)]

        if group_id == 0:
            self._append_more_approx(instructions, config, max_completed_layer or 0)
            return instructions

        if group_id < num_groups:
            if self.latest_approx_layer_queued > 0:
                instructions.append(
                    Instruction(
                        OpType.CORRECT_FORWARD,
                        {
                            "layers": (0, self.latest_approx_layer_queued),
                            "group_id": group_id,
                        },
                    )
                )
            if max_completed_layer is not None:
                self._append_more_approx(instructions, config, max_completed_layer)
            if config.early_exit_enabled():
                instructions.append(Instruction(OpType.HEAD_INFERENCE))
                instructions.append(Instruction(OpType.DECIDE_EXIT))
            return instructions

        if self.latest_approx_layer_queued > 0:
            instructions.append(
                Instruction(
                    OpType.CORRECT_FORWARD,
                    {
                        "layers": (0, self.latest_approx_layer_queued),
                        "group_id": group_id,
                    },
                )
            )

        if self.latest_approx_layer_queued < total_layers:
            instructions.append(
                Instruction(
                    OpType.APPROX_FORWARD,
                    {
                        "layers": (self.latest_approx_layer_queued, total_layers),
                        "source_kind": "local",
                    },
                )
            )
            self.latest_approx_layer_queued = total_layers

        self._append_finish(instructions)
        return instructions
