from typing import Any, List, Optional

from offload.common.protocol import ExperimentConfig, Instruction, OpType, Patch, Task

from ..interface import ISchedulingPolicy


class _COCOWindowSchedulingMixin:
    _NUM_WINDOW_GROUPS = 9

    @staticmethod
    def _total_layers(config: ExperimentConfig) -> int:
        return int(
            config.scheduler_kwargs.get(
                'total_layers',
                config.transmission_kwargs.get('total_layers', 40),
            )
        )

    @classmethod
    def _num_window_groups(cls, config: ExperimentConfig) -> int:
        return int(
            config.scheduler_kwargs.get(
                'num_window_groups',
                config.transmission_kwargs.get('num_window_groups', cls._NUM_WINDOW_GROUPS),
            )
        )

    @staticmethod
    def _global_source_mode(config: ExperimentConfig) -> str:
        appcorr_kwargs = getattr(config, 'appcorr_kwargs', {}) or {}
        mode = str(appcorr_kwargs.get('global_source_mode', 'global_first'))
        if mode not in {'global_first', 'final_correct', 'approx'}:
            return 'global_first'
        return mode

    @classmethod
    def _needs_global_first(cls, config: ExperimentConfig) -> bool:
        appcorr_kwargs = getattr(config, 'appcorr_kwargs', {}) or {}
        return (
            getattr(config, 'model_name', None) == 'dinov3_detector'
            and bool(appcorr_kwargs.get('generated_from_client', False))
            and cls._global_source_mode(config) == 'global_first'
        )

    @classmethod
    def _needs_final_global(cls, config: ExperimentConfig) -> bool:
        appcorr_kwargs = getattr(config, 'appcorr_kwargs', {}) or {}
        return (
            getattr(config, 'model_name', None) == 'dinov3_detector'
            and bool(appcorr_kwargs.get('generated_from_client', False))
            and cls._global_source_mode(config) == 'final_correct'
        )

    @classmethod
    def _append_global_first(cls, instructions: List[Instruction], config: ExperimentConfig) -> None:
        if cls._needs_global_first(config):
            instructions.append(
                Instruction(
                    OpType.APPROX_FORWARD,
                    {
                        'layers': (0, cls._total_layers(config)),
                        'global_only': True,
                        'source_kind': 'global',
                    },
                )
            )

    @classmethod
    def _append_final_global(cls, instructions: List[Instruction], config: ExperimentConfig) -> None:
        if cls._needs_final_global(config):
            instructions.append(
                Instruction(
                    OpType.APPROX_FORWARD,
                    {
                        'layers': (0, cls._total_layers(config)),
                        'global_only': True,
                        'source_kind': 'global',
                    },
                )
            )

    @classmethod
    def _append_finish(cls, instructions: List[Instruction]) -> None:
        instructions.append(Instruction(OpType.HEAD_INFERENCE))
        instructions.append(Instruction(OpType.EXIT_ALL))
        instructions.append(Instruction(OpType.SEND_RESPONSE))
        instructions.append(Instruction(OpType.FREE_SESSION))


class COCOWindowInterleavedPolicy(_COCOWindowSchedulingMixin, ISchedulingPolicy):
    """
    Static COCO detector appcorr schedule.

    The local backbone is advanced by a fixed layer chunk between row-major
    detector-window residual corrections.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.current_request_id = None
        self.latest_approx_layer_queued = 0

    def decide(
        self,
        buffer: List[Patch],
        config: ExperimentConfig,
        task_id_gen: Any,
        **kwargs,
    ) -> Optional[Task]:
        if not buffer:
            return None

        head_patch = buffer[0]
        current_group = head_patch.group_id
        target_count = head_patch.batch_group_total
        if len(buffer) < target_count:
            return None

        task_id = next(task_id_gen)
        if current_group == 0 or self.current_request_id is None:
            self.current_request_id = task_id
            self.latest_approx_layer_queued = 0

        payload = buffer[:target_count]
        instructions = self._get_pipeline_instructions(current_group, config)
        return Task(
            task_id=task_id,
            request_id=self.current_request_id,
            payload=payload,
            instructions=instructions,
        )

    @classmethod
    def _layer_boundaries(cls, config: ExperimentConfig) -> List[int]:
        total_layers = cls._total_layers(config)
        num_groups = cls._num_window_groups(config)
        base = total_layers // num_groups
        remainder = total_layers % num_groups

        boundaries = [0]
        cursor = 0
        for idx in range(num_groups):
            cursor += base + (1 if idx < remainder else 0)
            boundaries.append(cursor)
        boundaries[-1] = total_layers
        return boundaries

    def _append_next_static_approx(self, instructions: List[Instruction], config: ExperimentConfig, group_id: int) -> None:
        boundaries = self._layer_boundaries(config)
        num_groups = self._num_window_groups(config)
        next_idx = min(group_id + 1, num_groups)
        next_end = boundaries[next_idx]
        if self.latest_approx_layer_queued < next_end:
            instructions.append(
                Instruction(
                    OpType.APPROX_FORWARD,
                    {
                        'layers': (self.latest_approx_layer_queued, next_end),
                        'source_kind': 'local',
                    },
                )
            )
            self.latest_approx_layer_queued = next_end

    def _get_pipeline_instructions(self, group_id: int, config: ExperimentConfig) -> List[Instruction]:
        total_layers = self._total_layers(config)
        num_groups = self._num_window_groups(config)
        instructions = [Instruction(OpType.LOAD_INPUT), Instruction(OpType.PREPARE_TOKENS)]

        if group_id == 0:
            self._append_global_first(instructions, config)
            self._append_next_static_approx(instructions, config, 0)
            return instructions

        if group_id < num_groups:
            if self.latest_approx_layer_queued > 0:
                instructions.append(
                    Instruction(
                        OpType.CORRECT_FORWARD,
                        {
                            'layers': (0, self.latest_approx_layer_queued),
                            'group_id': group_id,
                        },
                    )
                )
            self._append_next_static_approx(instructions, config, group_id)
            return instructions

        if self.latest_approx_layer_queued > 0:
            instructions.append(
                Instruction(
                    OpType.CORRECT_FORWARD,
                    {
                        'layers': (0, self.latest_approx_layer_queued),
                        'group_id': group_id,
                    },
                )
            )
        if self.latest_approx_layer_queued < total_layers:
            instructions.append(
                Instruction(
                    OpType.APPROX_FORWARD,
                    {
                        'layers': (self.latest_approx_layer_queued, total_layers),
                        'source_kind': 'local',
                    },
                )
            )
            self.latest_approx_layer_queued = total_layers
        self._append_final_global(instructions, config)
        self._append_finish(instructions)
        return instructions


class COCOWindowDynamicPolicy(_COCOWindowSchedulingMixin, ISchedulingPolicy):
    """
    Greedy COCO detector appcorr schedule.

    Approximation advances while residual windows are still in flight. When a
    window residual group arrives, that window is corrected up to the latest
    queued approximate layer.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.current_request_id = None
        self.latest_approx_layer_queued = 0
        self.current_group_id = -1
        self.ahead_layers = 2
        if config is not None:
            self.ahead_layers = int(
                config.scheduler_kwargs.get(
                    'ahead_layers',
                    config.transmission_kwargs.get('ahead_layers', self.ahead_layers),
                )
            )

    def decide(
        self,
        buffer: List[Patch],
        config: ExperimentConfig,
        task_id_gen: Any,
        **kwargs,
    ) -> Optional[Task]:
        feedback_events = kwargs.get('feedback_events', [])
        max_completed_layer = max(feedback_events) if feedback_events else None

        if buffer:
            head_patch = buffer[0]
            current_group = head_patch.group_id
            target_count = head_patch.batch_group_total
            if len(buffer) >= target_count:
                task_id = next(task_id_gen)
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
        num_groups = self._num_window_groups(config)
        if self.current_group_id == -1 or self.current_group_id >= num_groups:
            return

        diff = self.latest_approx_layer_queued - max_completed_layer
        while diff < self.ahead_layers and self.latest_approx_layer_queued < total_layers:
            instructions.append(
                Instruction(
                    OpType.APPROX_FORWARD,
                    {
                        'layers': (self.latest_approx_layer_queued, self.latest_approx_layer_queued + 1),
                        'source_kind': 'local',
                    },
                )
            )
            self.latest_approx_layer_queued += 1
            diff += 1

    def _get_payload_instructions(
        self,
        group_id: int,
        config: ExperimentConfig,
        max_completed_layer: int | None = None,
    ) -> List[Instruction]:
        total_layers = self._total_layers(config)
        num_groups = self._num_window_groups(config)
        instructions = [Instruction(OpType.LOAD_INPUT), Instruction(OpType.PREPARE_TOKENS)]

        if group_id == 0:
            self._append_global_first(instructions, config)
            self._append_more_approx(instructions, config, max_completed_layer or 0)
            return instructions

        if group_id < num_groups:
            if self.latest_approx_layer_queued > 0:
                instructions.append(
                    Instruction(
                        OpType.CORRECT_FORWARD,
                        {
                            'layers': (0, self.latest_approx_layer_queued),
                            'group_id': group_id,
                        },
                    )
                )
            if max_completed_layer is not None:
                self._append_more_approx(instructions, config, max_completed_layer)
            return instructions

        if self.latest_approx_layer_queued > 0:
            instructions.append(
                Instruction(
                    OpType.CORRECT_FORWARD,
                    {
                        'layers': (0, self.latest_approx_layer_queued),
                        'group_id': group_id,
                    },
                )
            )
        if self.latest_approx_layer_queued < total_layers:
            instructions.append(
                Instruction(
                    OpType.APPROX_FORWARD,
                    {
                        'layers': (self.latest_approx_layer_queued, total_layers),
                        'source_kind': 'local',
                    },
                )
            )
            self.latest_approx_layer_queued = total_layers

        self._append_final_global(instructions, config)
        self._append_finish(instructions)
        return instructions
