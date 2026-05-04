from typing import List, Optional, Any

from offload.common.protocol import Patch, Task, ExperimentConfig, Instruction, OpType, normalize_appcorr_kwargs
from ..interface import ISchedulingPolicy


class NYUApproxCorrectPolicy(ISchedulingPolicy):
    """NYU depth estimation scheduling with approx-then-correct.

    Phase 1: Base layer (group 0) arrives → LOAD → PREPARE → APPROX(0, 40)
    Phase 2: Residual (group 1) arrives → CORRECT(0, 40) → HEAD → SEND_RESPONSE

    When appcorr is disabled, skips CORRECT in phase 2.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self._num_res_groups = None
        self._current_request_id = None
        self._has_current_request = False
        if config:
            self._num_res_groups = config.transmission_kwargs.get('num_groups', 1)

    def decide(
        self,
        buffer: List[Patch],
        config: ExperimentConfig,
        task_id_gen: Any,
        **kwargs,
    ) -> Optional[Task]:

        if self._num_res_groups is None:
            self._num_res_groups = config.transmission_kwargs.get('num_groups', 1)

        if not buffer:
            return None

        total_layers = int(
            config.scheduler_kwargs.get(
                "total_layers",
                config.transmission_kwargs.get("total_layers", 40),
            )
        )

        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        appcorr_enabled = appcorr_options.get("enabled", False) or appcorr_options.get("generated_from_client", False)
        head_patch = buffer[0]
        current_group = head_patch.group_id
        target_count = head_patch.batch_group_total

        if len(buffer) < target_count:
            return None

        current_batch_patches = buffer[:target_count]
        del buffer[:target_count]

        t_id = next(task_id_gen)

        if current_group == 0:
            self._current_request_id = t_id
            self._has_current_request = True
            instructions = [
                Instruction(OpType.LOAD_INPUT),
                Instruction(OpType.PREPARE_TOKENS),
                Instruction(OpType.APPROX_FORWARD, {"layers": (0, total_layers)}),
            ]
            return Task(
                task_id=t_id,
                request_id=t_id,
                payload=current_batch_patches,
                instructions=instructions,
            )
        else:
            instructions = [
                Instruction(OpType.LOAD_INPUT),
                Instruction(OpType.PREPARE_TOKENS),
            ]
            if appcorr_enabled:
                instructions.append(
                    Instruction(OpType.CORRECT_FORWARD, {"layers": (0, total_layers), "group_id": 0})
                )
            instructions.extend([
                Instruction(OpType.HEAD_INFERENCE),
                Instruction(OpType.SEND_RESPONSE),
                Instruction(OpType.FREE_SESSION),
            ])
            return Task(
                task_id=t_id,
                request_id=self._current_request_id if self._has_current_request else t_id,
                payload=current_batch_patches,
                instructions=instructions,
            )
