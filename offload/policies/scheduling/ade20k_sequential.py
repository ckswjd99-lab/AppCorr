from typing import List, Optional, Any

from offload.common.protocol import Patch, Task, ExperimentConfig, Instruction, OpType
from ..interface import ISchedulingPolicy


class ADE20KSequentialPolicy(ISchedulingPolicy):
    """Sequential ADE20K segmentor scheduling.

    Uses the decomposed pipeline (prepare_tokens -> approx_forward -> head_inference)
    instead of full_inference.  Designed for testing the decomposed segmentor path.
    """

    def __init__(self, config=None):
        self._num_groups = None
        if config:
            self._num_groups = config.transmission_kwargs.get('num_groups', None)

    def decide(
        self,
        buffer: List[Patch],
        config: ExperimentConfig,
        task_id_gen: Any,
        **kwargs,
    ) -> Optional[Task]:

        if not buffer:
            return None

        if self._num_groups is None:
            self._num_groups = config.transmission_kwargs.get('num_groups', None)

        if config.transmission_policy_name == "ProgressiveLaplacian":
            expected_groups = (self._num_groups or 1) + 1
        else:
            expected_groups = 1

        group_totals = {}
        for p in buffer:
            if p.group_id not in group_totals:
                group_totals[p.group_id] = p.batch_group_total

        if len(group_totals) < expected_groups:
            return None

        total_expected = sum(group_totals.values())

        if total_expected <= 0:
            patches_per_img = self._get_patches_per_image(config)
            total_expected = config.batch_size * patches_per_img

        if len(buffer) < total_expected:
            return None

        t_id = next(task_id_gen)
        current_batch_patches = buffer[:total_expected]

        total_layers = int(
            config.scheduler_kwargs.get(
                "total_layers",
                config.transmission_kwargs.get("total_layers", 40),
            )
        )

        instructions = [
            Instruction(OpType.LOAD_INPUT),
            Instruction(OpType.PREPARE_TOKENS),
            Instruction(OpType.APPROX_FORWARD, {"layers": (0, total_layers)}),
            Instruction(OpType.HEAD_INFERENCE),
            Instruction(OpType.SEND_RESPONSE),
            Instruction(OpType.FREE_SESSION),
        ]

        return Task(
            task_id=t_id,
            request_id=t_id,
            payload=current_batch_patches,
            instructions=instructions,
        )

    def _get_patches_per_image(self, config: ExperimentConfig) -> int:
        H, W = config.image_shape[:2]
        ph, pw = config.patch_size

        if config.transmission_policy_name in {"Laplacian", "ProgressiveLaplacian"}:
            levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 1, 0]), reverse=True)
            total = 0
            for lvl in levels:
                scale = 2 ** lvl
                if bool(config.transmission_kwargs.get('preserve_input_shape', False)):
                    short_side = min(H, W)
                    h = short_side // scale
                    w = short_side // scale
                else:
                    h = H // scale
                    w = W // scale
                gh = (h + ph - 1) // ph
                gw = (w + pw - 1) // pw
                total += gh * gw
            return total

        if bool(config.transmission_kwargs.get('preserve_input_shape', False)):
            return 1

        gh = (H + ph - 1) // ph
        gw = (W + pw - 1) // pw
        return gh * gw
