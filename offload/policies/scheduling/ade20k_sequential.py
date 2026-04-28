from typing import List, Optional, Any

from offload.common.protocol import Patch, Task, ExperimentConfig, Instruction, OpType
from ..interface import ISchedulingPolicy


class ADE20KSequentialPolicy(ISchedulingPolicy):
    """Sequential ADE20K segmentor scheduling.

    Uses the decomposed pipeline (prepare_tokens -> approx_forward -> head_inference)
    instead of full_inference.  Designed for testing the decomposed segmentor path.
    """

    def decide(
        self,
        buffer: List[Patch],
        config: ExperimentConfig,
        task_id_gen: Any,
        **kwargs,
    ) -> Optional[Task]:
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

    @staticmethod
    def _get_patches_per_image(config: ExperimentConfig) -> int:
        if bool(config.transmission_kwargs.get("preserve_input_shape", False)):
            return 1
        H, W = config.image_shape[:2]
        ph, pw = config.patch_size
        return ((H + ph - 1) // ph) * ((W + pw - 1) // pw)
