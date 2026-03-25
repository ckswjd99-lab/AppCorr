from typing import List, Optional, Any
from offload.common.protocol import Patch, Task, ExperimentConfig, Instruction, OpType
from ..interface import ISchedulingPolicy

class BatchCountBasedPolicy(ISchedulingPolicy):
    """
    Waits for a full batch of patches, then triggers a standard FULL_INFERENCE task.
    """

    def decide(
        self, 
        buffer: List[Patch], 
        config: ExperimentConfig, 
        task_id_gen: Any,
        **kwargs
    ) -> Optional[Task]:
        
        # Calculate dynamic patch count based on transmission policy
        patches_per_img = self._get_patches_per_image(config)
        total_expected = config.batch_size * patches_per_img
        
        # Check Trigger Condition
        if len(buffer) >= total_expected:
            t_id = next(task_id_gen)
            
            # Extract the full batch
            current_batch_patches = buffer[:total_expected]
            
            # Generate Instruction (Simple Pass-Through)
            instructions = [
                Instruction(OpType.LOAD_INPUT),
                Instruction(OpType.FULL_INFERENCE),
                Instruction(OpType.SEND_RESPONSE),
                Instruction(OpType.FREE_SESSION)
            ]
            
            # Create Task
            task = Task(
                task_id=t_id,
                request_id=t_id,
                payload=current_batch_patches, 
                instructions=instructions
            )
            return task
            
        return None

    def _get_patches_per_image(self, config: ExperimentConfig) -> int:
        """Calculates total patches per image based on transmission policy."""
        H, W = config.image_shape[:2]
        ph, pw = config.patch_size

        if config.transmission_policy_name == "Laplacian":
            # Sum patches for all active levels
            levels = config.transmission_kwargs.get('pyramid_levels', [2, 1, 0])
            total = 0
            for lvl in levels:
                scale = 2 ** lvl
                # Ceil division to cover edges
                gh = (H // scale + ph - 1) // ph
                gw = (W // scale + pw - 1) // pw
                total += gh * gw
            return total
        
        else:
            # Default (Raw, Zlib): Single level grid
            gh = (H + ph - 1) // ph
            gw = (W + pw - 1) // pw
            return gh * gw
