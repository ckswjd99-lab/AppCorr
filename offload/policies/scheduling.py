from typing import List, Optional, Any
from offload.common.protocol import Patch, Task, ExperimentConfig
from .interface import ISchedulingPolicy

class BatchCountBasedPolicy(ISchedulingPolicy):
    """
    Waits for a full batch of patches, then creates a task with ALL patches.
    Simple Pass-Through logic.
    """

    def decide(
        self, 
        buffer: List[Patch], 
        config: ExperimentConfig, 
        task_id_gen: Any
    ) -> Optional[Task]:
        
        # Calculate total expected patches for a full request (Batch Size * Patches/Image)
        total_expected = config.batch_size * config.patches_per_image
        
        # 1. Trigger Condition: Do we have enough data for the whole batch?
        if len(buffer) >= total_expected:
            t_id = next(task_id_gen)
            
            # Extract the full batch raw data from buffer
            current_batch_patches = buffer[:total_expected]
            
            # 2. No Filtering (Pass-Through)
            # Send all patches in the buffer to the worker
            task = Task(
                task_id=t_id,
                mode='APPROX',
                payload=current_batch_patches, 
                layer_range=(0, 12)
            )
            return task
            
        return None