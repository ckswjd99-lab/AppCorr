from typing import List, Optional, Any
from offload.common.protocol import Patch, Task, ExperimentConfig
from .interface import ISchedulingPolicy

class BatchCountBasedPolicy(ISchedulingPolicy):
    """
    Waits for a full batch of patches, then creates a task with ALL patches.
    Dynamically calculates expected patch count based on transmission policy.
    """

    def decide(
        self, 
        buffer: List[Patch], 
        config: ExperimentConfig, 
        task_id_gen: Any
    ) -> Optional[Task]:
        
        # Calculate the exact number of patches required per image
        patches_per_img = self._get_patches_per_image(config)
        total_expected = config.batch_size * patches_per_img
        
        # Check if buffer has enough data for a full batch
        if len(buffer) >= total_expected:
            t_id = next(task_id_gen)
            
            # Extract the full batch
            current_batch_patches = buffer[:total_expected]
            
            # Create Task with all patches (Pass-Through)
            task = Task(
                task_id=t_id,
                mode='APPROX',
                payload=current_batch_patches, 
                layer_range=(0, 12)
            )
            return task
            
        return None

    def _get_patches_per_image(self, config: ExperimentConfig) -> int:
        """Calculates total patches per image based on current policy."""
        H, W = config.image_shape[:2]
        ph, pw = config.patch_size

        if config.transmission_policy_name == "Laplacian":
            # Sum patches for all active levels defined in kwargs
            levels = config.transmission_kwargs.get('pyramid_levels', [2, 1, 0])
            total = 0
            for lvl in levels:
                scale = 2 ** lvl
                # Ceil division to account for edge patches
                gh = (H // scale + ph - 1) // ph
                gw = (W // scale + pw - 1) // pw
                total += gh * gw
            return total
        
        else:
            # Default (Raw, Zlib): Single level grid
            gh = (H + ph - 1) // ph
            gw = (W + pw - 1) // pw
            return gh * gw