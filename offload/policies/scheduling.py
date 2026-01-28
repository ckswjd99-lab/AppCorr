from typing import List, Optional, Any
from offload.common.protocol import Patch, Task, ExperimentConfig, Instruction, OpType
from .interface import ISchedulingPolicy

class BatchCountBasedPolicy(ISchedulingPolicy):
    """
    Waits for a full batch of patches, then triggers a standard FULL_INFERENCE task.
    - Dynamically calculates patch count (e.g., for Laplacian).
    - Instructs the worker to perform a standard forward pass.
    """

    def decide(
        self, 
        buffer: List[Patch], 
        config: ExperimentConfig, 
        task_id_gen: Any
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

class GroupTriggerPolicy(ISchedulingPolicy):
    """
    Pipelined Scheduling.
    Triggers task when a transmission group is collected.
    Dynamically generates pipeline instructions.
    """

    def __init__(self):
        self.current_request_id = None

    def decide(
        self, 
        buffer: List[Patch], 
        config: ExperimentConfig, 
        task_id_gen: Any
    ) -> Optional[Task]:
        
        if not buffer:
            return None
        
        # Peek header
        head_patch = buffer[0]
        current_group = head_patch.group_id
        target_count = head_patch.batch_group_total
        
        # Check trigger condition
        if len(buffer) >= target_count:
            t_id = next(task_id_gen)
            
            # Manage Request ID (New for Group 0, reuse for others)
            if current_group == 0 or self.current_request_id is None:
                self.current_request_id = t_id
            
            # Extract patches
            current_batch_patches = buffer[:target_count]
            
            # Generate instructions
            instructions = self._get_pipeline_instructions(current_group, config)
            
            task = Task(
                task_id=t_id,
                request_id=self.current_request_id,
                payload=current_batch_patches,
                instructions=instructions
            )
            return task
            
        return None

    def _get_pipeline_instructions(self, group_id: int, config: ExperimentConfig) -> List[Instruction]:
        """
        Maps Group IDs to pipeline steps.
        """
        total_layers = config.transmission_kwargs.get('total_layers', 40)
        num_res_groups = config.transmission_kwargs.get('num_groups', 4)
        
        chunk_size = total_layers // num_res_groups
        instructions = [Instruction(OpType.LOAD_INPUT), Instruction(OpType.PREPARE_TOKENS)]

        if group_id < num_res_groups:
            # Standard Phase: Correct valid history -> Approx next chunk
            current_chunk_start = group_id * chunk_size
            current_chunk_end = (group_id + 1) * chunk_size
            
            if current_chunk_start > 0:
                instructions.append(
                    Instruction(OpType.CORRECT_FORWARD, {
                        'layers': (0, current_chunk_start),
                        'group_id': group_id
                    })
                )
            
            instructions.append(
                Instruction(OpType.APPROX_FORWARD, {
                    'layers': (current_chunk_start, current_chunk_end)
                })
            )

        else:
            # Final Phase: Correct entire model & Inference
            instructions.append(
                Instruction(OpType.CORRECT_FORWARD, {
                    'layers': (0, total_layers),
                    'group_id': group_id
                })
            )
            instructions.append(Instruction(OpType.HEAD_INFERENCE))
            instructions.append(Instruction(OpType.SEND_RESPONSE))
            instructions.append(Instruction(OpType.FREE_SESSION))
            
        return instructions