from typing import List, Optional, Any
from offload.common.protocol import Patch, Task, ExperimentConfig, Instruction, OpType
from ..interface import ISchedulingPolicy

class GroupTriggerPolicy(ISchedulingPolicy):
    """
    Pipelined Scheduling.
    Triggers task when a transmission group is collected.
    Dynamically generates pipeline instructions.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
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
        total_layers = config.transmission_kwargs.get('total_layers', 40)
        num_res_groups = config.transmission_kwargs.get('num_groups', 4)
        
        chunk_size = total_layers // num_res_groups
        instructions = [Instruction(OpType.LOAD_INPUT), Instruction(OpType.PREPARE_TOKENS)]

        if group_id < num_res_groups:
            # Correct valid history -> Approx next chunk
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
            instructions.append(Instruction(OpType.EXIT_ALL))
            instructions.append(Instruction(OpType.SEND_RESPONSE))
            instructions.append(Instruction(OpType.FREE_SESSION))
            
        return instructions
