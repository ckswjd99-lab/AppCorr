from typing import List, Optional, Any
from offload.common.protocol import Patch, Task, ExperimentConfig, Instruction, OpType
from ..interface import ISchedulingPolicy

class DynamicGroupTriggerPolicy(ISchedulingPolicy):
    """
    Dynamic Pipelined Scheduling.
    Interleaves correct and approx dynamically layer by layer.
    """

    def __init__(self):
        self.current_request_id = None
        self.latest_approx_layer_queued = 0
        self.current_group_id = -1
        self.ahead_layers = 2  # Proactive approx depth

    def decide(
        self, 
        buffer: List[Patch], 
        config: ExperimentConfig, 
        task_id_gen: Any,
        **kwargs
    ) -> Optional[Task]:
        
        feedback_events = kwargs.get('feedback_events', [])
        
        if not buffer and not feedback_events:
            return None
            
        if buffer:
            head_patch = buffer[0]
            current_group = head_patch.group_id
            target_count = head_patch.batch_group_total
            
            if len(buffer) >= target_count:
                t_id = next(task_id_gen)
                
                if current_group == 0 or self.current_request_id is None:
                    self.current_request_id = t_id
                    self.latest_approx_layer_queued = 0
                    self.current_group_id = -1
                
                self.current_group_id = current_group
                current_batch_patches = buffer[:target_count]
                
                total_layers = config.transmission_kwargs.get('total_layers', 40)
                num_res_groups = config.transmission_kwargs.get('num_groups', 4)
                
                instructions = [Instruction(OpType.LOAD_INPUT), Instruction(OpType.PREPARE_TOKENS)]
                
                if current_group < num_res_groups:
                    if self.latest_approx_layer_queued > 0:
                        instructions.append(
                            Instruction(OpType.CORRECT_FORWARD, {
                                'layers': (0, self.latest_approx_layer_queued),
                                'group_id': current_group
                            })
                        )
                    
                    # Proactive queuing
                    if self.latest_approx_layer_queued == 0:
                        for _ in range(self.ahead_layers):
                            if self.latest_approx_layer_queued < total_layers:
                                instructions.append(
                                    Instruction(OpType.APPROX_FORWARD, {
                                        'layers': (self.latest_approx_layer_queued, self.latest_approx_layer_queued + 1)
                                    })
                                )
                                self.latest_approx_layer_queued += 1
                    
                    self._append_extra_decide_instructions(instructions)
                else:
                    # Final Phase:
                    # 1. Correct up to what we approximated
                    if self.latest_approx_layer_queued > 0:
                        instructions.append(
                            Instruction(OpType.CORRECT_FORWARD, {
                                'layers': (0, self.latest_approx_layer_queued),
                                'group_id': current_group
                            })
                        )
                    
                    # 2. Forward (Approx) the rest
                    if self.latest_approx_layer_queued < total_layers:
                        instructions.append(
                            Instruction(OpType.APPROX_FORWARD, {
                                'layers': (self.latest_approx_layer_queued, total_layers)
                            })
                        )
                    
                    instructions.append(Instruction(OpType.HEAD_INFERENCE))
                    instructions.append(Instruction(OpType.EXIT_ALL))
                    instructions.append(Instruction(OpType.SEND_RESPONSE))
                    instructions.append(Instruction(OpType.FREE_SESSION))
                    
                task = Task(
                    task_id=t_id,
                    request_id=self.current_request_id,
                    payload=current_batch_patches,
                    instructions=instructions
                )
                return task
            
        # Handle feedback if buffer is not full or we just processed it
        if feedback_events:
            # We process the latest feedback event for queuing more approx
            # since feedback monotonically increases the completed layer.
            # But here we just want to ensure we keep pushing approx if needed.
            max_layer_idx = max(feedback_events)
            
            num_res_groups = config.transmission_kwargs.get('num_groups', 4)
            total_layers = config.transmission_kwargs.get('total_layers', 40)
            
            # Enqueue more approx if corrections still active
            if self.current_group_id != -1 and self.current_group_id < num_res_groups and self.latest_approx_layer_queued < total_layers:
                diff = self.latest_approx_layer_queued - max_layer_idx
                
                instructions = []
                while diff < self.ahead_layers and self.latest_approx_layer_queued < total_layers:
                    instructions.append(
                        Instruction(OpType.APPROX_FORWARD, {
                            'layers': (self.latest_approx_layer_queued, self.latest_approx_layer_queued + 1)
                        })
                    )
                    self.latest_approx_layer_queued += 1
                    diff += 1
                    
                if instructions:
                    t_id = next(task_id_gen)
                    return Task(
                        task_id=t_id,
                        request_id=self.current_request_id,
                        payload=[], # No patches needed
                        instructions=instructions
                    )
        return None

    def _append_extra_decide_instructions(self, instructions: List[Instruction]):
        """Hook for subclasses to add extra instructions like early exit checks."""
        pass

class DynamicGroupTriggerEarlyExitPolicy(DynamicGroupTriggerPolicy):
    def _append_extra_decide_instructions(self, instructions: List[Instruction]):
        instructions.append(Instruction(OpType.HEAD_INFERENCE))
        instructions.append(Instruction(OpType.DECIDE_EXIT))
