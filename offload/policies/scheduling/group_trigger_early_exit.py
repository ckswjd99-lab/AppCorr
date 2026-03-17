from typing import List
from offload.common.protocol import ExperimentConfig, Instruction, OpType
from .group_trigger import GroupTriggerPolicy

class GroupTriggerEarlyExitPolicy(GroupTriggerPolicy):
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

            instructions.append(Instruction(OpType.HEAD_INFERENCE))
            instructions.append(Instruction(OpType.DECIDE_EXIT))

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
