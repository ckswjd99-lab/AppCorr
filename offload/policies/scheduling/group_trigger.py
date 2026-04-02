from typing import List, Optional, Any
from offload.common.protocol import Patch, Task, ExperimentConfig, Instruction, OpType, normalize_appcorr_kwargs
from ..interface import ISchedulingPolicy

class GroupTriggerPolicy(ISchedulingPolicy):
    """
    Pipelined Scheduling.
    Triggers task when a transmission group is collected.
    Dynamically generates pipeline instructions.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.current_request_id = None

    @staticmethod
    def _get_patch_request_id(patch: Patch) -> int | None:
        request_id = getattr(patch, 'request_id', -1)
        if isinstance(request_id, int) and request_id >= 0:
            return request_id
        return None

    @staticmethod
    def _mobile_hint_required(config: ExperimentConfig) -> bool:
        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs)
        return (
            appcorr_options.get('mobile_pscore', 'none') != 'none'
            and float(appcorr_options.get('mobile_pscore_weight', 0.0)) != 0.0
        )

    @staticmethod
    def _hint_ready_prefix_end(
        request_id: int | None,
        total_layers: int,
        hint_ready_layers_by_request: dict[int, set[int]] | None,
    ) -> int:
        if request_id is None:
            return 0
        ready_layers = (hint_ready_layers_by_request or {}).get(int(request_id), set())
        prefix_end = 0
        while prefix_end < total_layers and prefix_end in ready_layers:
            prefix_end += 1
        return prefix_end

    @staticmethod
    def _needs_final_global_approx(config: ExperimentConfig) -> bool:
        if getattr(config, 'model_name', None) != 'dinov3_detector':
            return False
        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs)
        return (
            bool(appcorr_options.get('generated_from_client', False))
            and appcorr_options.get('global_source_mode', 'final_correct') == 'final_correct'
        )

    def decide(
        self, 
        buffer: List[Patch], 
        config: ExperimentConfig, 
        task_id_gen: Any,
        **kwargs
    ) -> Optional[Task]:
        
        if not buffer:
            return None
        
        # Peek header
        head_patch = buffer[0]
        current_group = head_patch.group_id
        target_count = head_patch.batch_group_total
        
        # Check trigger condition
        if len(buffer) >= target_count:
            mobile_request_id = self._get_patch_request_id(head_patch)
            if current_group == 0 or self.current_request_id is None:
                self.current_request_id = mobile_request_id
            elif mobile_request_id is not None:
                self.current_request_id = mobile_request_id

            request_id = self.current_request_id
            total_layers = int(config.transmission_kwargs.get('total_layers', 40))
            num_res_groups = int(config.transmission_kwargs.get('num_groups', 4))
            chunk_size = total_layers // num_res_groups

            if self._mobile_hint_required(config):
                if current_group < num_res_groups:
                    required_end = current_group * chunk_size
                else:
                    required_end = total_layers
                ready_prefix_end = self._hint_ready_prefix_end(
                    request_id,
                    total_layers,
                    kwargs.get('hint_ready_layers_by_request'),
                )
                if required_end > 0 and ready_prefix_end < required_end:
                    return None

            t_id = next(task_id_gen)
            if request_id is None:
                request_id = t_id
                self.current_request_id = request_id
            
            # Extract patches
            current_batch_patches = buffer[:target_count]
            
            # Generate instructions
            instructions = self._get_pipeline_instructions(current_group, config)
            
            task = Task(
                task_id=t_id,
                request_id=request_id,
                payload=current_batch_patches,
                instructions=instructions
            )
            return task
            
        return None

    def _get_pipeline_instructions(self, group_id: int, config: ExperimentConfig) -> List[Instruction]:
        total_layers = config.transmission_kwargs.get('total_layers', 40)
        num_res_groups = config.transmission_kwargs.get('num_groups', 4)
        early_exit = config.early_exit_enabled()
        
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

            if early_exit:
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
            if self._needs_final_global_approx(config):
                instructions.append(
                    Instruction(OpType.APPROX_FORWARD, {
                        'layers': (0, total_layers),
                        'global_only': True,
                        'source_kind': 'global',
                    })
                )
            instructions.append(Instruction(OpType.HEAD_INFERENCE))
            instructions.append(Instruction(OpType.EXIT_ALL))
            instructions.append(Instruction(OpType.SEND_RESPONSE))
            instructions.append(Instruction(OpType.FREE_SESSION))
            
        return instructions
