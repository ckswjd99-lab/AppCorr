from typing import List, Optional, Any
from offload.common.protocol import Patch, Task, ExperimentConfig, Instruction, OpType, normalize_appcorr_kwargs
from ..interface import ISchedulingPolicy


def build_balanced_layer_boundaries(
    total_layers: int,
    num_correction_groups: int,
    final_full_layers: int,
) -> tuple[int, ...]:
    """Split the progressively corrected prefix into balanced contiguous stages."""
    total_layers = int(total_layers)
    num_correction_groups = int(num_correction_groups)
    final_full_layers = int(final_full_layers)

    if total_layers <= 0:
        raise ValueError(f"total_layers must be positive, got {total_layers}")
    if num_correction_groups <= 0:
        raise ValueError(
            f"num_correction_groups must be positive, got {num_correction_groups}"
        )
    if final_full_layers < 0:
        raise ValueError(
            f"final_full_layers must be non-negative, got {final_full_layers}"
        )

    progressive_layers = total_layers - final_full_layers
    if progressive_layers < num_correction_groups:
        raise ValueError(
            "The progressively corrected prefix must contain at least one layer per "
            f"correction group, got total_layers={total_layers}, "
            f"final_full_layers={final_full_layers}, "
            f"num_correction_groups={num_correction_groups}"
        )

    base_size, remainder = divmod(progressive_layers, num_correction_groups)
    boundaries = [0]
    for stage_idx in range(num_correction_groups):
        stage_size = base_size + (1 if stage_idx < remainder else 0)
        boundaries.append(boundaries[-1] + stage_size)
    return tuple(boundaries)


class GroupTriggerPolicy(ISchedulingPolicy):
    """
    Pipelined Scheduling.
    Triggers task when a transmission group is collected.
    Dynamically generates pipeline instructions.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.current_request_id = None

    @staticmethod
    def _needs_final_global_approx(config: ExperimentConfig) -> bool:
        if getattr(config, 'model_name', None) != 'dinov3_detector':
            return False
        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
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
            t_id = next(task_id_gen)
            
            # Manage Request ID (New for Group 0, reuse for others)
            if current_group == 0 or self.current_request_id is None:
                self.current_request_id = t_id
            
            # Extract patches
            current_batch_patches = buffer[:target_count]
            
            # Generate instructions
            instructions = self._get_pipeline_instructions(head_patch, config)
            
            task = Task(
                task_id=t_id,
                request_id=self.current_request_id,
                payload=current_batch_patches,
                instructions=instructions
            )
            return task
            
        return None

    def _get_pipeline_instructions(
        self,
        head_patch: Patch,
        config: ExperimentConfig,
    ) -> List[Instruction]:
        group_id = int(head_patch.group_id)
        total_layers = config.transmission_kwargs.get('total_layers', 40)
        num_res_groups = int(
            head_patch.num_correction_groups
            or config.transmission_kwargs.get('num_groups', 4)
        )
        final_full_layers = int(
            config.scheduler_kwargs.get('final_full_layers', 0)
        )
        early_exit = config.early_exit_enabled()

        layer_boundaries = build_balanced_layer_boundaries(
            total_layers,
            num_res_groups,
            final_full_layers,
        )
        progressive_end = layer_boundaries[-1]
        instructions = [Instruction(OpType.LOAD_INPUT), Instruction(OpType.PREPARE_TOKENS)]

        if group_id < num_res_groups:
            # Correct valid history -> Approx next chunk
            current_chunk_start = layer_boundaries[group_id]
            current_chunk_end = layer_boundaries[group_id + 1]
            
            if current_chunk_start > 0:
                instructions.append(
                    Instruction(OpType.CORRECT_FORWARD, {
                        'layers': (0, current_chunk_start),
                        'group_id': group_id
                    })
                )
            
            instructions.append(
                Instruction(OpType.APPROX_FORWARD, {
                    'layers': (current_chunk_start, current_chunk_end),
                })
            )

            if early_exit:
                instructions.append(Instruction(OpType.HEAD_INFERENCE))
                instructions.append(Instruction(OpType.DECIDE_EXIT))

        else:
            if group_id > num_res_groups:
                raise ValueError(
                    f"Unexpected group_id={group_id}; final group is {num_res_groups}"
                )

            # Final Phase: correct every layer that has an approximation cache, then
            # run the held-back suffix once on the final-resolution feature.
            if progressive_end > 0:
                instructions.append(
                    Instruction(OpType.CORRECT_FORWARD, {
                        'layers': (0, progressive_end),
                        'group_id': group_id
                    })
                )
            if final_full_layers > 0:
                instructions.append(
                    Instruction(OpType.APPROX_FORWARD, {
                        'layers': (progressive_end, total_layers),
                        'cache_mode': 'none',
                        'phase': 'final_full',
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
