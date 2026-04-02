from typing import List, Optional, Any
from offload.common.protocol import Patch, Task, ExperimentConfig, Instruction, OpType, normalize_appcorr_kwargs
from ..interface import ISchedulingPolicy

class DynamicGroupTriggerPolicy(ISchedulingPolicy):
    """
    Dynamic Pipelined Scheduling.
    Interleaves correct and approx dynamically layer by layer.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.current_request_id = None
        self.latest_approx_layer_queued = 0
        self.latest_approx_layer_completed = 0
        self.latest_corrected_layer_queued = 0
        self.current_group_id = -1
        self.ahead_layers = 2  # Proactive approx depth
        
        if config:
            self.ahead_layers = config.scheduler_kwargs.get(
                "ahead_layers",
                config.transmission_kwargs.get("ahead_layers", self.ahead_layers)
            )

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

    def _reset_request_state(self):
        self.latest_approx_layer_queued = 0
        self.latest_approx_layer_completed = 0
        self.latest_corrected_layer_queued = 0
        self.current_group_id = -1

    def decide(
        self, 
        buffer: List[Patch], 
        config: ExperimentConfig, 
        task_id_gen: Any,
        **kwargs
    ) -> Optional[Task]:
        
        feedback_events = kwargs.get('feedback_events', [])
        hint_ready_layers_by_request = kwargs.get('hint_ready_layers_by_request')
        hint_updated = bool(kwargs.get('hint_updated', False))
        if feedback_events:
            self.latest_approx_layer_completed = max(
                self.latest_approx_layer_completed,
                max(int(layer_idx) for layer_idx in feedback_events),
            )

        if not buffer and not feedback_events and not hint_updated:
            return None

        total_layers = int(config.transmission_kwargs.get('total_layers', 40))
        num_res_groups = int(config.transmission_kwargs.get('num_groups', 4))
        hint_required = self._mobile_hint_required(config)

        if buffer:
            head_patch = buffer[0]
            current_group = head_patch.group_id
            target_count = head_patch.batch_group_total
            
            if len(buffer) >= target_count:
                mobile_request_id = self._get_patch_request_id(head_patch)
                if current_group == 0 or self.current_request_id is None:
                    self.current_request_id = mobile_request_id
                    self._reset_request_state()
                elif mobile_request_id is not None:
                    self.current_request_id = mobile_request_id
                
                self.current_group_id = current_group
                current_batch_patches = buffer[:target_count]
                ready_prefix_end = (
                    total_layers
                    if not hint_required
                    else self._hint_ready_prefix_end(
                        self.current_request_id,
                        total_layers,
                        hint_ready_layers_by_request,
                    )
                )

                instructions = [Instruction(OpType.LOAD_INPUT), Instruction(OpType.PREPARE_TOKENS)]
                
                if current_group < num_res_groups:
                    correction_end = min(self.latest_approx_layer_queued, ready_prefix_end)
                    if correction_end > self.latest_corrected_layer_queued:
                        instructions.append(
                            Instruction(OpType.CORRECT_FORWARD, {
                                'layers': (0, correction_end),
                                'group_id': current_group
                            })
                        )
                        self.latest_corrected_layer_queued = correction_end
                    
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
                    
                    if config.early_exit_enabled() and len(instructions) > 2:
                        instructions.append(Instruction(OpType.HEAD_INFERENCE))
                        instructions.append(Instruction(OpType.DECIDE_EXIT))
                else:
                    # Final Phase:
                    if ready_prefix_end >= self.latest_approx_layer_queued:
                        if self.latest_approx_layer_queued > self.latest_corrected_layer_queued:
                            instructions.append(
                                Instruction(OpType.CORRECT_FORWARD, {
                                    'layers': (0, self.latest_approx_layer_queued),
                                    'group_id': current_group
                                })
                            )
                            self.latest_corrected_layer_queued = self.latest_approx_layer_queued

                        if self.latest_approx_layer_queued < total_layers:
                            instructions.append(
                                Instruction(OpType.APPROX_FORWARD, {
                                    'layers': (self.latest_approx_layer_queued, total_layers)
                                })
                            )
                            self.latest_approx_layer_queued = total_layers

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
                    
                t_id = next(task_id_gen)
                request_id = self.current_request_id if self.current_request_id is not None else t_id
                self.current_request_id = request_id
                task = Task(
                    task_id=t_id,
                    request_id=request_id,
                    payload=current_batch_patches,
                    instructions=instructions
                )
                return task

        if self.current_group_id == -1 or self.current_request_id is None:
            return None

        ready_prefix_end = (
            total_layers
            if not hint_required
            else self._hint_ready_prefix_end(
                self.current_request_id,
                total_layers,
                hint_ready_layers_by_request,
            )
        )
        instructions = []

        if self.current_group_id < num_res_groups:
            correction_end = min(self.latest_approx_layer_queued, ready_prefix_end)
            if correction_end > self.latest_corrected_layer_queued:
                instructions.append(
                    Instruction(OpType.CORRECT_FORWARD, {
                        'layers': (0, correction_end),
                        'group_id': self.current_group_id
                    })
                )
                self.latest_corrected_layer_queued = correction_end
                if config.early_exit_enabled():
                    instructions.append(Instruction(OpType.HEAD_INFERENCE))
                    instructions.append(Instruction(OpType.DECIDE_EXIT))

            diff = self.latest_approx_layer_queued - self.latest_approx_layer_completed
            while diff < self.ahead_layers and self.latest_approx_layer_queued < total_layers:
                instructions.append(
                    Instruction(OpType.APPROX_FORWARD, {
                        'layers': (self.latest_approx_layer_queued, self.latest_approx_layer_queued + 1)
                    })
                )
                self.latest_approx_layer_queued += 1
                diff += 1
        else:
            if ready_prefix_end >= self.latest_approx_layer_queued:
                if self.latest_approx_layer_queued > self.latest_corrected_layer_queued:
                    instructions.append(
                        Instruction(OpType.CORRECT_FORWARD, {
                            'layers': (0, self.latest_approx_layer_queued),
                            'group_id': self.current_group_id
                        })
                    )
                    self.latest_corrected_layer_queued = self.latest_approx_layer_queued

                if self.latest_approx_layer_queued < total_layers:
                    instructions.append(
                        Instruction(OpType.APPROX_FORWARD, {
                            'layers': (self.latest_approx_layer_queued, total_layers)
                        })
                    )
                    self.latest_approx_layer_queued = total_layers

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

        if instructions:
            t_id = next(task_id_gen)
            return Task(
                task_id=t_id,
                request_id=self.current_request_id,
                payload=[],
                instructions=instructions
            )
        return None
