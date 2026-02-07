import multiprocessing
import time
import torch
import numpy as np
import traceback
from typing import Dict, Any

from offload.common import Task, InferenceResult
from offload.common.protocol import OpType, Instruction
from offload.policies import get_transmission
from offload.server.model import get_model_executor

class WorkerModule(multiprocessing.Process):
    """
    Stateful Worker Process.
    Maintains session context and delegates model execution to ModelExecutor.
    """

    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        
        # Context: { request_id : { 'patch_buffer': [], 'current_feat': Tensor ... } }
        self.sessions: Dict[int, Dict[str, Any]] = {}
        
        self.config = None
        self.policy = None
        self.executor = None

    def run(self):
        print("[Worker] Started.")
        
        # Init CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Worker] Running on {self.device}")
        
        with torch.no_grad():
            while True:
                try:
                    msg_type, payload = self.input_queue.get()
                    
                    if msg_type == 'CONFIG':
                        self.config = payload
                        self.sessions = {} # Clear sessions
                        self.policy = get_transmission(self.config.transmission_policy_name)
                        
                        self._load_model(self.config.model_name)
                        print(f"[Worker] Configured. Policy: {self.config.transmission_policy_name}, Model: {self.config.model_name}")
                        continue
                    
                    if msg_type == 'TASK':
                        if self.executor is None:
                            print("!!! [Worker] Warning: Model Executor not loaded.")
                            continue
                        self.execute_pipeline(payload)

                    elif msg_type == 'TIME_SYNC':
                        # Echo back with server timestamp
                        self.output_queue.put(time.time())

                        
                except Exception as e:
                    print(f"!!! [Worker] Main Loop Error: {e}")
                    traceback.print_exc()

    def _load_model(self, model_name: str):
        """Loads model executor."""
        print(f"[Worker] Loading Model Executor for: {model_name}...")
        try:
            self.executor = get_model_executor(model_name, self.device)
            self.executor.load_model(model_name, self.config)
        except Exception as e:
            print(f"!!! [Worker] Failed to load executor: {e}")
            self.executor = None
            raise e

    @torch.autocast('cuda', dtype=torch.bfloat16)
    def execute_pipeline(self, task: Task):
        req_id = task.request_id
        
        if req_id not in self.sessions:
            self.sessions[req_id] = {'events': []}
        context = self.sessions[req_id]

        try:
            for instr in task.instructions:
                t_start = time.time()
                meta = self._dispatch(instr, task, context)
                torch.cuda.synchronize()
                t_end = time.time()
                
                # Record Event
                if 'events' in context:
                    event_data = {
                        'type': instr.op_type.name,
                        'start': t_start,
                        'end': t_end,
                        'params': instr.params.copy()
                    }
                    if meta:
                        event_data['meta'] = meta
                    
                    context['events'].append(event_data)
        except Exception as e:
            print(f"!!! [Worker] Pipeline Error (Req {req_id}): {e}")
            traceback.print_exc()
            if req_id in self.sessions:
                del self.sessions[req_id]

    def _dispatch(self, instr: Instruction, task: Task, context: Dict[str, Any]):
        # Helper for batch slicing initialization
        if 'active_indices' not in context:
             # Identity mapping initially: [0, 1, 2, ... B-1]
             context['active_indices'] = torch.arange(self.config.batch_size, device=self.device)
        
        op = instr.op_type
        
        # Optimize: Skip Logic if All Exited
        if len(context['active_indices']) == 0:
            SKIPPABLE_OPS = {
                OpType.LOAD_INPUT, OpType.PREPARE_TOKENS,
                OpType.FULL_INFERENCE, OpType.APPROX_FORWARD, OpType.CORRECT_FORWARD,
                OpType.HEAD_INFERENCE, OpType.DECIDE_EXIT
            }
            if op in SKIPPABLE_OPS:
                return {'skipped': True}

        # Control Ops
        if op == OpType.LOAD_INPUT:
            if not task.payload: return
            
            # Decode (Transmission Policy)
            if 'patch_buffer' not in context:
                context['patch_buffer'] = []
            context['patch_buffer'].extend(task.payload)
            
            # Decode
            batch_np = self.policy.decode(context['patch_buffer'], self.config) # [B, H, W, C]
            
            # Preprocess (Model Executor)
            self.executor.preprocess(batch_np, task, context, self.config)

        elif op == OpType.PREPARE_TOKENS:
            self.executor.prepare_tokens(task, context, self.config)

        elif op == OpType.SEND_RESPONSE:
            # Consolidate Results
            # If there are active indices remaining (not early exited), get their results now
            if 'active_indices' in context and len(context['active_indices']) > 0:
                final_batch_results = self.executor.get_final_results(task, context, self.config)
                if 'final_results' not in context:
                    context['final_results'] = {}
                context['final_results'].update(final_batch_results)

            if 'final_results' in context:
                 # Reconstruct ordered list
                 final_map = context['final_results']
                 preds = []
                 for i in range(self.config.batch_size):
                     preds.append(final_map.get(i, [])) # Return empty list if missing
            else:
                preds = []
            
            # Include accumulated server events
            server_events = context.get('events', [])
            result = InferenceResult(task.task_id, time.time(), preds, server_events)
            self.output_queue.put(result)

        elif op == OpType.FREE_SESSION:
            if task.request_id in self.sessions:
                del self.sessions[task.request_id]

        # Computation Ops
        elif op == OpType.FULL_INFERENCE:
            self.executor.full_inference(task, context, self.config)

        elif op == OpType.APPROX_FORWARD:
            self.executor.approx_forward(instr.params, context, self.config)

        elif op == OpType.CORRECT_FORWARD:
            self.executor.correct_forward(instr.params, context, self.config)

        elif op == OpType.HEAD_INFERENCE:
            return self.executor.head_inference(task, context, self.config)

        elif op == OpType.DECIDE_EXIT:
            return self.executor.decide_exit(task, context, self.config)

        elif op == OpType.EXIT_ALL:
             # Flush everything remaining to final_results
             final_batch_results = self.executor.get_final_results(task, context, self.config)
             if 'final_results' not in context:
                 context['final_results'] = {}
             context['final_results'].update(final_batch_results)
            
             # Clear state
             context['active_indices'] = torch.empty(0, device=self.device, dtype=torch.long)