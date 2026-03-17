import multiprocessing
import time
import torch
import numpy as np
import traceback
from typing import Dict, Any
import threading
import queue

from offload.common import Task, InferenceResult
from offload.common.protocol import OpType, Instruction
from offload.policies import get_transmission
from offload.server.model import get_model_executor

class WorkerModule(multiprocessing.Process):
    """
    Stateful Worker Process.
    Maintains session context and delegates model execution to ModelExecutor.
    """

    def __init__(self, input_queue, output_queue, feedback_queue=None):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.feedback_queue = feedback_queue
        
        # Track session contexts
        self.sessions: Dict[int, Dict[str, Any]] = {}
        
        self.config = None
        self.policy = None
        self.executor = None

    def run(self):
        print("[Worker] Started.")
        
        # Set default device (can be overridden by config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Worker] Started with default device: {self.device}")
        
        # GPU Task Queue
        self.gpu_queue = queue.Queue()
        
        # Start Decoder Thread
        decoder_thread = threading.Thread(target=self._decoder_worker)
        decoder_thread.daemon = True
        decoder_thread.start()
        
        with torch.no_grad():
            self._gpu_worker()
            
    def _decoder_worker(self):
        """Continuously decode incoming tasks and pre-populate canvas."""
        print("[Decoder Thread] Started.")
        while True:
            try:
                msg = self.input_queue.get()
                if msg == 'STOP':
                    self.gpu_queue.put('STOP')
                    break
                
                msg_type, payload = msg
                
                if msg_type == 'CONFIG':
                    self.gpu_queue.put(msg)
                    continue
                    
                if msg_type == 'TIME_SYNC':
                    self.gpu_queue.put(msg)
                    continue
                    
                if msg_type == 'TASK':
                    task = payload
                    req_id = task.request_id
                    
                    if req_id not in self.sessions:
                         self.sessions[req_id] = {'events': [], 'canvas_np': None}
                    
                    context = self.sessions[req_id]
                    if task.payload:
                        for instr in task.instructions:
                            if instr.op_type == OpType.LOAD_INPUT:
                                max_arrival_time = max((p.arrival_time for p in task.payload if hasattr(p, 'arrival_time')), default=0.0)
                                if max_arrival_time > 0 and 'events' in context:
                                    context['events'].append({
                                        'type': 'SERVER_RECEIVE',
                                        'start': max_arrival_time,
                                        'end': max_arrival_time,
                                        'params': {}
                                    })
                                
                                # Accumulate patches for full reconstruction (required for Laplacian/Progressive)
                                if 'patch_buffer' not in context:
                                    context['patch_buffer'] = []
                                context['patch_buffer'].extend(task.payload)
                                
                                t_decode_start = time.time()
                                context['canvas_np'] = self.policy.decode(context['patch_buffer'], self.config, canvas=context.get('canvas_np'))
                                t_decode_end = time.time()
                                
                                if 'events' in context:
                                    context['events'].append({
                                        'type': 'Decode',
                                        'start': t_decode_start,
                                        'end': t_decode_end,
                                        'params': {}
                                    })
                                break

                    # Pass decoded canvas to GPU thread, copying to prevent race conditions.
                    canvas_ref = None
                    if context.get('canvas_np') is not None:
                         canvas_ref = context['canvas_np'].copy()
                         
                    self.gpu_queue.put(('TASK', (task, canvas_ref)))
                    
            except Exception as e:
                print(f"!!! [Decoder] Error: {e}")
                traceback.print_exc()
                
    def _gpu_worker(self):
        """Main GPU process loop."""
        print("[GPU Worker] Started.")
        while True:
            try:
                msg = self.gpu_queue.get()
                if msg == 'STOP':
                     break
                     
                msg_type, payload = msg
                
                if msg_type == 'CONFIG':
                    self.config = payload
                    self.sessions = {} # Clear sessions
                    
                    # Apply device override if specified in config
                    if hasattr(self.config, 'device') and self.config.device is not None:
                        self.device = torch.device(self.config.device)
                        print(f"[Worker] Device overridden by Config: {self.device}")
                        
                    self.policy = get_transmission(self.config.transmission_policy_name)
                    
                    self._load_model(self.config.model_name)
                    print(f"[Worker] Configured. Policy: {self.config.transmission_policy_name}, Model: {self.config.model_name}, Device: {self.device}")
                    continue
                    
                if msg_type == 'TASK':
                    task, decoded_canvas = payload
                    
                    if self.executor is None:
                        print("!!! [Worker] Warning: Model Executor not loaded.")
                        continue
                    
                    # Inject pre-decoded canvas into context
                    req_id = task.request_id
                    if req_id not in self.sessions:
                        self.sessions[req_id] = {'events': []}
                    self.sessions[req_id]['canvas_np'] = decoded_canvas
                    
                    self.execute_pipeline(task)

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
                with torch.cuda.nvtx.range(instr.op_type.name):
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
                
                if instr.op_type == OpType.APPROX_FORWARD and self.feedback_queue is not None:
                    end_layer = instr.params.get('layers', (0, 0))[1]
                    self.feedback_queue.put(('APPROX_DONE', req_id, end_layer))

        except Exception as e:
            print(f"!!! [Worker] Pipeline Error (Req {req_id}): {e}")
            traceback.print_exc()
            if req_id in self.sessions:
                del self.sessions[req_id]

    def _dispatch(self, instr: Instruction, task: Task, context: Dict[str, Any]):
        # Initialize batch slicing
        if 'active_indices' not in context:
             # Identity mapping
             context['active_indices'] = torch.arange(self.config.batch_size, device=self.device)
        
        op = instr.op_type
        
        # Skip processing if all samples exited
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
            # Canvas is pre-decoded by CPU thread.
            if context.get('canvas_np') is None: return
            
            # Preprocess (Model Executor)
            with torch.cuda.nvtx.range("Preprocess"):
                self.executor.preprocess(context['canvas_np'], task, context, self.config)

        elif op == OpType.PREPARE_TOKENS:
            self.executor.prepare_tokens(task, context, self.config)

        elif op == OpType.SEND_RESPONSE:
            # Consolidate results for remaining active indices
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