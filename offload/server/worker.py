import multiprocessing
import time
import torch
import numpy as np
import traceback
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import threading
import queue

from offload.common import Task, InferenceResult
from offload.common.protocol import OpType, Instruction
from offload.policies import get_transmission
from offload.server.model import get_model_executor
from offload.server.sr import create_lowres_sr_engine


@dataclass
class _MonitorJob:
    """Represents a single dispatched instruction awaiting GPU completion."""
    op_type: OpType
    start_ev: torch.cuda.Event
    end_ev: torch.cuda.Event
    req_id: int
    task: Task
    params: dict
    meta: Any = None


class WorkerModule(multiprocessing.Process):
    """
    Stateful Worker Process for asynchronous GPU execution.

    Architecture: [Decoder] -> gpu_queue -> [GPU Worker] -> monitor_queue -> [Reaper]

    GPU Worker acts as a pure launcher (no sync). Reaper thread handles CUDA event
    synchronization, timing, feedback, and response delivery.
    """

    def __init__(self, input_queue, output_queue, feedback_queue=None):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.feedback_queue = feedback_queue

        self.sessions: Dict[int, Dict[str, Any]] = {}

        self.config = None
        self.policy = None
        self.executor = None
        self.sr_engine = None

    def run(self):
        print("[Worker] Started.")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Worker] Started with default device: {self.device}")

        # GPU Task Queue (Decoder → GPU Worker)
        self.gpu_queue = queue.Queue()

        # Monitor Queue (GPU Worker → Reaper Thread)
        self.monitor_queue = queue.Queue()

        # Global timing anchor: ties CPU wall-clock to the CUDA timeline.
        # anchor_cpu + anchor_ev.elapsed_time(ev) / 1000.0 → absolute timestamp.
        self.anchor_ev = torch.cuda.Event(enable_timing=True)
        self.anchor_ev.record()
        self.anchor_cpu = time.time()

        # Start Decoder Thread
        decoder_thread = threading.Thread(target=self._decoder_worker, daemon=True)
        decoder_thread.start()

        # Start Reaper Thread
        reaper_thread = threading.Thread(target=self._reaper_worker, daemon=True)
        reaper_thread.start()

        with torch.no_grad():
            self._gpu_worker()

    # ------------------------------------------------------------------ #
    #  Decoder Thread                                                      #
    # ------------------------------------------------------------------ #

    def _decoder_worker(self):
        """Decodes incoming tasks and pre-populates the canvas."""
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
                        self.sessions[req_id] = self._create_session_context()

                    context = self.sessions[req_id]
                    if task.payload:
                        for instr in task.instructions:
                            if instr.op_type == OpType.LOAD_INPUT:
                                group_id = task.payload[0].group_id
                                max_arrival_time = max(
                                    (p.arrival_time for p in task.payload if hasattr(p, 'arrival_time')),
                                    default=0.0
                                )
                                if max_arrival_time > 0 and 'events' in context:
                                    context['events'].append({
                                        'type': 'SERVER_RECEIVE',
                                        'start': max_arrival_time,
                                        'end': max_arrival_time,
                                        'params': {}
                                    })

                                if 'patch_buffer' not in context:
                                    context['patch_buffer'] = []
                                context['patch_buffer'].extend(task.payload)

                                t_decode_start = time.time()
                                context['input_hr_np'] = self.policy.decode(
                                    context['patch_buffer'], self.config,
                                    canvas=context.get('input_hr_np')
                                )
                                if group_id == 0 and self.config.lowres_sr_enabled():
                                    context['input_lr_native_np'] = self.policy.decode_lowres(task.payload, self.config)
                                    context['input_sr_tensor'] = None
                                t_decode_end = time.time()

                                if 'events' in context:
                                    context['events'].append({
                                        'type': 'Decode',
                                        'start': t_decode_start,
                                        'end': t_decode_end,
                                        'params': {}
                                    })
                                break

                    input_state_ref = self._snapshot_input_state(context)
                    self.gpu_queue.put(('TASK', (task, input_state_ref)))

            except Exception as e:
                print(f"!!! [Decoder] Error: {e}")
                traceback.print_exc()

    # ------------------------------------------------------------------ #
    #  GPU Worker — pure kernel launcher, never synchronizes              #
    # ------------------------------------------------------------------ #

    def _gpu_worker(self):
        """Main GPU loop. Records events and dispatches kernels without blocking."""
        print("[GPU Worker] Started.")
        while True:
            try:
                msg = self.gpu_queue.get()
                if msg == 'STOP':
                    self.monitor_queue.put('STOP')
                    break

                msg_type, payload = msg

                if msg_type == 'CONFIG':
                    self.config = payload
                    self.sessions = {}
                    if hasattr(self.config, 'device') and self.config.device is not None:
                        self.device = torch.device(self.config.device)
                        print(f"[Worker] Device overridden by Config: {self.device}")
                    self.policy = get_transmission(self.config.transmission_policy_name)
                    self._validate_lowres_sr_config()
                    self._load_sr_engine()
                    self._load_model(self.config.model_name)
                    print(f"[Worker] Configured. Policy: {self.config.transmission_policy_name}, "
                          f"Model: {self.config.model_name}, Device: {self.device}")
                    continue

                if msg_type == 'TASK':
                    task, input_state = payload
                    if self.executor is None:
                        print("!!! [Worker] Warning: Model Executor not loaded.")
                        continue
                    req_id = task.request_id
                    if req_id not in self.sessions:
                        self.sessions[req_id] = self._create_session_context()
                    self.sessions[req_id].update(input_state)
                    self.execute_pipeline(task)

                elif msg_type == 'TIME_SYNC':
                    self.output_queue.put(time.time())

            except Exception as e:
                print(f"!!! [Worker] Main Loop Error: {e}")
                traceback.print_exc()

    # ------------------------------------------------------------------ #
    #  Reaper Thread — waits for GPU completion, fires callbacks          #
    # ------------------------------------------------------------------ #

    def _reaper_worker(self):
        """Processes completed GPU jobs: handles timing, feedback, responses, and cleanup."""
        print("[Reaper Thread] Started.")
        while True:
            try:
                job = self.monitor_queue.get()
                if job == 'STOP':
                    break

                # Block THIS thread until the GPU has completed this instruction.
                job.end_ev.synchronize()

                # Compute absolute CPU-equivalent timestamps from the CUDA timeline.
                # elapsed_time() returns ms; divide by 1000 for seconds.
                ts_start = self.anchor_cpu + self.anchor_ev.elapsed_time(job.start_ev) / 1000.0
                ts_end   = self.anchor_cpu + self.anchor_ev.elapsed_time(job.end_ev)   / 1000.0

                context = self.sessions.get(job.req_id)
                if context is None:
                    continue  # Session was already freed

                # Log event
                if 'events' in context:
                    event_data = {
                        'type': job.op_type.name,
                        'start': ts_start,
                        'end': ts_end,
                        'params': job.params,
                    }
                    if job.meta:
                        event_data['meta'] = job.meta
                    context['events'].append(event_data)

                # Feedback: notify scheduler that an approximation pass finished
                if job.op_type == OpType.APPROX_FORWARD and self.feedback_queue is not None:
                    end_layer = job.params.get('layers', (0, 0))[1]
                    self.feedback_queue.put(('APPROX_DONE', job.req_id, end_layer))

                # Response: all previous events are guaranteed to be logged (FIFO),
                # so server_events is complete at this point.
                if job.op_type == OpType.SEND_RESPONSE:
                    self._finalize_and_send_response(job.task, context)

                # Session cleanup (deferred so Reaper can finish logging first)
                if job.op_type == OpType.FREE_SESSION:
                    self.sessions.pop(job.req_id, None)

            except Exception as e:
                print(f"!!! [Reaper] Error: {e}")
                traceback.print_exc()

    def _finalize_and_send_response(self, task: Task, context: Dict[str, Any]):
        """Called by Reaper Thread after GPU is done and all events are logged."""
        server_events = context.get('events', [])
        preds = context.get('_pending_preds', [])
        result = InferenceResult(task.task_id, time.time(), preds, server_events)
        self.output_queue.put(result)

    # ------------------------------------------------------------------ #
    #  Model Loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_model(self, model_name: str):
        print(f"[Worker] Loading Model Executor for: {model_name}...")
        try:
            self.executor = get_model_executor(model_name, self.device)
            self.executor.load_model(model_name, self.config)
        except Exception as e:
            print(f"!!! [Worker] Failed to load executor: {e}")
            self.executor = None
            raise e

    def _load_sr_engine(self):
        self.sr_engine = None
        if not self.config.lowres_sr_enabled():
            return

        sr_config = self.config.get_lowres_sr_config()
        print(
            f"[Worker] Loading low-res SR engine: {sr_config['model']} "
            f"from {sr_config['weights_dir']}"
        )
        self.sr_engine = create_lowres_sr_engine(sr_config, self.device)

    def _validate_lowres_sr_config(self):
        if not self.config.lowres_sr_enabled():
            return

        supported_transmissions = {'Laplacian', 'ProgressiveLaplacian'}
        if self.config.transmission_policy_name not in supported_transmissions:
            raise ValueError(
                "lowres_sr requires Laplacian or ProgressiveLaplacian transmission, "
                f"got {self.config.transmission_policy_name}"
            )

    # ------------------------------------------------------------------ #
    #  Pipeline Execution                                                  #
    # ------------------------------------------------------------------ #

    @torch.autocast('cuda', dtype=torch.bfloat16)
    def execute_pipeline(self, task: Task):
        req_id = task.request_id
        if req_id not in self.sessions:
            self.sessions[req_id] = self._create_session_context()
        context = self.sessions[req_id]

        try:
            for instr in task.instructions:
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev   = torch.cuda.Event(enable_timing=True)

                start_ev.record()
                with torch.cuda.nvtx.range(instr.op_type.name):
                    meta = self._dispatch(instr, task, context)
                end_ev.record()

                # Hand off to Reaper Thread immediately; GPU Worker moves on.
                self.monitor_queue.put(_MonitorJob(
                    op_type=instr.op_type,
                    start_ev=start_ev,
                    end_ev=end_ev,
                    req_id=req_id,
                    task=task,
                    params=instr.params.copy(),
                    meta=meta,
                ))

        except Exception as e:
            print(f"!!! [Worker] Pipeline Error (Req {req_id}): {e}")
            traceback.print_exc()
            if req_id in self.sessions:
                del self.sessions[req_id]

    def _dispatch(self, instr: Instruction, task: Task, context: Dict[str, Any]):
        if 'active_indices' not in context:
            context['active_indices'] = torch.arange(self.config.batch_size, device=self.device)

        op = instr.op_type

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
            if context.get('input_hr_np') is None:
                return
            group_id = self._get_task_group_id(task)
            batch_np = context['input_hr_np']

            if group_id == 0 and self.config.lowres_sr_enabled():
                if context.get('input_lr_native_np') is None:
                    raise RuntimeError("Missing low-resolution input for SR-enabled LOAD_INPUT.")
                if self.sr_engine is None:
                    raise RuntimeError("Low-resolution SR engine is not loaded.")
                if context.get('input_sr_tensor') is None:
                    with torch.cuda.nvtx.range("LowResSR"):
                        with torch.autocast('cuda', enabled=False):
                            context['input_sr_tensor'] = self.sr_engine.upscale_tensor(
                                context['input_lr_native_np'],
                                target_hw=self.config.image_shape[:2],
                            )
                batch_np = context['input_sr_tensor']
            with torch.cuda.nvtx.range("Preprocess"):
                self.executor.preprocess(batch_np, task, context, self.config)

        elif op == OpType.PREPARE_TOKENS:
            self.executor.prepare_tokens(task, context, self.config)

        elif op == OpType.SEND_RESPONSE:
            # Collect results now (may involve GPU ops); actual queue.put is
            # deferred to Reaper Thread after end_ev.synchronize() confirms
            # all GPU work is done and all events have been logged.
            if 'active_indices' in context and len(context['active_indices']) > 0:
                final_batch_results = self.executor.get_final_results(task, context, self.config)
                if 'final_results' not in context:
                    context['final_results'] = {}
                context['final_results'].update(final_batch_results)

            if 'final_results' in context:
                final_map = context['final_results']
                preds = [final_map.get(i, []) for i in range(self.config.batch_size)]
            else:
                preds = []

            # Stash for Reaper Thread to pick up in _finalize_and_send_response()
            context['_pending_preds'] = preds

        elif op == OpType.FREE_SESSION:
            # Actual deletion is deferred to Reaper Thread to avoid deleting
            # the session while previous jobs for this session are still in flight.
            pass

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
            final_batch_results = self.executor.get_final_results(task, context, self.config)
            if 'final_results' not in context:
                context['final_results'] = {}
            context['final_results'].update(final_batch_results)
            context['active_indices'] = torch.empty(0, device=self.device, dtype=torch.long)

    def _create_session_context(self) -> Dict[str, Any]:
        return {
            'events': [],
            'patch_buffer': [],
            'input_hr_np': None,
            'input_lr_native_np': None,
            'input_sr_tensor': None,
        }

    def _snapshot_input_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = {}
        for key in ('input_hr_np', 'input_lr_native_np'):
            value = context.get(key)
            snapshot[key] = value.copy() if value is not None else None
        return snapshot

    def _get_task_group_id(self, task: Task) -> Optional[int]:
        if task.payload:
            return task.payload[0].group_id
        return None
