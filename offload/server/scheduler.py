import multiprocessing
import itertools
import time
from typing import List
from offload.common import Patch, Task, ExperimentConfig
from offload.policies import get_scheduler

class SchedulerModule(multiprocessing.Process):
    """Buffers patches and triggers Worker based on Policy."""

    def __init__(self, input_queue, worker_queue, control_queue):
        super().__init__()
        self.input_queue = input_queue
        self.worker_queue = worker_queue
        self.control_queue = control_queue
        
        self.buffer: List[Patch] = []
        self.config: ExperimentConfig = None
        self.policy = None
        self.task_counter = itertools.count()

    def run(self):
        print("[Scheduler] Started.")
        
        while True:
            # 1. Handle Configuration Updates
            if not self.control_queue.empty():
                cmd, data = self.control_queue.get()
                if cmd == 'CONFIG':
                    self.config = data
                    self.policy = get_scheduler(self.config.scheduler_policy_name)
                    self.worker_queue.put(('CONFIG', data))
                    self.buffer = []
                    self.task_counter = itertools.count()
                    print(f"[Scheduler] Configured with {self.config.scheduler_policy_name}")

            # 2. Process Incoming Patches
            # Drain queue to batch process incoming patches
            while not self.input_queue.empty():
                try:
                    patch = self.input_queue.get_nowait()
                    self.buffer.append(patch)
                except:
                    break
            
            # 3. Consult Policy (Loop until buffer is drained)
            if self.policy and self.config:
                while True:
                    # Decide task based on buffer (does not modify buffer yet)
                    task = self.policy.decide(self.buffer, self.config, self.task_counter)
                    
                    if task:
                        # [CRITICAL] Consume based on TOTAL patches in a batch request
                        # Config defines how many patches constitute one 'Request'
                        consume_count = self.config.batch_size * self.config.patches_per_image
                        
                        # Remove processed patches from buffer
                        self.buffer = self.buffer[consume_count:]
                        
                        self.worker_queue.put(('TASK', task))
                    else:
                        # No more tasks can be formed
                        break
            
            time.sleep(0.001) # Yield CPU