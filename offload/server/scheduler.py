import multiprocessing
import itertools
import time
from typing import List
from offload.common import Patch, Task, ExperimentConfig
from offload.policies import get_scheduler
from offload.common.utils import calculate_total_patches

class SchedulerModule(multiprocessing.Process):
    """Buffers patches and triggers Worker based on Policy."""

    def __init__(self, input_queue, worker_queue, control_queue, feedback_queue=None):
        super().__init__()
        self.input_queue = input_queue
        self.worker_queue = worker_queue
        self.control_queue = control_queue
        self.feedback_queue = feedback_queue
        
        self.buffer: List[Patch] = []
        self.config: ExperimentConfig = None
        self.policy = None
        self.task_counter = itertools.count()

    def run(self):
        print("[Scheduler] Started.")
        
        while True:
            if not self.control_queue.empty():
                cmd, data = self.control_queue.get()
                if cmd == 'CONFIG':
                    self.config = data
                    self.policy = get_scheduler(self.config.scheduler_policy_name)
                    self.worker_queue.put(('CONFIG', data))
                    self.buffer = []
                    self.task_counter = itertools.count()

                    print(f"[Scheduler] Configured with {self.config.scheduler_policy_name}")
                
                elif cmd == 'TIME_SYNC':
                     self.worker_queue.put(('TIME_SYNC', data))

            # Drain queue
            while not self.input_queue.empty():
                try:
                    item = self.input_queue.get_nowait()
                    self.buffer.append(item)
                except:
                    break
                    
            feedback_events = []
            # Drain feedback queue
            if self.feedback_queue is not None:
                while not self.feedback_queue.empty():
                    try:
                        fb_cmd, fb_req_id, fb_data = self.feedback_queue.get_nowait()
                        if fb_cmd == 'APPROX_DONE':
                            feedback_events.append(fb_data)
                    except:
                        break
            
            if self.policy and self.config:
                while True:
                    kwargs = {}
                    if feedback_events:
                        kwargs['feedback_events'] = feedback_events
                        
                    task = self.policy.decide(self.buffer, self.config, self.task_counter, **kwargs)
                    
                    # Pass feedback events once
                    feedback_events = []
                    
                    if task:
                        consume_count = len(task.payload)
                        
                        self.buffer = self.buffer[consume_count:]
                        
                        self.worker_queue.put(('TASK', task))
                    else:
                        break
            
            time.sleep(0.001)