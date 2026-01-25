import multiprocessing
import time
import numpy as np
from offload.common import Task, InferenceResult
from offload.policies import get_transmission

class WorkerModule(multiprocessing.Process):
    """Executes tasks (Batch execution with stateful cache)."""

    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.cache = {} 
        self.config = None
        self.policy = None # To hold transmission policy for decoding

    def run(self):
        print("[Worker] Started.")
        while True:
            msg_type, payload = self.input_queue.get()
            
            if msg_type == 'CONFIG':
                self.config = payload
                self.cache = {}
                # Load the same policy as mobile to decode
                self.policy = get_transmission(self.config.transmission_policy_name)
                print(f"[Worker] Configured. Policy: {self.config.transmission_policy_name}")
                continue
            
            if msg_type == 'TASK':
                task: Task = payload
                self.execute_inference(task)

    def execute_inference(self, task: Task):
        try:
            # 1. Decode Batch
            # policy.decode handles reconstruction of (B, H, W, C)
            # Sparse patches (not in payload) will result in zero-filled areas
            batch_input = self.policy.decode(task.payload, self.config)
            
            # Debug: Check shape
            # Expected: (Batch_Size, H, W, C)
            print(f"[Worker] Input Shape: {batch_input.shape}") 
            
            # 2. Simulate Inference
            time.sleep(0.05) 
            
            # 3. Dummy Prediction (Batch)
            # Return list of predictions, one per image in batch
            B = self.config.batch_size
            fake_preds = np.random.randint(0, 1000, size=(B,))

            result = InferenceResult(
                task_id=task.task_id,
                timestamp=time.time(), # This should carry original timestamp ideally
                output=fake_preds 
            )
            self.output_queue.put(result)

        except Exception as e:
            print(f"!!! [Worker] Error: {e}")