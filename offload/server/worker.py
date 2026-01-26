import multiprocessing
import time
import numpy as np
import timm
import torch
import traceback

from offload.common import Task, InferenceResult
from offload.policies import get_transmission

class WorkerModule(multiprocessing.Process):
    """Executes tasks (Batch execution with stateful cache)."""

    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        
        # Initialize lightweight variables only (No CUDA operations here)
        self.cache = {} 
        self.config = None
        self.policy = None
        
        # Define normalization constants (numpy)
        self.normalize_avg = np.array([0.485, 0.456, 0.406])
        self.normalize_std = np.array([0.229, 0.224, 0.225])

    def run(self):
        print("[Worker] Started.")
        
        # Initialize CUDA context inside the child process
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Worker] Loading Model on {self.device}...")
        
        self.model = timm.create_model('resnet18', pretrained=True).to(self.device)
        self.model.eval()
        
        # Pre-load normalization tensors to GPU for efficiency
        self.norm_mean = torch.tensor(self.normalize_avg).view(1, 3, 1, 1).to(self.device).float()
        self.norm_std = torch.tensor(self.normalize_std).view(1, 3, 1, 1).to(self.device).float()

        # Disable gradient calculation globally for the loop
        with torch.no_grad(): 
            while True:
                msg_type, payload = self.input_queue.get()
                
                if msg_type == 'CONFIG':
                    self.config = payload
                    self.cache = {}
                    self.policy = get_transmission(self.config.transmission_policy_name)
                    print(f"[Worker] Configured. Policy: {self.config.transmission_policy_name}")
                    continue
                
                if msg_type == 'TASK':
                    task: Task = payload
                    self.execute_inference(task)

    def execute_inference(self, task: Task):
        try:
            # Decode Batch to Numpy array: (B, H, W, C)
            batch_input = self.policy.decode(task.payload, self.config)
            
            # Preprocessing & Move to GPU
            # 1. Numpy -> Tensor (CPU) -> Permute -> Float -> GPU
            input_tensor = torch.from_numpy(batch_input).permute(0, 3, 1, 2).float().to(self.device)
            
            # 2. Normalize using pre-loaded GPU tensors
            input_tensor = input_tensor / 255.0
            input_tensor = (input_tensor - self.norm_mean) / self.norm_std
            
            # Inference
            # output: (Batch, Num_Classes) -> indices: (Batch, 5)
            output = self.model(input_tensor)
            _, top5_indices = torch.topk(output, k=5, dim=1)
            preds = top5_indices.cpu().numpy().tolist()

            # Send results
            result = InferenceResult(
                task_id=task.task_id,
                timestamp=time.time(), 
                output=preds 
            )
            self.output_queue.put(result)

        except Exception as e:
            print(f"!!! [Worker] Error: {e}")
            traceback.print_exc()