import multiprocessing
import time
import torch
import numpy as np
import traceback
from typing import Dict, Any

from offload.common import Task, InferenceResult
from offload.common.protocol import OpType, Instruction
from offload.policies import get_transmission

class WorkerModule(multiprocessing.Process):
    """
    Stateful Worker Process.
    Maintains session context and accumulates patches for progressive decoding.
    """

    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        
        # Context: { request_id : { 'patch_buffer': [], 'current_feat': Tensor ... } }
        self.sessions: Dict[int, Dict[str, Any]] = {}
        
        self.config = None
        self.policy = None
        self.model = None
        
        # ImageNet normalization constants
        self.normalize_avg = np.array([0.485, 0.456, 0.406])
        self.normalize_std = np.array([0.229, 0.224, 0.225])

    def run(self):
        print("[Worker] Started.")
        
        # 1. Init CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Worker] Running on {self.device}")
        
        # 2. Pre-load normalization tensors
        self.norm_mean = torch.tensor(self.normalize_avg).view(1, 3, 1, 1).to(self.device).float()
        self.norm_std = torch.tensor(self.normalize_std).view(1, 3, 1, 1).to(self.device).float()

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
                        if self.model is None:
                            print("!!! [Worker] Warning: Model not loaded.")
                            continue
                        self.execute_pipeline(payload)
                        
                except Exception as e:
                    print(f"!!! [Worker] Main Loop Error: {e}")
                    traceback.print_exc()

    def _load_model(self, model_name: str):
        """Loads model architecture dynamically."""
        print(f"[Worker] Loading Model architecture: {model_name}...")
        
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

        if model_name == "resnet18":
            import timm

            self.model = timm.create_model('resnet18', pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
        elif model_name == "dinov3_classifier":
            from appcorr.models.dinov3.hub.classifiers import dinov3_vit7b16_lc

            self.model = dinov3_vit7b16_lc(
                pretrained=True,
                weights="~/cjpark/weights/dinov3/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth",
                backbone_weights="~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
            )

            self.model.backbone.set_appcorr_mode(
                enabled=False,
            )

            self.model.to(dtype=torch.bfloat16)
            self.model.to(self.device)
            self.model.eval()

        else:
            self.model = None
            raise ValueError(f"Unknown model name: {model_name}")

    @torch.autocast('cuda', dtype=torch.bfloat16)
    def execute_pipeline(self, task: Task):
        req_id = task.request_id
        
        if req_id not in self.sessions:
            self.sessions[req_id] = {}
        context = self.sessions[req_id]

        try:
            for instr in task.instructions:
                self._dispatch(instr, task, context)
        except Exception as e:
            print(f"!!! [Worker] Pipeline Error (Req {req_id}): {e}")
            traceback.print_exc()
            if req_id in self.sessions:
                del self.sessions[req_id]

    def _dispatch(self, instr: Instruction, task: Task, context: Dict[str, Any]):
        op = instr.op_type
        params = instr.params

        # --- Control Ops ---
        if op == OpType.LOAD_INPUT:
            if not task.payload: return
            
            # [CRITICAL] Accumulate patches for progressive decoding
            if 'patch_buffer' not in context:
                context['patch_buffer'] = []
            context['patch_buffer'].extend(task.payload)
            
            # Decode using accumulated buffer
            batch_np = self.policy.decode(context['patch_buffer'], self.config)
            
            # Preprocess
            tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().to(self.device)
            tensor = tensor / 255.0
            tensor = (tensor - self.norm_mean) / self.norm_std
            
            context['input'] = tensor
            context['current_feat'] = tensor 

        elif op == OpType.SEND_RESPONSE:
            if 'output' in context:
                output_logits = context['output']
                _, top5 = torch.topk(output_logits, k=5, dim=1)
                preds = top5.cpu().numpy().tolist()
            else:
                preds = []
            
            result = InferenceResult(task.task_id, time.time(), preds)
            self.output_queue.put(result)

        elif op == OpType.FREE_SESSION:
            if task.request_id in self.sessions:
                del self.sessions[task.request_id]

        # --- Computation Ops ---
        elif op == OpType.FULL_INFERENCE:
            inp = context.get('input')
            context['output'] = self.model(inp)

        elif op == OpType.APPROX_FORWARD:
            pass

        elif op == OpType.CORRECT_FORWARD:
            pass

        elif op == OpType.HEAD_INFERENCE:
            input = context.get('input')
            context['output'] = self.model(input)

            # feat = context.get('current_feat')
            # if hasattr(self.model, 'forward_head'):
            #     context['output'] = self.model.forward_head(feat)
            # else:
            #     context['output'] = self.model(feat)