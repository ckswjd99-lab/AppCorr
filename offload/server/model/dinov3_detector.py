from typing import Any, Dict
import torch
import numpy as np
from offload.common import Task
from .base import ModelExecutor
from .utils import load_weight_mmap

class DINOv3DetectorExecutor(ModelExecutor):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.normalize_avg = np.array([0.485, 0.456, 0.406])
        self.normalize_std = np.array([0.229, 0.224, 0.225])
        self.norm_mean = torch.tensor(self.normalize_avg).view(1, 3, 1, 1).to(self.device).float()
        self.norm_std = torch.tensor(self.normalize_std).view(1, 3, 1, 1).to(self.device).float()

    def load_model(self, model_name: str, config: Any):
        print(f"[Executor] Loading Detector Model (MMap): {model_name}...")
        if self.model is not None:
             del self.model
             torch.cuda.empty_cache()

        from appcorr.models.dinov3.hub.detectors import dinov3_vit7b16_de

        # Init empty
        self.model = dinov3_vit7b16_de(
             pretrained=False,
             weights="COCO2017",
             backbone_weights="LVD1689M"
        )
        
        # Move to device/dtype
        self.model.to(dtype=torch.bfloat16)
        self.model.to(self.device)
        
        try:
             # 1. Load Backbone Weights
             backbone_path = "~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
             print(f"[Executor] Loading Backbone Weights from {backbone_path}")
             backbone_state = load_weight_mmap(backbone_path)
             
             vit_backbone = self._get_vit_backbone()
             vit_backbone.load_state_dict(backbone_state, strict=True)
             del backbone_state
             
             # 2. Load Detector Weights
             # Note: This might partially overwrite backbone if the checkpoint includes it (fine-tuned)
             head_path = "~/cjpark/weights/dinov3/dinov3_vit7b16_coco_detr_head-b0235ff7.pth"
             print(f"[Executor] Loading Detector Head Weights from {head_path}")
             head_ckpt = load_weight_mmap(head_path)
             
             if "model" in head_ckpt:
                 head_state = head_ckpt["model"]
             else:
                 head_state = head_ckpt
                 
             self.model.detector.load_state_dict(head_state, strict=False)
             del head_ckpt, head_state
             
        except Exception as e:
            print(f"!!! [Executor] Failed to load detector weights: {e}")
            raise e
            
        self.model.eval()
        
    def _get_vit_backbone(self):
        # Traverse detector to find the DINOv3 ViT backbone
        # detector (Detr) -> backbone (Joiner) -> [0] (WindowsWrapper or DINOBackbone) -> (.backbone or ._backbone.backbone)
        
        joiner = self.model.detector.backbone
        # Joiner is Sequential(backbone, pos_encoding)
        inner = joiner[0]
        
        if hasattr(inner, "_backbone"): # WindowsWrapper
             dino_backbone = inner._backbone
        else: # DINOBackbone
             dino_backbone = inner
             
        return dino_backbone.backbone

    def preprocess(self, batch_np: np.ndarray, task: Task, context: Dict[str, Any], config: Any):
        # Normalization logic same as Classifier mostly
        tensor = torch.from_numpy(batch_np).to(self.device).permute(0, 3, 1, 2).float()
        tensor = tensor / 255.0
        tensor = (tensor - self.norm_mean) / self.norm_std
        
        if 'active_indices' in context and len(context['active_indices']) < config.batch_size:
             active_indices = context['active_indices']
             tensor = tensor[active_indices]

        context['input_tensor'] = tensor

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        pass

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        pass

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        pass

    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        # reuse head_inference dummy logic
        self.head_inference(task, context, config)

    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        """
        Runs the actual detector model on the input_tensor.
        """
        if 'input_tensor' not in context:
            return {}

        input_tensor = context['input_tensor']
        
        # Inference
        with torch.inference_mode():
            outputs = self.model(input_tensor)
            
        # Store for get_final_results
        context['det_outputs'] = outputs
        
        return {}

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        results = {}
        if 'det_outputs' in context and 'active_indices' in context:
             outputs = context['det_outputs']
             current_active_indices = context['active_indices']
             
             cpu_indices = current_active_indices.cpu().numpy()
             
             for i, orig_idx in enumerate(cpu_indices):
                 if i < len(outputs):
                     pred = outputs[i]
                     serializable_pred = {
                         'scores': pred['scores'].float().cpu().tolist(),
                         'labels': pred['labels'].long().cpu().tolist(),
                         'boxes': pred['boxes'].float().cpu().tolist()
                     }
                     results[int(orig_idx)] = serializable_pred
                     
        return results

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return {'num_exits': 0}
