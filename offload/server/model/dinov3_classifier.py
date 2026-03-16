from typing import Any, Dict
import torch
import numpy as np
from offload.common import Task
from .base import ModelExecutor
from .utils import load_weight_mmap

class DINOv3ClassifierExecutor(ModelExecutor):
    def __init__(self, device: torch.device):
        super().__init__(device)
        # ImageNet normalization constants
        self.normalize_avg = np.array([0.485, 0.456, 0.406])
        self.normalize_std = np.array([0.229, 0.224, 0.225])
        self.norm_mean = torch.tensor(self.normalize_avg).view(1, 3, 1, 1).to(self.device).float()
        self.norm_std = torch.tensor(self.normalize_std).view(1, 3, 1, 1).to(self.device).float()

    def load_model(self, model_name: str, config: Any):
        print(f"[Executor] Loading Model (MMap): {model_name}...")
        if self.model is not None:
             del self.model
             torch.cuda.empty_cache()

        from appcorr.models.dinov3.hub.classifiers import dinov3_vit7b16_lc

        # Init empty model (random weights)
        self.model = dinov3_vit7b16_lc(
            pretrained=False,
            # We must provide some valid enum or string, but it won't be used for loading
            weights="IMAGENET1K", 
            backbone_weights="LVD1689M",
        )

        # Move to target device/dtype BEFORE loading weights to save RAM
        self.model.to(dtype=torch.bfloat16)
        self.model.to(self.device)
        
        try:
            # Load Backbone Weights
            backbone_path = "~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
            print(f"[Executor] Loading Backbone Weights from {backbone_path}")
            backbone_state = load_weight_mmap(backbone_path)
            self.model.backbone.load_state_dict(backbone_state, strict=True)
            del backbone_state
            
            # Load Head Weights
            head_path = "~/cjpark/weights/dinov3/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth"
            print(f"[Executor] Loading Head Weights from {head_path}")
            head_state = load_weight_mmap(head_path)
            self.model.linear_head.load_state_dict(head_state, strict=True)
            del head_state
            
        except Exception as e:
            print(f"!!! [Executor] Failed to load weights: {e}")
            raise e

        # Configure AppCorr Mode via Config
        appcorr_kwargs = config.appcorr_kwargs.copy()
        # Remove non-argument keys if any
        appcorr_kwargs.pop('generated_from_client', None)
        
        if appcorr_kwargs:
            print(f"[Executor] Enabling AppCorr Mode: {appcorr_kwargs}")
            self.model.backbone.set_appcorr_mode(**appcorr_kwargs)
        else:
            self.model.backbone.set_appcorr_mode(enabled=False)

        self.model.eval()

    def preprocess(self, batch_np: np.ndarray, task: Task, context: Dict[str, Any], config: Any):
        # Preprocess
        with torch.cuda.nvtx.range("Preprocess::PinMemory"):
            tensor = torch.from_numpy(batch_np)
            if hasattr(tensor, 'pin_memory'):
                tensor = tensor.pin_memory()
                
        with torch.cuda.nvtx.range("Preprocess::ToDevice"):
            tensor = tensor.to(device=self.device, non_blocking=True).permute(0, 3, 1, 2).float()
            tensor = tensor / 255.0
            tensor = (tensor - self.norm_mean) / self.norm_std

        # Sliced Update Handling
        with torch.cuda.nvtx.range("Preprocess::Slicing"):
            if 'active_indices' in context and len(context['active_indices']) < config.batch_size:
                active_indices = context['active_indices']
                tensor = tensor[active_indices]
            context['input_tensor'] = tensor

        with torch.cuda.nvtx.range("Preprocess::GroupMap"):
            # Optimization: Vectorized dindice construction with group_map
            if 'cached_dindices' not in context:
                context['cached_dindices'] = {}

            # Initialize or Retrieve group_map: [CurrentBatch, Max_Tokens]
            if 'group_map' not in context:
                H, W = config.image_shape[:2]
                if isinstance(config.patch_size, int):
                    ph = pw = config.patch_size
                else:
                    ph, pw = config.patch_size
                num_patches = (H // ph) * (W // pw)
                # Init with -1 (no group)
                curr_B = tensor.shape[0]
                context['group_map'] = torch.full((curr_B, num_patches), -1, device=self.device, dtype=torch.long)
            
            group_map = context['group_map']
            
            # Vectorized Update of Group Map - Map Original Index -> Local Index if sliced
            if 'active_indices' in context and len(context['active_indices']) < config.batch_size:
                # Build lookup: { original_idx : local_idx }
                active_list = context['active_indices'].tolist()
                idx_map = { orig: local for local, orig in enumerate(active_list) }
                
                # Filter payload
                valid_payload = [p for p in task.payload if p.image_idx in idx_map]
                
                if valid_payload:
                    b_list = [idx_map[p.image_idx] for p in valid_payload]
                    s_list = [p.spatial_idx for p in valid_payload]
                    g_list = [p.group_id for p in valid_payload]
                    
                    b_t = torch.tensor(b_list, device=self.device, dtype=torch.long)
                    s_t = torch.tensor(s_list, device=self.device, dtype=torch.long)
                    g_t = torch.tensor(g_list, device=self.device, dtype=torch.long)
                    
                    group_map[b_t, s_t] = g_t
            else:
                # Standard full-batch update (Identity mapping)
                b_list = [p.image_idx for p in task.payload]
                s_list = [p.spatial_idx for p in task.payload]
                g_list = [p.group_id for p in task.payload]
                
                if b_list:
                    b_t = torch.tensor(b_list, device=self.device, dtype=torch.long)
                    s_t = torch.tensor(s_list, device=self.device, dtype=torch.long)
                    g_t = torch.tensor(g_list, device=self.device, dtype=torch.long)
                    
                    group_map[b_t, s_t] = g_t
                    
        with torch.cuda.nvtx.range("Preprocess::Dindices"):
            # Update Cached Dindices for involved groups
            if b_list: # Check if we had any updates
                 involved_groups = set(g_list)
                 num_pretokens = 1 + self.model.backbone.n_storage_tokens
                 B = group_map.shape[0]
                 
                 for gid in involved_groups:
                      mask = (g_t == gid)
                      if not mask.any(): continue
                      try:
                          # Directly use the known spatial indices for this group from s_t (and b_t if needed)
                          # Since s_t is the list of spatial indices for patches of this group
                          spatial_indices = s_t[mask].view(B, -1)
                      except RuntimeError:
                          print(f"!!! [Executor] Non-uniform group {gid} size detected. Fallback skip.")
                          continue

                      patch_indices = spatial_indices + num_pretokens
                      pre_indices = torch.arange(num_pretokens, device=self.device).unsqueeze(0).expand(B, -1)
                      context['cached_dindices'][gid] = torch.cat([pre_indices, patch_indices], dim=1)
    

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        if 'input_tensor' not in context: return

        tensor = context['input_tensor']
        backbone = self.model.backbone
            
        if hasattr(backbone, 'prepare_tokens_with_masks'):
            t2_x, hw_tuple = backbone.prepare_tokens_with_masks(tensor, None)
            context['input_tokens'] = t2_x
            context['hw_tuple'] = hw_tuple
            
            if 'current_feature' not in context:
                context['current_feature'] = t2_x

            if self.model.backbone.rope_embed is not None and 'rope_sincos' not in context:
                rope_sincos = self.model.backbone.rope_embed(H=hw_tuple[0], W=hw_tuple[1])
                context['rope_sincos'] = rope_sincos

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        layers = params.get('layers', range(0, 40))
        # Ensure context has required items
        if 'current_feature' not in context:
             if 'input_tokens' in context:
                 context['current_feature'] = context['input_tokens']
             else:
                 # Should not happen if flow is correct
                 return

        x_feature = context['current_feature']
        rope_sincos = context.get('rope_sincos')
        cache = context.get('cache_feature', {})
        
        start_l, end_l = layers[0], layers[1]
        
        if start_l == 0:
            x_feature = context['input_tokens']

        for lidx in range(start_l, end_l):
            blk = self.model.backbone.blocks[lidx]
            x_feature, cache = blk.approx(
                x_feature, rope_sincos, cache, tag=f"layer{lidx}",
                debug=False
            )
        
        context['current_feature'] = x_feature
        context['cache_feature'] = cache

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        layers = params.get('layers', range(0, 40))
        group_id = params.get('group_id', 1)
        start_l, end_l = layers[0], layers[1]

        # Prepare Update Indices
        if 'cached_dindices' in context and group_id in context['cached_dindices']:
            dindice = context['cached_dindices'][group_id]
        else:
             # Fallback recompute
             print(f"!!! [Executor] Warning: dindice cache miss for group {group_id}. Recomputing...")
             if 'group_map' in context:
                  mask = (context['group_map'] == group_id)
                  nonzero_indices = torch.nonzero(mask)
                  num_pretokens = 1 + self.model.backbone.n_storage_tokens
                  B = context['current_feature'].shape[0]
                  try:
                      spatial_indices = nonzero_indices[:, 1].view(B, -1)
                      patch_indices = spatial_indices + num_pretokens
                      pre_indices = torch.arange(num_pretokens, device=self.device).unsqueeze(0).expand(B, -1)
                      dindice = torch.cat([pre_indices, patch_indices], dim=1)
                      # Cache it
                      if 'cached_dindices' not in context: context['cached_dindices'] = {}
                      context['cached_dindices'][group_id] = dindice
                  except:
                      print("!!! [Executor] Failed to recompute dindice from map.")
                      return
             else:
                 # Cannot compute without group map
                 return

        cache = context.get('cache_feature', {})
        rope_sincos = context.get('rope_sincos')
        
        # logic from worker.py: x_temp starts from input_tokens
        x_temp = context.get('input_tokens')
        
        alive_ratio = getattr(self.model.backbone, 'appcorr_cls_alive_ratio', 0.2)
        
        # dindice must be passed to correct()
        if 'dindice' not in locals(): 
             return 

        for lidx in range(start_l, end_l):
            blk = self.model.backbone.blocks[lidx]
            x_temp, cache = blk.correct(
                x_temp, dindice, rope_sincos, cache, tag=f"layer{lidx}",
                cls_alive_ratio=alive_ratio,
                debug=False
            )
        
        context['current_feature'] = x_temp
        context['cache_feature'] = cache

    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        feat = context.get('current_feature')
        backbone = self.model.backbone
        
        # Apply Norm
        x_norm = backbone.norm(feat)
        x_norm_clstoken = x_norm[:, 0]
        x_norm_patchtokens = x_norm[:, backbone.n_storage_tokens + 1 :]
        
        # Simulating Classifier Wrapper
        linear_input = torch.cat(
            [x_norm_clstoken, x_norm_patchtokens.mean(dim=1)],
            dim=1,
        )
        output_logits = self.model.linear_head(linear_input)
        context['output'] = output_logits
        
        # Analysis Logging
        probs = torch.softmax(output_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)
        
        return {
            'entropy': entropy.cpu().numpy().tolist(),
            'top5_probs': top5_probs.cpu().numpy().tolist(),
            'top5_indices': top5_indices.cpu().numpy().tolist(),
            'active_indices': context.get('active_indices', torch.arange(len(probs), device=self.device)).cpu().numpy().tolist()
        }

    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        inp = context.get('input_tensor')
        if inp is not None:
            context['output'] = self.model(inp)

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        """Extracts Top-5 predictions from context['output'] for all active indices."""
        results = {}
        if 'output' in context and 'active_indices' in context:
            output_logits = context['output']
            _, top5 = torch.topk(output_logits, k=5, dim=1)
            
            current_active_indices = context['active_indices']
            preds_list = top5.cpu().numpy().tolist()
            
            for orig_idx, pred_list in zip(current_active_indices.cpu().numpy(), preds_list):
                results[int(orig_idx)] = pred_list
        return results

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        if 'output' not in context: return {}

        # Get Criteria
        metric = config.early_exit_kwargs.get('metric', 'max_prob')
        threshold = config.early_exit_kwargs.get('threshold', 0.9)
        
        output_logits = context['output']
        probs = torch.softmax(output_logits, dim=1)
        
        # Compute Top-5 for potential exits
        _, top5 = torch.topk(output_logits, k=5, dim=1)
        
        # Identify Exits
        if metric == 'max_prob':
            max_probs, _ = torch.max(probs, dim=1)
            exit_mask = max_probs >= threshold
        elif metric == 'top2_margin':
            top2_vals, _ = torch.topk(probs, k=2, dim=1)
            margin = top2_vals[:, 0] - top2_vals[:, 1]
            exit_mask = margin >= threshold
        elif metric == 'entropy':
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            exit_mask = entropy <= threshold
        else:
            # Default/Fallback
            max_probs, _ = torch.max(probs, dim=1)
            exit_mask = torch.zeros_like(max_probs, dtype=torch.bool)
            
        num_exits = exit_mask.sum().item()
        
        if num_exits > 0:
            print(f"[Executor] exiting {num_exits} samples (metric {metric}, thresh {threshold})")
            
            # Store Results for Exited Samples
            if 'final_results' not in context:
                context['final_results'] = {}
            
            # Map active index to original request index
            current_active_indices = context['active_indices']
            
            exited_original_indices = current_active_indices[exit_mask]
            exited_preds = top5[exit_mask].cpu().numpy().tolist()
            
            for orig_idx, pred_list in zip(exited_original_indices.cpu().numpy(), exited_preds):
                context['final_results'][int(orig_idx)] = pred_list
                
            # Slice State
            keep_mask = ~exit_mask
            
            if keep_mask.sum() == 0:
                context['active_indices'] = torch.empty(0, device=self.device, dtype=torch.long)
                if 'current_feature' in context: del context['current_feature']
                if 'input_tokens' in context: del context['input_tokens']
            else:
                 # Helper to slice
                 self._slice_context(context, keep_mask)

        return {
            'num_exits': num_exits,
            'threshold': threshold
        }

    def _slice_context(self, context, keep_mask):
        # Metadata
        context['active_indices'] = context['active_indices'][keep_mask]
        
        # Group Map
        if 'group_map' in context:
            context['group_map'] = context['group_map'][keep_mask]
            
        # Input Tensors
        if 'input_tokens' in context:
            context['input_tokens'] = context['input_tokens'][keep_mask]
            
        if 'input_tensor' in context:
            context['input_tensor'] = context['input_tensor'][keep_mask]
            
        # Current Feature
        if 'current_feature' in context:
            context['current_feature'] = context['current_feature'][keep_mask]
            
        # Cache (KV Cache)
        if 'cache_feature' in context:
            cache = context['cache_feature']
            for k in list(cache.keys()):
                v = cache[k]
                if isinstance(v, tuple):
                    new_val = tuple(x[keep_mask] if isinstance(x, torch.Tensor) else x for x in v)
                    cache[k] = new_val
                elif isinstance(v, torch.Tensor):
                    cache[k] = v[keep_mask]
                del v
                
        # Clear derived structures
        if 'cached_dindices' in context:
            context['cached_dindices'] = {}

        # Output (Logits)
        if 'output' in context:
            context['output'] = context['output'][keep_mask]
