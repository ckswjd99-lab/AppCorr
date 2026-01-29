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
        
        # Init CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Worker] Running on {self.device}")
        
        # Pre-load normalization tensors
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

                    elif msg_type == 'TIME_SYNC':
                        # Echo back with server timestamp
                        self.output_queue.put(time.time())

                        
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

            # Configure AppCorr Mode via Config
            appcorr_kwargs = self.config.appcorr_kwargs.copy()
            # Remove non-argument keys if any
            appcorr_kwargs.pop('generated_from_client', None)
            
            if appcorr_kwargs:
                print(f"[Worker] Enabling AppCorr Mode: {appcorr_kwargs}")
                self.model.backbone.set_appcorr_mode(**appcorr_kwargs)
            else:
                self.model.backbone.set_appcorr_mode(enabled=False)

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
        # Helper for batch slicing
        if 'active_indices' not in context:
             # Identity mapping initially: [0, 1, 2, ... B-1]
             context['active_indices'] = torch.arange(self.config.batch_size, device=self.device)
        
        op = instr.op_type
        params = instr.params
        backbone = self.model.backbone

        # --- Optimize: Skip Logic if All Exited ---
        # If active_indices is empty, we only run Finalization Ops
        if len(context['active_indices']) == 0:
            SKIPPABLE_OPS = {
                OpType.LOAD_INPUT, OpType.PREPARE_TOKENS,
                OpType.FULL_INFERENCE, OpType.APPROX_FORWARD, OpType.CORRECT_FORWARD,
                OpType.HEAD_INFERENCE, OpType.DECIDE_EXIT
            }
            if op in SKIPPABLE_OPS:
                return {'skipped': True}

        # --- Control Ops ---
        if op == OpType.LOAD_INPUT:
            if not task.payload: return
            
            # Accumulate patches for progressive decoding
            if 'patch_buffer' not in context:
                context['patch_buffer'] = []
            context['patch_buffer'].extend(task.payload)
            
            # Track spatial indices per group for AppCorr
            if 'group_indices' not in context:
                # { group_id: { batch_idx: [spatial_idx, ...] } }
                context['group_indices'] = {}
            
            for p in task.payload:
                if p.group_id not in context['group_indices']:
                    context['group_indices'][p.group_id] = {b: [] for b in range(self.config.batch_size)}
                
                if p.image_idx < self.config.batch_size:
                    context['group_indices'][p.group_id][p.image_idx].append(p.spatial_idx)
            
            # Decode using accumulated buffer
            batch_np = self.policy.decode(context['patch_buffer'], self.config) # [B, H, W, C]
            
            # Preprocess
            tensor = torch.from_numpy(batch_np).to(self.device).permute(0, 3, 1, 2).float()
            tensor = tensor / 255.0
            tensor = (tensor - self.norm_mean) / self.norm_std

            # --- Sliced Update Handling ---
            if 'active_indices' in context and len(context['active_indices']) < self.config.batch_size:
                active_indices = context['active_indices']
                tensor = tensor[active_indices]

            context['input_tensor'] = tensor

            # Optimization: Vectorized dindice construction with group_map
            if 'cached_dindices' not in context:
                context['cached_dindices'] = {}

            # Initialize or Retrieve group_map: [CurrentBatch, Max_Tokens]
            if 'group_map' not in context:
                H, W = self.config.image_shape[:2]
                if isinstance(self.config.patch_size, int):
                    ph = pw = self.config.patch_size
                else:
                    ph, pw = self.config.patch_size
                num_patches = (H // ph) * (W // pw)
                # Init with -1 (no group)
                curr_B = tensor.shape[0]
                context['group_map'] = torch.full((curr_B, num_patches), -1, device=self.device, dtype=torch.long)
            
            group_map = context['group_map']
            
            # Vectorized Update of Group Map
            # Map Original Index -> Local Index if sliced
            if 'active_indices' in context and len(context['active_indices']) < self.config.batch_size:
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
            
            # Update Cached Dindices for involved groups
            involved_groups = set(g_list)
            num_pretokens = 1 + self.model.backbone.n_storage_tokens
            # B = self.config.batch_size # <--- BUG: This is static 32
            B = group_map.shape[0]       # <--- FIX: Use dynamic active batch size
            
            for gid in involved_groups:
                 # Mask: [B, N] boolean
                 mask = (group_map == gid)

                 # Vectorized Extraction
                 nonzero_indices = torch.nonzero(mask) # [Total_K, 2]

                 # Reshape to [B, K]
                 try:
                     spatial_indices = nonzero_indices[:, 1].view(B, -1)
                 except RuntimeError:
                     print(f"!!! [Worker] Non-uniform group {gid} size detected. Fallback to ragged construction.")
                     # Fallback to naive
                     continue

                 # Add Offset & Pre-tokens
                 patch_indices = spatial_indices + num_pretokens
                 pre_indices = torch.arange(num_pretokens, device=self.device).unsqueeze(0).expand(B, -1)
                 dindice = torch.cat([pre_indices, patch_indices], dim=1)
                 
                 context['cached_dindices'][gid] = dindice

        elif op == OpType.PREPARE_TOKENS:
            if 'input_tensor' not in context: return

            tensor = context['input_tensor']
             
            if hasattr(backbone, 'prepare_tokens_with_masks'):
                t2_x, hw_tuple = backbone.prepare_tokens_with_masks(tensor, None)
                context['input_tokens'] = t2_x
                context['hw_tuple'] = hw_tuple
                
                if 'current_feature' not in context:
                    context['current_feature'] = t2_x

                if self.model.backbone.rope_embed is not None and 'rope_sincos' not in context:
                    rope_sincos = self.model.backbone.rope_embed(H=hw_tuple[0], W=hw_tuple[1])
                    context['rope_sincos'] = rope_sincos

        elif op == OpType.SEND_RESPONSE:
            if 'final_results' in context:
                 # Reconstruct ordered list
                 final_map = context['final_results']
                 preds = []
                 for i in range(self.config.batch_size):
                     preds.append(final_map.get(i, [])) # Return empty list if missing (fallback)
            elif 'output' in context:
                output_logits = context['output']
                _, top5 = torch.topk(output_logits, k=5, dim=1)
                preds = top5.cpu().numpy().tolist()
            else:
                preds = []
            
            # Include accumulated server events
            server_events = context.get('events', [])
            result = InferenceResult(task.task_id, time.time(), preds, server_events)
            self.output_queue.put(result)

        elif op == OpType.FREE_SESSION:
            if task.request_id in self.sessions:
                del self.sessions[task.request_id]

        # --- Computation Ops ---
        elif op == OpType.FULL_INFERENCE:
            inp = context.get('input_tensor')
            context['output'] = self.model(inp)

        elif op == OpType.APPROX_FORWARD:
            # Get params
            layers = params.get('layers', range(0, 40))

            # Approx Loop
            x_feature = context['current_feature']
            rope_sincos = context['rope_sincos']
            cache = context.get('cache_feature', {})
            
            start_l, end_l = layers[0], layers[1]
            
            # Ensure proper input if starting from 0
            if start_l == 0:
                x_feature = context['input_tokens']

            for lidx in range(start_l, end_l):
                blk = self.model.backbone.blocks[lidx]
                
                x_feature, cache = blk.approx(
                    x_feature, rope_sincos, cache, tag=f"layer{lidx}",
                    debug=False
                )
            
            # Update Context
            context['current_feature'] = x_feature
            context['cache_feature'] = cache

        elif op == OpType.CORRECT_FORWARD:
            # Get params
            layers = params.get('layers', range(0, 40))
            group_id = params.get('group_id', 1)
            
            start_l, end_l = layers[0], layers[1]

            # Prepare Update Indices
            # Optimization: Retrieve cached dindice
            if 'cached_dindices' in context and group_id in context['cached_dindices']:
                dindice = context['cached_dindices'][group_id]
            else:
                 # Fallback (Should typically be cached)
                 # Reconstruct from group_map if possible or error out
                 print(f"!!! [Worker] Warning: dindice cache miss for group {group_id}. Recomputing...")
                 # Attempt to compute from group_map if available
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
                          # Cache it for next time
                          if 'cached_dindices' not in context: context['cached_dindices'] = {}
                          context['cached_dindices'][group_id] = dindice
                      except:
                          print("!!! [Worker] Failed to recompute dindice from map. Using expensive fallback.")
                          # Original fallback...
                          pass
            

            # Correct Loop
            cache = context.get('cache_feature', {})
            rope_sincos = context['rope_sincos']
            
            x_temp = context['input_tokens']
            
            # Get ratio from model config
            alive_ratio = getattr(self.model.backbone, 'appcorr_cls_alive_ratio', 0.2)

            for lidx in range(start_l, end_l):
                blk = self.model.backbone.blocks[lidx]
                
                x_temp, cache = blk.correct(
                    x_temp, dindice, rope_sincos, cache, tag=f"layer{lidx}",
                    cls_alive_ratio=alive_ratio,
                    debug=False
                )
            
            # Update Context
            context['current_feature'] = x_temp
            context['cache_feature'] = cache

        elif op == OpType.HEAD_INFERENCE:
            # Final Feature to Head
            feat = context.get('current_feature')
            
            # Apply Norm
            x_norm = backbone.norm(feat)
            x_norm_clstoken = x_norm[:, 0]
            x_norm_patchtokens = x_norm[:, backbone.n_storage_tokens + 1 :]
            
            # Simulating Classifier Wrapper
            linear_input = torch.cat(
                [x_norm_clstoken, x_norm_patchtokens.mean(dim=1)],
                dim=1,
            )
            context['output'] = self.model.linear_head(linear_input)

        elif op == OpType.DECIDE_EXIT:
            if 'output' not in context: return

            # Get Criteria
            metric = self.config.early_exit_kwargs.get('metric', 'max_prob')
            threshold = self.config.early_exit_kwargs.get('threshold', 0.9)
            
            output_logits = context['output']
            probs = torch.softmax(output_logits, dim=1)
            
            # Compute Top-5 for potential exits
            _, top5 = torch.topk(output_logits, k=5, dim=1)
            
            # Identify Exits
            if metric == 'max_prob':
                max_probs, _ = torch.max(probs, dim=1)
                exit_mask = max_probs >= threshold
            elif metric == 'entropy':
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                exit_mask = entropy <= threshold
            else:
                # Default/Fallback
                max_probs, _ = torch.max(probs, dim=1)
                exit_mask = torch.zeros_like(max_probs, dtype=torch.bool)
                
            num_exits = exit_mask.sum().item()
            if num_exits > 0:
                print(f"[Worker] exiting {num_exits} samples (metric {metric}, thresh {threshold})")
                
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
                
                # Check if all exited
                if keep_mask.sum() == 0:
                    context['active_indices'] = torch.empty(0, device=self.device, dtype=torch.long)
                    if 'current_feature' in context: del context['current_feature']
                    if 'input_tokens' in context: del context['input_tokens']
                else:
                    self._slice_state(context, keep_mask)
            
            return {
                'num_exits': num_exits,
                'threshold': threshold
            }

        elif op == OpType.EXIT_ALL:
             # Flush everything remaining to final_results
             if 'output' in context and 'active_indices' in context:
                output_logits = context['output']
                # Get Top-5
                _, top5 = torch.topk(output_logits, k=5, dim=1)
                
                current_active_indices = context['active_indices']
                
                if 'final_results' not in context:
                    context['final_results'] = {}
                    
                preds_list = top5.cpu().numpy().tolist()
                for orig_idx, pred_list in zip(current_active_indices.cpu().numpy(), preds_list):
                    context['final_results'][int(orig_idx)] = pred_list
            
             # Clear state
             context['active_indices'] = torch.empty(0, device=self.device, dtype=torch.long)

    def _slice_state(self, context: Dict[str, Any], keep_mask: torch.Tensor):
        """Slices all relevant tensors in context based on keep_mask."""
        
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
        # Update in-place to save memory
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
                
        # Clear derived structures that depend on batch index/size
        if 'cached_dindices' in context:
            context['cached_dindices'] = {}

        # Output (Logits)
        if 'output' in context:
            context['output'] = context['output'][keep_mask]

        