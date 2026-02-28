from typing import Any, Dict, List
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
import math
import numpy as np
from offload.common import Task
from .base import ModelExecutor
from .utils import load_weight_mmap

from appcorr.models.dinov3.eval.detection.util.misc import nested_tensor_from_tensor_list, NestedTensor
from appcorr.models.dinov3.eval.detection.util import box_ops

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
        self.model = dinov3_vit7b16_de(pretrained=False, weights="COCO2017", backbone_weights="LVD1689M")
        self.model.to(dtype=torch.bfloat16, device=self.device)
        
        try:
             backbone_path = "~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
             print(f"[Executor] Loading Backbone Weights from {backbone_path}")
             vit_backbone = self._get_vit_backbone()
             vit_backbone.load_state_dict(load_weight_mmap(backbone_path), strict=True)
             
             head_path = "~/cjpark/weights/dinov3/dinov3_vit7b16_coco_detr_head-b0235ff7.pth"
             print(f"[Executor] Loading Detector Head Weights from {head_path}")
             head_ckpt = load_weight_mmap(head_path)
             self.model.detector.load_state_dict(head_ckpt.get("model", head_ckpt), strict=False)
             del head_ckpt
        except Exception as e:
            print(f"!!! [Executor] Failed to load detector weights: {e}")
            raise e
            
        self.model.eval()
        
    def _get_vit_backbone(self):
        inner = self.model.detector.backbone[0]
        return inner._backbone.backbone if hasattr(inner, "_backbone") else inner.backbone

    def preprocess(self, batch_np: np.ndarray, task: Task, context: Dict[str, Any], config: Any):
        tensor = torch.from_numpy(batch_np).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        tensor = (tensor - self.norm_mean) / self.norm_std
        idx = context.get('active_indices')
        if idx is not None and len(idx) < config.batch_size:
            tensor = tensor[idx]
        context['input_tensor'] = tensor

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        if 'input_tensor' not in context: return {}

        input_tensor = context['input_tensor']
        context['sizes_tensor'] = torch.tensor([s.shape[1:] for s in input_tensor], device=input_tensor[0].device)
        
        if not isinstance(input_tensor, NestedTensor):
            input_tensor = nested_tensor_from_tensor_list(input_tensor)

        detector = self.model.detector
        win_wrapper = detector.backbone[0]
        dino_bb = win_wrapper._backbone.backbone
        tensors, mask = input_tensor.tensors, input_tensor.mask
        context['input_tensor_mask'] = mask
        
        orig_h, orig_w = tensors.shape[2:]
        win_h = math.ceil((orig_h // win_wrapper._n_windows_h) / win_wrapper._patch_size) * win_wrapper._patch_size
        win_w = math.ceil((orig_w // win_wrapper._n_windows_w) / win_wrapper._patch_size) * win_wrapper._patch_size
        
        all_h = [win_h] * (win_wrapper._n_windows_h - 1) + [orig_h - win_h * (win_wrapper._n_windows_h - 1)]
        all_w = [win_w] * (win_wrapper._n_windows_w - 1) + [orig_w - win_w * (win_wrapper._n_windows_w - 1)]
        all_h_cum, all_w_cum = [0] + list(np.cumsum(all_h)), [0] + list(np.cumsum(all_w))

        context['window_patch_tensors'], context['window_patch_masks'], context['window_patch_tokens'] = [], [], []
        all_x_backbones, all_rope_sincos = [], []

        def _prep(x):
            xb, (H, W) = dino_bb.prepare_tokens_with_masks(x)
            rs = dino_bb.rope_embed(H=H, W=W) if dino_bb.rope_embed else None
            return xb, rs

        for ih in range(win_wrapper._n_windows_h):
            row_t, row_m, row_x = [], [], []
            for iw in range(win_wrapper._n_windows_w):
                wt = v2.functional.crop(tensors, all_h_cum[ih], all_w_cum[iw], all_h[ih], all_w[iw])
                wm = v2.functional.crop(mask, all_h_cum[ih], all_w_cum[iw], all_h[ih], all_w[iw])
                x = NestedTensor(wt, wm).tensors
                
                row_t.append(wt); row_m.append(wm); row_x.append(x)
                xb, rs = _prep(x)
                all_x_backbones.append(xb); all_rope_sincos.append(rs)
                
            context['window_patch_tensors'].append(row_t)
            context['window_patch_masks'].append(row_m)
            context['window_patch_tokens'].append(row_x)

        context['global_x'] = NestedTensor(v2.functional.resize(tensors, size=(win_h, win_w)), mask).tensors
        g_xb, g_rs = _prep(context['global_x'])
        all_x_backbones.append(g_xb); all_rope_sincos.append(g_rs)

        context['all_x_backbones'], context['all_rope_sincos'] = all_x_backbones, all_rope_sincos

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        pass

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        dino_backbone = self.model.detector.backbone[0]._backbone
        all_x_backbones = context.get('all_x_backbones')
        all_rope_sincos = context.get('all_rope_sincos')
        n, blocks = dino_backbone.layers_to_use, dino_backbone.backbone.blocks
        blocks_to_take = range(len(blocks) - n, len(blocks)) if isinstance(n, int) else n

        all_outputs = []
        for x_backbone, rope_sincos in zip(all_x_backbones, all_rope_sincos):
            output = []
            for i, blk in enumerate(blocks):
                x_backbone = blk(x_backbone, rope_sincos)
                if i in blocks_to_take:
                    output.append(x_backbone)
            all_outputs.append(output)

        context['all_outputs'] = all_outputs

    def _process_outputs(self, outputs, x, dt_backbone):
        normed = []
        for out in outputs:
            if dt_backbone.backbone.untie_cls_and_patch_norms:
                cls_reg = dt_backbone.backbone.cls_norm(out[:, : dt_backbone.backbone.n_storage_tokens + 1])
                patch = dt_backbone.backbone.norm(out[:, dt_backbone.backbone.n_storage_tokens + 1 :])
                normed.append(torch.cat((cls_reg, patch), dim=1))
            else:
                normed.append(dt_backbone.backbone.norm(out))
        outputs = [out[:, dt_backbone.backbone.n_storage_tokens + 1 :] for out in normed]

        B, _, h, w = x.shape
        ps = dt_backbone.backbone.patch_size
        xs = [o.reshape(B, h // ps, w // ps, -1).permute(0, 3, 1, 2).contiguous() for o in outputs]
        
        if dt_backbone.use_layernorm:
            xs = [ln(x).contiguous() for ln, x in zip(dt_backbone.layer_norms, xs)]
        return torch.cat(xs, dim=1)

    def _run_transformer_and_postprocess(self, detector, features, sizes_tensor):
        pos = [detector.backbone[1][idx](x).to(x.tensors.dtype) for idx, x in enumerate(features)]
        srcs, masks = [], []
        for layer, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(detector.input_proj[layer](src))
            masks.append(mask)

        query_embeds = detector.query_embed.weight[:detector.num_queries] if not detector.two_stage or detector.mixed_selection else None
        
        self_attn_mask = torch.zeros([detector.num_queries, detector.num_queries], dtype=bool, device=src.device)
        self_attn_mask[detector.num_queries_one2one :, : detector.num_queries_one2one] = True
        self_attn_mask[: detector.num_queries_one2one, detector.num_queries_one2one :] = True

        (hs, init_ref, inter_refs, enc_out_cls, enc_out_coord, enc_out_delta, prop, max_shape) = detector.transformer(
            srcs, masks, pos, query_embeds, self_attn_mask
        )

        o_cls_1, o_coords_1, o_cls_m, o_coords_m = [], [], [], []
        o_coords_old_1, o_deltas_1, o_coords_old_m, o_deltas_m = [], [], [], []
        n1 = detector.num_queries_one2one

        for lvl in range(hs.shape[0]):
            ref = init_ref if lvl == 0 else inter_refs[lvl - 1]
            out_class = detector.class_embed[lvl](hs[lvl])
            tmp = detector.bbox_embed[lvl](hs[lvl])
            out_coord = box_ops.box_xyxy_to_cxcywh(box_ops.delta2bbox(ref, tmp, max_shape))

            o_cls_1.append(out_class[:, :n1]); o_cls_m.append(out_class[:, n1:])
            o_coords_1.append(out_coord[:, :n1]); o_coords_m.append(out_coord[:, n1:])
            o_coords_old_1.append(ref[:, :n1]); o_coords_old_m.append(ref[:, n1:])
            o_deltas_1.append(tmp[:, :n1]); o_deltas_m.append(tmp[:, n1:])

        out = {
            "pred_logits": torch.stack(o_cls_1)[-1], "pred_boxes": torch.stack(o_coords_1)[-1],
            "pred_logits_one2many": torch.stack(o_cls_m)[-1], "pred_boxes_one2many": torch.stack(o_coords_m)[-1],
            "pred_boxes_old": o_coords_old_1[-1], "pred_deltas": o_deltas_1[-1],
            "pred_boxes_old_one2many": o_coords_old_m[-1], "pred_deltas_one2many": o_deltas_m[-1],
        }

        if detector.aux_loss:
            out["aux_outputs"] = detector._set_aux_loss(o_cls_1, o_coords_1, o_coords_old_1, o_deltas_1)
            out["aux_outputs_one2many"] = detector._set_aux_loss(o_cls_m, o_coords_m, o_coords_old_m, o_deltas_m)

        if detector.two_stage:
            out["enc_outputs"] = {
                "pred_logits": enc_out_cls, "pred_boxes": enc_out_coord, "pred_boxes_old": prop, "pred_deltas": enc_out_delta
            }
        
        return self.model.postprocessor(out, target_sizes=sizes_tensor, original_target_sizes=sizes_tensor)

    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        detector = self.model.detector
        win_wrapper = detector.backbone[0]
        dino_bb = win_wrapper._backbone

        all_outputs = context.get('all_outputs')
        global_output, window_outputs = all_outputs[-1], all_outputs[:-1]

        win_patch_tensors, win_patch_masks = context.get('window_patch_tensors'), context.get('window_patch_masks')
        win_patch_tokens = context.get('window_patch_tokens')

        win_patch_features = []
        idx = 0
        for ih in range(win_wrapper._n_windows_h):
            row_features = []
            for iw in range(win_wrapper._n_windows_w):
                x = self._process_outputs(window_outputs[idx], win_patch_tokens[ih][iw], dino_bb)
                m = NestedTensor(win_patch_tensors[ih][iw], win_patch_masks[ih][iw]).mask
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                row_features.append(NestedTensor(x, mask))
                idx += 1
            win_patch_features.append(row_features)

        window_features = torch.cat([torch.cat([el.tensors for el in row], dim=-1) for row in win_patch_features], dim=-2)
        global_x_tensor = self._process_outputs(global_output, context.get('global_x'), dino_bb)
        
        input_tensor_mask = context.get('input_tensor_mask')
        
        concat_tensors = torch.cat([v2.functional.resize(global_x_tensor, size=window_features.shape[-2:]), window_features], dim=1)
        global_mask = F.interpolate(input_tensor_mask[None].float(), size=concat_tensors.shape[-2:]).to(torch.bool)[0]
        features = [NestedTensor(tensors=concat_tensors, mask=global_mask)]

        context['det_outputs'] = self._run_transformer_and_postprocess(detector, features, context.get('sizes_tensor'))
        return {}

    @torch.inference_mode()
    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        if 'input_tensor' not in context: return {}
        self.prepare_tokens(task, context, config)
        self.correct_forward({}, context, config)
        self.head_inference(task, context, config)
        return {}

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        results = {}
        if 'det_outputs' in context and 'active_indices' in context:
             outputs, indices = context.get('det_outputs'), context.get('active_indices').cpu().numpy()
             for i, orig_idx in enumerate(indices):
                 if i < len(outputs):
                     pred = outputs[i]
                     results[int(orig_idx)] = {
                         'scores': pred['scores'].float().cpu().tolist(),
                         'labels': pred['labels'].long().cpu().tolist(),
                         'boxes': pred['boxes'].float().cpu().tolist()
                     }
        return results

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return {'num_exits': 0}

    def _full_inference_analyzed(self, input_tensor: list[torch.Tensor]):
        detector = self.model.detector

        sizes_tensor = torch.tensor([sample.shape[1:] for sample in input_tensor], device=input_tensor[0].device)  # N * [3, H, W]
        
        if not isinstance(input_tensor, NestedTensor):
            input_tensor = nested_tensor_from_tensor_list(input_tensor)
            
        windows_wrapper = detector.backbone[0]
        tensors = input_tensor.tensors
        original_h, original_w = tensors.shape[2], tensors.shape[3]
        # Get height and width of the windows, such that it is a multiple of the patch size
        window_h = math.ceil((original_h // windows_wrapper._n_windows_h) / windows_wrapper._patch_size) * windows_wrapper._patch_size
        window_w = math.ceil((original_w // windows_wrapper._n_windows_w) / windows_wrapper._patch_size) * windows_wrapper._patch_size
        all_h = [window_h] * (windows_wrapper._n_windows_h - 1) + [original_h - window_h * (windows_wrapper._n_windows_h - 1)]
        all_w = [window_w] * (windows_wrapper._n_windows_w - 1) + [original_w - window_w * (windows_wrapper._n_windows_w - 1)]
        all_h_cumsum = [0] + list(np.cumsum(all_h))
        all_w_cumsum = [0] + list(np.cumsum(all_w))
        window_patch_features = [[0 for _ in range(windows_wrapper._n_windows_w)] for _ in range(windows_wrapper._n_windows_h)]

        for ih in range(windows_wrapper._n_windows_h):
            for iw in range(windows_wrapper._n_windows_w):
                window_tensor = v2.functional.crop(
                    tensors, top=all_h_cumsum[ih], left=all_w_cumsum[iw], height=all_h[ih], width=all_w[iw]
                )
                window_mask = v2.functional.crop(
                    input_tensor.mask, top=all_h_cumsum[ih], left=all_w_cumsum[iw], height=all_h[ih], width=all_w[iw]
                )

                dino_backbone = windows_wrapper._backbone
                x = NestedTensor(tensors=window_tensor, mask=window_mask).tensors
                n = dino_backbone.layers_to_use

                x_backbone = x
                x_backbone, (H, W) = dino_backbone.backbone.prepare_tokens_with_masks(x_backbone)
                # If n is an int, take the n last blocks. If it's a list, take them
                output, total_block_len = [], len(dino_backbone.backbone.blocks)
                blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
                for i, blk in enumerate(dino_backbone.backbone.blocks):
                    if dino_backbone.backbone.rope_embed is not None:
                        rope_sincos = dino_backbone.backbone.rope_embed(H=H, W=W)
                    else:
                        rope_sincos = None
                    x_backbone = blk(x_backbone, rope_sincos)
                    if i in blocks_to_take:
                        output.append(x_backbone)
                assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
                outputs = output

                outputs_normed = []
                for out in outputs:
                    if dino_backbone.backbone.untie_cls_and_patch_norms:
                        x_norm_cls_reg = dino_backbone.backbone.cls_norm(out[:, : dino_backbone.backbone.n_storage_tokens + 1])
                        x_norm_patch = dino_backbone.backbone.norm(out[:, dino_backbone.backbone.n_storage_tokens + 1 :])
                        outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                    else:
                        outputs_normed.append(dino_backbone.backbone.norm(out))
                outputs = [out[:, dino_backbone.backbone.n_storage_tokens + 1 :] for out in outputs_normed]

                B, _, h, w = x.shape
                xs = [
                    out.reshape(B, h // dino_backbone.backbone.patch_size, w // dino_backbone.backbone.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                    for out in outputs
                ]

                if dino_backbone.use_layernorm:
                    xs = [ln(x).contiguous() for ln, x in zip(dino_backbone.layer_norms, xs)]

                x = torch.cat(xs, axis=1)

                m = NestedTensor(tensors=window_tensor, mask=window_mask).mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                window_patch_features[ih][iw] = NestedTensor(x, mask)

        window_tensors = torch.cat(
            [
                torch.cat([el.tensors for el in window_patch_features[ih]], dim=-1)  # type: ignore
                for ih in range(len(window_patch_features))
            ],
            dim=-2,
        )
        # Also compute the global features in a "preferential" setting, of lower resolution
        resized_global_tensor = v2.functional.resize(tensors, size=(window_h, window_w))
        global_features = windows_wrapper._backbone(
            NestedTensor(tensors=resized_global_tensor, mask=input_tensor.mask)
        )  # mask is not used

        concat_tensors = torch.cat(
            [v2.functional.resize(global_features[0].tensors, size=window_tensors.shape[-2:]), window_tensors], dim=1
        )
        global_mask = F.interpolate(input_tensor.mask[None].float(), size=concat_tensors.shape[-2:]).to(torch.bool)[0]
        features = [NestedTensor(tensors=concat_tensors, mask=global_mask)]

        pos = [detector.backbone[1][idx](x).to(x.tensors.dtype) for idx, x in enumerate(features)]

        srcs = []
        masks = []
        for layer, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(detector.input_proj[layer](src))
            masks.append(mask)
            assert mask is not None

        query_embeds = None
        if not detector.two_stage or detector.mixed_selection:
            query_embeds = detector.query_embed.weight[0 : detector.num_queries, :]

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = torch.zeros(
            [
                detector.num_queries,
                detector.num_queries,
            ],
            dtype=bool,
            device=src.device,
        )
        self_attn_mask[
            detector.num_queries_one2one :,
            0 : detector.num_queries_one2one,
        ] = True
        self_attn_mask[
            0 : detector.num_queries_one2one,
            detector.num_queries_one2one :,
        ] = True

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
            max_shape,
        ) = detector.transformer(srcs, masks, pos, query_embeds, self_attn_mask)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []

        outputs_coords_old_one2one = []
        outputs_deltas_one2one = []
        outputs_coords_old_one2many = []
        outputs_deltas_one2many = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            outputs_class = detector.class_embed[lvl](hs[lvl])
            tmp = detector.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                outputs_coord = box_ops.box_xyxy_to_cxcywh(box_ops.delta2bbox(reference, tmp, max_shape))
            else:
                raise NotImplementedError

            outputs_classes_one2one.append(outputs_class[:, 0 : detector.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, detector.num_queries_one2one :])

            outputs_coords_one2one.append(outputs_coord[:, 0 : detector.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, detector.num_queries_one2one :])

            outputs_coords_old_one2one.append(reference[:, : detector.num_queries_one2one])
            outputs_coords_old_one2many.append(reference[:, detector.num_queries_one2one :])
            outputs_deltas_one2one.append(tmp[:, : detector.num_queries_one2one])
            outputs_deltas_one2many.append(tmp[:, detector.num_queries_one2one :])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
            "pred_boxes_old": outputs_coords_old_one2one[-1],
            "pred_deltas": outputs_deltas_one2one[-1],
            "pred_boxes_old_one2many": outputs_coords_old_one2many[-1],
            "pred_deltas_one2many": outputs_deltas_one2many[-1],
        }

        if detector.aux_loss:
            out["aux_outputs"] = detector._set_aux_loss(
                outputs_classes_one2one, outputs_coords_one2one, outputs_coords_old_one2one, outputs_deltas_one2one
            )
            out["aux_outputs_one2many"] = detector._set_aux_loss(
                outputs_classes_one2many, outputs_coords_one2many, outputs_coords_old_one2many, outputs_deltas_one2many
            )

        if detector.two_stage:
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord_unact,
                "pred_boxes_old": output_proposals,
                "pred_deltas": enc_outputs_delta,
            }
        
        outputs = self.model.postprocessor(out, target_sizes=sizes_tensor, original_target_sizes=sizes_tensor)
        return outputs
        