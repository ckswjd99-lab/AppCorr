from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from offload.common import ExperimentConfig
from offload.server.model.dinov3_classifier import DINOv3ClassifierExecutor


class Dinov3SignalProbe:
    def __init__(self, device: torch.device, image_size: int, layers: list[int] | None = None):
        self.device = device
        self.image_size = image_size
        self.layers = None if layers is None else set(layers)
        self.executor = DINOv3ClassifierExecutor(device)
        self.config = ExperimentConfig(
            model_name="dinov3_classifier",
            image_shape=(image_size, image_size, 3),
            patch_size=(16, 16),
        )
        self.executor.load_model("dinov3_classifier", self.config)
        self.model = self.executor.model
        self.backbone = self.model.backbone
        self.patch_start = 1 + self.backbone.n_storage_tokens

    def _autocast_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def _prepare_input(self, batch_bchw_uint8: torch.Tensor) -> torch.Tensor:
        tensor = batch_bchw_uint8.to(device=self.device, non_blocking=True).float() / 255.0
        tensor = (tensor - self.executor.norm_mean) / self.executor.norm_std
        return tensor

    def _collect_ffn_signals(self, block: Any, x_norm2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mlp = block.mlp
        if hasattr(mlp, "w1") and hasattr(mlp, "w2") and hasattr(mlp, "w3"):
            gate = F.silu(mlp.w1(x_norm2))
            value = mlp.w2(x_norm2)
            effective = gate * value
            gate_score = gate.abs().mean(dim=-1)
            effective_gate_score = effective.abs().mean(dim=-1)
            mlp_out = mlp.w3(effective)
            return mlp_out, gate_score, effective_gate_score

        hidden = mlp.fc1(x_norm2)
        activated = mlp.act(hidden)
        if hasattr(mlp, "drop"):
            activated = mlp.drop(activated)
        mlp_out = mlp.fc2(activated)
        if hasattr(mlp, "drop"):
            mlp_out = mlp.drop(mlp_out)
        gate_score = activated.abs().mean(dim=-1)
        return mlp_out, gate_score, gate_score

    @torch.no_grad()
    def run(self, batch_bchw_uint8: torch.Tensor) -> dict[int, dict[str, torch.Tensor]]:
        signal_by_layer: dict[int, dict[str, torch.Tensor]] = {}
        tensor = self._prepare_input(batch_bchw_uint8)

        with self._autocast_context():
            x, hw_tuple = self.backbone.prepare_tokens_with_masks(tensor, None)
            rope = self.backbone.rope_embed(H=hw_tuple[0], W=hw_tuple[1]) if self.backbone.rope_embed is not None else None

            for layer_idx, block in enumerate(self.backbone.blocks):
                x_norm1 = block.norm1(x)
                qkv = block.attn.qkv(x_norm1)
                batch, num_tokens, _ = qkv.shape
                embed_dim = block.attn.qkv.in_features
                qkv = qkv.reshape(batch, num_tokens, 3, block.attn.num_heads, embed_dim // block.attn.num_heads)
                q, k, v = torch.unbind(qkv, dim=2)
                q, k, v = [tensor_.transpose(1, 2) for tensor_ in (q, k, v)]

                if rope is not None:
                    q, k = block.attn.apply_rope(q, k, rope)

                attn_logits = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn_probs = torch.softmax(attn_logits.float(), dim=-1)
                attn_output = attn_probs.to(dtype=v.dtype) @ v
                attn_output = attn_output.transpose(1, 2).reshape(batch, num_tokens, embed_dim)
                attn_output = block.attn.proj(attn_output)
                attn_output = block.attn.proj_drop(attn_output)

                x_attn = x + block.ls1(attn_output)

                x_norm2 = block.norm2(x_attn)
                mlp_out_raw, gate_score, effective_gate_score = self._collect_ffn_signals(block, x_norm2)
                x = x_attn + block.ls2(mlp_out_raw)

                if self.layers is None or layer_idx in self.layers:
                    signal_by_layer[layer_idx] = {
                        "attn_mean": attn_probs.mean(dim=1).cpu(),
                        "gate_score": gate_score.float().cpu(),
                        "effective_gate_score": effective_gate_score.float().cpu(),
                    }

        return signal_by_layer
