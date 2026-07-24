"""
oft_heads.py

Standalone re-definitions of OpenVLA-OFT's continuous-action components -- the L1-regression action
head (MLPResNet) and the proprioception projector -- so their `.pt` checkpoints can be loaded WITHOUT
importing the full `openvla-oft` package (whose top-level `import prismatic` pulls the entire
tensorflow/dlimp training stack). Structures mirror
`openvla-oft/prismatic/models/action_heads.py` and `.../projectors.py` exactly (verified against the
released `action_head--*.pt` / `proprio_projector--*.pt` state dicts: strict load, 0 missing/0
unexpected once the DataParallel `module.` prefix is stripped).

OFT-LIBERO constants: NUM_ACTIONS_CHUNK=8, ACTION_DIM=7, PROPRIO_DIM=8, llm hidden=4096.
"""

from typing import Dict

import torch
import torch.nn as nn

NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8
LLM_DIM = 4096


class MLPResNetBlock(nn.Module):
    """One pre-LN MLP-ResNet block with a residual connection (matches OFT exactly)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x):
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([MLPResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


class L1RegressionActionHead(nn.Module):
    """MLP action head: concatenated per-chunk-step action-token hidden states -> continuous actions."""

    def __init__(self, input_dim: int = LLM_DIM, hidden_dim: int = LLM_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim * ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        # actions_hidden_states: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, hidden) -> (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
        B = actions_hidden_states.shape[0]
        rearranged = actions_hidden_states.reshape(B, NUM_ACTIONS_CHUNK, -1)
        return self.model(rearranged)


class ProprioProjector(nn.Module):
    """Projects low-dim proprioception to one llm-dim token (2-layer MLP, matches OFT)."""

    def __init__(self, llm_dim: int = LLM_DIM, proprio_dim: int = PROPRIO_DIM):
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim)
        self.fc2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, proprio):
        x = self.fc1(proprio)
        x = nn.functional.gelu(x)
        x = self.fc2(x)
        return x


def _strip_module(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k.replace(prefix, "", 1) if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def load_action_head(pt_path: str, device, dtype=torch.bfloat16) -> L1RegressionActionHead:
    head = L1RegressionActionHead()
    sd = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = _strip_module(sd, "module.model.")  # checkpoint keys: module.model.<...> -> <...> under .model
    # our keys are prefixed with "model." -> add it back
    sd = {f"model.{k}": v for k, v in sd.items()}
    missing, unexpected = head.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"action_head load mismatch: missing={missing} unexpected={unexpected}"
    return head.to(device=device, dtype=dtype).eval()


def load_proprio_projector(pt_path: str, device, dtype=torch.bfloat16) -> ProprioProjector:
    proj = ProprioProjector()
    sd = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = _strip_module(sd, "module.")
    missing, unexpected = proj.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"proprio load mismatch: missing={missing} unexpected={unexpected}"
    return proj.to(device=device, dtype=dtype).eval()
