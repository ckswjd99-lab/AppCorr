import torch
import os
from typing import Any, Dict

def load_weight_mmap(path: str, map_location: str = 'cpu') -> Dict[str, Any]:
    """
    Loads a checkpoint using mmap=True to reduce RAM usage.
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
         raise FileNotFoundError(f"Weight file not found: {path}")
    
    print(f"[Utils] Loading weights from {path} with mmap=True...")
    return torch.load(path, map_location=map_location, mmap=True)

def safe_load(model: torch.nn.Module, state_dict: Dict[str, Any], strict: bool = False, key_prefix: str = ""):
    """
    Loads state_dict into model.
    """
    if key_prefix:
        # Filter and strip prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(key_prefix):
                new_state_dict[k[len(key_prefix):]] = v
        state_dict = new_state_dict

    msg = model.load_state_dict(state_dict, strict=strict)
    print(f"[Utils] Loaded state dict. Msg: {msg}")
    return msg
