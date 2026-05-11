from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist


def is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_enabled() else 1


def get_rank() -> int:
    return dist.get_rank() if is_enabled() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def is_subgroup_main_process() -> bool:
    return is_main_process()


def gather_all_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    if not is_enabled():
        return [tensor]

    world_size = get_world_size()
    local_shape = torch.tensor(tensor.shape, device=tensor.device, dtype=torch.long)
    all_shapes = [torch.empty_like(local_shape) for _ in range(world_size)]
    dist.all_gather(all_shapes, local_shape)

    max_shape = torch.stack(all_shapes).max(dim=0).values
    if tuple(local_shape.tolist()) == tuple(max_shape.tolist()):
        padded = tensor.contiguous()
    else:
        padded = tensor.new_zeros(tuple(max_shape.tolist()))
        slices = tuple(slice(0, dim) for dim in tensor.shape)
        padded[slices] = tensor

    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    outputs = []
    for gathered_tensor, shape in zip(gathered, all_shapes):
        slices = tuple(slice(0, int(dim)) for dim in shape.tolist())
        outputs.append(gathered_tensor[slices].contiguous())
    return outputs
