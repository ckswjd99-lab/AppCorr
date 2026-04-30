from typing import Optional

from offload.common.protocol import ExperimentConfig
from .interface import ISchedulingPolicy, ITransmissionPolicy
from .scheduling import (
    ADE20KApproxCorrectPolicy,
    BatchCountBasedPolicy,
    COCOWindowDynamicPolicy,
    COCOWindowInterleavedPolicy,
    DynamicGroupTriggerPolicy,
    GroupTriggerPolicy,
)
from .transmission import (
    RawTransmissionPolicy, 
    ZlibTransmissionPolicy, 
    COCOWindowProgressiveLaplacianPolicy,
    LaplacianPyramidPolicy, 
    ProgressiveLPyramidPolicy, 
    FullImageCompressionPolicy
)

# Registry for dynamic instantiation
SCHEDULER_REGISTRY = {
    "ADE20KApproxCorrect": ADE20KApproxCorrectPolicy,
    "BatchCountBased": BatchCountBasedPolicy,
    "GroupTrigger": GroupTriggerPolicy,
    "DynamicGroupTrigger": DynamicGroupTriggerPolicy,
    "COCOWindowInterleaved": COCOWindowInterleavedPolicy,
    "COCOWindowDynamic": COCOWindowDynamicPolicy,
}

TRANSMISSION_REGISTRY = {
    "Raw": RawTransmissionPolicy,
    "Zlib": ZlibTransmissionPolicy,
    "Laplacian": LaplacianPyramidPolicy,
    "ProgressiveLaplacian": ProgressiveLPyramidPolicy,
    "COCOWindowProgressiveLaplacian": COCOWindowProgressiveLaplacianPolicy,
    "FullImageCompression": FullImageCompressionPolicy,
}

def get_scheduler(name: str, config: Optional[ExperimentConfig] = None) -> ISchedulingPolicy:
    policy_cls = SCHEDULER_REGISTRY.get(name)
    if policy_cls is None:
        raise ValueError(f"Unknown scheduler policy: {name}")
    return policy_cls(config=config)

def get_transmission(name: str) -> ITransmissionPolicy:
    policy_cls = TRANSMISSION_REGISTRY.get(name)
    if policy_cls is None:
        raise ValueError(f"Unknown transmission policy: {name}")
    return policy_cls()
