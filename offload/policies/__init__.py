from typing import Optional

from offload.common.protocol import ExperimentConfig
from .interface import ISchedulingPolicy, ITransmissionPolicy
from .scheduling import (
    ADE20KApproxCorrectPolicy,
    ADE20KSequentialPolicy,
    BatchCountBasedPolicy,
    COCOWindowDynamicPolicy,
    COCOWindowInterleavedPolicy,
    DynamicGroupTriggerPolicy,
    GroupTriggerPolicy,
    NYUApproxCorrectPolicy,
)
from .transmission import (
    RawTransmissionPolicy, 
    ZlibTransmissionPolicy, 
    COCOWindowProgressiveLaplacianPolicy,
    LaplacianPyramidPolicy, 
    NYUAppCorrLaplacianPolicy,
    NYUAppCorrProgressiveLaplacianPolicy,
    NYUAppCorrRawTransmissionPolicy,
    ProgressiveLPyramidPolicy, 
    FullImageCompressionPolicy
)

# Registry for dynamic instantiation
SCHEDULER_REGISTRY = {
    "ADE20KApproxCorrect": ADE20KApproxCorrectPolicy,
    "ADE20KSequential": ADE20KSequentialPolicy,
    "NYUApproxCorrect": NYUApproxCorrectPolicy,
    "BatchCountBased": BatchCountBasedPolicy,
    "GroupTrigger": GroupTriggerPolicy,
    "DynamicGroupTrigger": DynamicGroupTriggerPolicy,
    "COCOWindowInterleaved": COCOWindowInterleavedPolicy,
    "COCOWindowDynamic": COCOWindowDynamicPolicy,
}

TRANSMISSION_REGISTRY = {
    "Raw": RawTransmissionPolicy,
    "NYUAppCorrRaw": NYUAppCorrRawTransmissionPolicy,
    "Zlib": ZlibTransmissionPolicy,
    "Laplacian": LaplacianPyramidPolicy,
    "NYUAppCorrLaplacian": NYUAppCorrLaplacianPolicy,
    "ProgressiveLaplacian": ProgressiveLPyramidPolicy,
    "NYUAppCorrProgressiveLaplacian": NYUAppCorrProgressiveLaplacianPolicy,
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
