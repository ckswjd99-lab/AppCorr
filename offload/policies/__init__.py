from typing import Optional

from offload.common.protocol import ExperimentConfig
from .interface import ISchedulingPolicy, ITransmissionPolicy
from .scheduling import (
    ADE20KApproxCorrectPolicy,
    ADE20KInterleavedDynamicPolicy,
    ADE20KSequentialPolicy,
    ADE20KWindowInterleavedPolicy,
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
    ADE20KWindowProgressiveLaplacianPolicy,
    LaplacianPyramidPolicy,
    NYUAppCorrLaplacianPolicy,
    NYUAppCorrProgressiveLaplacianPolicy,
    NYUAppCorrRawTransmissionPolicy,
    ProgressiveLPyramidPolicy,
    FullImageCompressionPolicy,
    FourierProgressiveTransmissionPolicy,
    FourierLaplacianHybridPolicy,
    FourierLaplacianProgressivePolicy,
    NYUAppCorrFourierLaplacianHybridPolicy,
)

# Registry for dynamic instantiation
SCHEDULER_REGISTRY = {
    "ADE20KApproxCorrect": ADE20KApproxCorrectPolicy,
    "ADE20KInterleavedDynamic": ADE20KInterleavedDynamicPolicy,
    "ADE20KSequential": ADE20KSequentialPolicy,
    "NYUApproxCorrect": NYUApproxCorrectPolicy,
    "BatchCountBased": BatchCountBasedPolicy,
    "GroupTrigger": GroupTriggerPolicy,
    "DynamicGroupTrigger": DynamicGroupTriggerPolicy,
    "COCOWindowInterleaved": COCOWindowInterleavedPolicy,
    "COCOWindowDynamic": COCOWindowDynamicPolicy,
    "ADE20KWindowInterleaved": ADE20KWindowInterleavedPolicy,
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
    "ADE20KWindowProgressiveLaplacian": ADE20KWindowProgressiveLaplacianPolicy,
    "FullImageCompression": FullImageCompressionPolicy,
    "FourierProgressive": FourierProgressiveTransmissionPolicy,
    "FourierLaplacianHybrid": FourierLaplacianHybridPolicy,
    "FourierLaplacianProgressive": FourierLaplacianProgressivePolicy,
    "NYUAppCorrFourierLaplacianHybrid": NYUAppCorrFourierLaplacianHybridPolicy,
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
