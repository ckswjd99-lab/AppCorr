from .interface import ISchedulingPolicy, ITransmissionPolicy
from .scheduling import BatchCountBasedPolicy, GroupTriggerPolicy
from .transmission import RawTransmissionPolicy, ZlibTransmissionPolicy, LaplacianPyramidPolicy, ProgressiveLPyramidPolicy

# Registry for dynamic instantiation
SCHEDULER_REGISTRY = {
    "BatchCountBased": BatchCountBasedPolicy,
    "GroupTrigger": GroupTriggerPolicy,
}

TRANSMISSION_REGISTRY = {
    "Raw": RawTransmissionPolicy,
    "Zlib": ZlibTransmissionPolicy,
    "Laplacian": LaplacianPyramidPolicy,
    "ProgressiveLaplacian": ProgressiveLPyramidPolicy,
}

def get_scheduler(name: str) -> ISchedulingPolicy:
    return SCHEDULER_REGISTRY.get(name, BatchCountBasedPolicy)()

def get_transmission(name: str) -> ITransmissionPolicy:
    return TRANSMISSION_REGISTRY.get(name, RawTransmissionPolicy)()