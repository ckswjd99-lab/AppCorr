from .interface import ISchedulingPolicy, ITransmissionPolicy
from .scheduling import BatchCountBasedPolicy, GroupTriggerPolicy, GroupTriggerEarlyExitPolicy
from .transmission import RawTransmissionPolicy, ZlibTransmissionPolicy, LaplacianPyramidPolicy, ProgressiveLPyramidPolicy, FullImageCompressionPolicy

# Registry for dynamic instantiation
SCHEDULER_REGISTRY = {
    "BatchCountBased": BatchCountBasedPolicy,
    "GroupTrigger": GroupTriggerPolicy,
    "GroupTriggerEarlyExit": GroupTriggerEarlyExitPolicy,
}

TRANSMISSION_REGISTRY = {
    "Raw": RawTransmissionPolicy,
    "Zlib": ZlibTransmissionPolicy,
    "Laplacian": LaplacianPyramidPolicy,
    "ProgressiveLaplacian": ProgressiveLPyramidPolicy,
    "FullImageCompression": FullImageCompressionPolicy,
}

def get_scheduler(name: str) -> ISchedulingPolicy:
    return SCHEDULER_REGISTRY.get(name, BatchCountBasedPolicy)()

def get_transmission(name: str) -> ITransmissionPolicy:
    return TRANSMISSION_REGISTRY.get(name, RawTransmissionPolicy)()