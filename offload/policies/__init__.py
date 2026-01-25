from .interface import ISchedulingPolicy, ITransmissionPolicy
from .scheduling import BatchCountBasedPolicy
from .transmission import RawTransmissionPolicy

# Registry for dynamic instantiation
SCHEDULER_REGISTRY = {
    "BatchCountBased": BatchCountBasedPolicy,
}

TRANSMISSION_REGISTRY = {
    "Raw": RawTransmissionPolicy,
}

def get_scheduler(name: str) -> ISchedulingPolicy:
    return SCHEDULER_REGISTRY.get(name, BatchCountBasedPolicy)()

def get_transmission(name: str) -> ITransmissionPolicy:
    return TRANSMISSION_REGISTRY.get(name, RawTransmissionPolicy)()