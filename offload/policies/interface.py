from abc import ABC, abstractmethod
from typing import List, Optional, Any, Generator
import numpy as np

from offload.common.protocol import Patch, Task, ExperimentConfig

class ISchedulingPolicy(ABC):
    """Scheduling logic interface."""

    def __init__(self, config: Optional[ExperimentConfig] = None):
        pass
    
    @abstractmethod
    def decide(
        self, 
        buffer: List[Patch], 
        config: ExperimentConfig,
        task_id_gen: Any,
        **kwargs
    ) -> Optional[Task]:
        """Check if task dispatch is ready."""
        pass

class ITransmissionPolicy(ABC):
    """Codec and packetization interface."""
    
    @abstractmethod
    def encode(self, image: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        """Compress and split image into patches."""
        pass

    @abstractmethod
    def decode(self, patches: List[Patch], config: ExperimentConfig) -> np.ndarray:
        """Reconstruct image from patches."""
        pass