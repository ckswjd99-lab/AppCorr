from abc import ABC, abstractmethod
from typing import List, Optional, Any, Generator
import numpy as np

from offload.common.protocol import Patch, Task, ExperimentConfig

class ISchedulingPolicy(ABC):
    """Interface for server-side scheduling logic."""
    
    @abstractmethod
    def decide(
        self, 
        buffer: List[Patch], 
        config: ExperimentConfig,
        task_id_gen: Any
    ) -> Optional[Task]:
        """Determine if a task should be dispatched based on buffer status."""
        pass

class ITransmissionPolicy(ABC):
    """Interface for image codec and packetization."""
    
    @abstractmethod
    def encode(self, image: np.ndarray, config: ExperimentConfig) -> List[Patch]:
        """[Mobile] Compress/Split image into patches."""
        pass

    @abstractmethod
    def decode(self, patches: List[Patch], config: ExperimentConfig) -> np.ndarray:
        """[Server] Reconstruct image/tensor from patches."""
        pass