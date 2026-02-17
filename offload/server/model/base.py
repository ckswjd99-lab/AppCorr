from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import torch
import numpy as np
from offload.common import Task

class ModelExecutor(ABC):
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None

    @abstractmethod
    def load_model(self, model_name: str, config: Any):
        pass

    @abstractmethod
    def preprocess(self, batch_np: np.ndarray, task: Task, context: Dict[str, Any], config: Any):
        """Handles OpType.LOAD_INPUT logic (decoding done by listener, normalization here)"""
        pass

    @abstractmethod
    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        """Handles OpType.PREPARE_TOKENS"""
        pass

    @abstractmethod
    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        """Handles OpType.APPROX_FORWARD"""
        pass

    @abstractmethod
    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        """Handles OpType.CORRECT_FORWARD"""
        pass

    @abstractmethod
    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        """Handles OpType.FULL_INFERENCE"""
        pass

    @abstractmethod
    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        """Handles OpType.HEAD_INFERENCE"""
        pass

    @abstractmethod
    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        """Handles OpType.DECIDE_EXIT"""
        pass

    @abstractmethod
    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        """
        Returns the final formatted results for the current batch state.
        Should return a dictionary mapping original request index to the result payload.
        Used by the worker to populate the response.
        """
        pass
