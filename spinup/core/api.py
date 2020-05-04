from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import torch


class IAgent(ABC):
    @abstractmethod
    def act(self, *obs, deterministic=False):
        pass

    @abstractmethod
    def reset(self):
        pass


class IAgentModel(IAgent, torch.nn.Module):
    @abstractmethod
    def step(self, *obs_tensors: torch.tensor) -> torch.tensor:
        # takes observation tensors and returns action tensor
        pass

    @abstractmethod
    def reset_for_training(self):
        pass

    @abstractmethod
    def get_context(self) -> Any:
        pass

    @abstractmethod
    def set_context(self, context: Any):
        pass

    def get_device(self):
        return next(iter(self.parameters())).device


class IActorCritic(IAgentModel):
    @abstractmethod
    def predict_value(self, *obs_tensors: torch.tensor, context: Any) -> torch.tensor:
        pass

    @abstractmethod
    def compute_loss(self,
                     rewards: np.ndarray,
                     dones: np.ndarray,
                     next_value: float,
                     discount_factor: float,
                     **kwargs) -> (torch.tensor, Dict):
        pass
