from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import torch


class IAgent(ABC):
    @abstractmethod
    def act(self, obs, deterministic=False):
        pass

    @abstractmethod
    def reset(self):
        pass


class IAgentModel(IAgent, torch.nn.Module):
    @abstractmethod
    def step(self, obs_tensor: torch.tensor) -> torch.tensor:
        # takes observation tensor and returns action tensor
        pass


class IActorCritic(IAgentModel):
    @abstractmethod
    def predict_value(self, obs_tensor: torch.tensor) -> torch.tensor:
        pass

    @abstractmethod
    def compute_loss(self,
                     rewards: np.ndarray,
                     dones: np.ndarray,
                     next_value: float,
                     discount_factor: float,
                     **kwargs) -> (torch.tensor, Dict):
        pass
