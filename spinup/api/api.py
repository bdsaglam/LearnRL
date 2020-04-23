import itertools
from abc import ABC, abstractmethod


class IAgent(ABC):
    @abstractmethod
    def act(self, obs, deterministic=False):
        pass

    @abstractmethod
    def reset(self):
        pass


class IActorCritic(ABC):
    @abstractmethod
    def infer_value_action_dist(self, feature_tensor):
        pass

    @abstractmethod
    def infer_value(self, feature_tensor):
        pass

    @abstractmethod
    def infer_action_dist(self, feature_tensor):
        pass

    @abstractmethod
    def critic_parameters(self):
        pass

    @abstractmethod
    def actor_parameters(self):
        pass

    def parameters(self):
        return itertools.chain(self.critic_parameters(), self.actor_parameters())


class IPolicy(ABC):
    @abstractmethod
    def action_dist(self, obs):
        pass
